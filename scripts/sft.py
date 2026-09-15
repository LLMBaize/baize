#!/usr/bin/env python3
"""
BaiZe 指令微调（SFT）。

数据格式：jsonl，每行一个多轮对话：
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    支持 system 角色。prompt 部分 label 置 -100，只对回复计算 loss。

用法：
    python scripts/sft.py --data data/sft.jsonl --from_weight pretrain
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baize import BaiZeConfig, BaiZeForCausalLM, BaiZeTokenizer
from baize.trainer_utils import (
    Logger, build_scaler, get_lr, init_distributed_mode, is_main_process,
    load_weights, log_model_params, save_weights, setup_seed,
)


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.samples = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                messages = json.loads(line)["messages"]
                input_ids, labels = tokenizer.encode_chat(messages, max_length)
                if sum(1 for t in labels if t != -100) < 2:
                    skipped += 1
                    continue
                self.samples.append((input_ids, labels))
        Logger(f"SFT 样本: {len(self.samples)} 条（跳过无效 {skipped} 条）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        # 返回 list，由 collate_fn 统一 padding 并创建 tensor
        return self.samples[i]


def collate_fn(batch, pad_id=0):
    maxlen = max(len(x) for x, _ in batch)
    input_ids, labels = [], []
    for x, y in batch:
        pad = maxlen - len(x)
        input_ids.append(x + [pad_id] * pad)
        labels.append(y + [-100] * pad)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="BaiZe SFT")
    parser.add_argument("--data", type=str, default="data/sft.jsonl")
    parser.add_argument("--tokenizer", type=str, default="tokenizer")
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--save_weight", type=str, default="sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="SFT 学习率通常低于预训练")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载进程数；小数据集用 0 更快")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--n_loops_train", type=int, default=None)
    parser.add_argument("--from_weight", type=str, default="pretrain", help="预训练权重名（none=从零）")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = BaiZeTokenizer.from_pretrained(args.tokenizer)

    # ---- 模型：优先从 from_weight 的 config 恢复架构 ----
    config = None
    if args.from_weight != "none":
        wp = f"{args.save_dir}/{args.from_weight}.safetensors"
        if not os.path.exists(wp) and os.path.exists(os.path.splitext(wp)[0] + ".pth"):
            wp = os.path.splitext(wp)[0] + ".pth"
        if os.path.exists(wp) and os.path.exists(os.path.join(args.save_dir, "config.json")):
            config = BaiZeConfig.from_pretrained(args.save_dir)
            config.vocab_size = tokenizer.vocab_size
    if config is None:
        config = BaiZeConfig(vocab_size=tokenizer.vocab_size)
    model = BaiZeForCausalLM(config)
    if args.from_weight != "none":
        wp = f"{args.save_dir}/{args.from_weight}.safetensors"
        if os.path.exists(wp) or os.path.exists(os.path.splitext(wp)[0] + ".pth"):
            n_loaded = load_weights(model, wp)
            Logger(f"从 {wp} 加载 {n_loaded} 个权重")
    if args.use_compile:
        model = torch.compile(model)
    model = model.to(args.device)
    log_model_params(model)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    autocast_ctx = torch.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    scaler = build_scaler(enabled=(args.dtype == "float16"))

    train_ds = SFTDataset(args.data, tokenizer, args.max_seq_len)
    sampler = torch.utils.data.distributed.DistributedSampler(train_ds) if dist.is_initialized() else None
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    iters_per_epoch = math.ceil(len(loader) / args.accumulation_steps)
    total_steps = args.epochs * iters_per_epoch

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)

    start_step = 0
    ckpt_path = f"{args.save_dir}/ckpt_{args.save_weight}.pt"
    if args.from_resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        raw = model.module if isinstance(model, DistributedDataParallel) else model
        raw.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        Logger(f"断点续训: step={start_step}")

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    n_loops = args.n_loops_train or config.max_loop_iters
    global_step = start_step
    start_time = time.time()
    model.train()
    for epoch in range(args.epochs):
        sampler and sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(loader):
            if global_step >= total_steps:
                break
            x, y = x.to(args.device), y.to(args.device)
            lr = get_lr(global_step, total_steps, args.learning_rate)
            for g in optimizer.param_groups:
                g["lr"] = lr
            with autocast_ctx:
                out = model(x, labels=y, n_loops=n_loops)
                loss = (out.loss + out.aux_loss) / args.accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_interval == 0 and is_main_process():
                    spend = time.time() - start_time
                    eta = spend / max(global_step - start_step, 1) * (total_steps - global_step) / 60
                    Logger(f"step:{global_step}/{total_steps} loss:{loss.item() * args.accumulation_steps:.4f} "
                           f"aux:{out.aux_loss.item():.4f} lr:{lr:.2e} eta:{eta:.1f}min")
                if global_step % args.save_interval == 0 and is_main_process():
                    save_ckpt(model, optimizer, scaler, global_step, config, args, ckpt_path)

    if is_main_process():
        save_ckpt(model, optimizer, scaler, global_step, config, args, ckpt_path)
        raw = model.module if isinstance(model, DistributedDataParallel) else model
        raw = getattr(raw, "_orig_mod", raw)
        final = save_weights(raw, f"{args.save_dir}/{args.save_weight}.safetensors")
        config.save_pretrained(args.save_dir)
        Logger(f"SFT 完成，权重保存至 {final}")

    if dist.is_initialized():
        dist.destroy_process_group()


def save_ckpt(model, optimizer, scaler, step, config, args, path):
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    raw = getattr(raw, "_orig_mod", raw)
    torch.save(
        {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": config.to_dict(),
        },
        path,
    )


if __name__ == "__main__":
    main()
