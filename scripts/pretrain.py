#!/usr/bin/env python3
"""
BaiZe 预训练（next-token LM）。

用法：
    python scripts/pretrain.py --data data/corpus.txt
    torchrun --nproc_per_node=2 scripts/pretrain.py --data data/corpus.txt

数据：纯文本文件，文档以空行分隔；内部打包为固定长度 seq_len 的样本。
"""

import argparse
import glob
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


class PretrainDataset(Dataset):
    """把语料打包成 (input_ids, labels) 定长样本。"""

    def __init__(self, files, tokenizer, seq_len):
        self.seq_len = seq_len
        ids = []
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ids.extend(tokenizer.encode(line))
                        ids.append(tokenizer.eos_token_id)
        self.data = ids
        self.n_samples = max(1, len(self.data) // seq_len)
        Logger(f"预训练语料: {len(self.data):,} tokens → {self.n_samples:,} 个样本(seq_len={seq_len})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        # 注意：BaiZeForCausalLM.forward 内部会做 next-token shift，
        # 这里 input_ids 与 labels 必须对齐返回，否则会错位两次（变成预测 t+2）
        start = i * self.seq_len
        chunk = self.data[start : start + self.seq_len]
        n_pad = self.seq_len - len(chunk)
        input_ids = torch.tensor(chunk + [0] * n_pad, dtype=torch.long)
        labels = torch.tensor(chunk + [-100] * n_pad, dtype=torch.long)  # padding 不计 loss
        return input_ids, labels


def main():
    parser = argparse.ArgumentParser(description="BaiZe Pretrain")
    parser.add_argument("--data", type=str, default="data/corpus*.txt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer")
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--save_weight", type=str, default="pretrain")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载进程数；小数据集用 0 更快")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=200)
    # ---- 模型配置 ----
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--prelude_layers", type=int, default=2)
    parser.add_argument("--coda_layers", type=int, default=2)
    parser.add_argument("--max_loop_iters", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--attn_type", type=str, default="gqa", choices=["gqa", "mla"])
    parser.add_argument("--use_act", type=int, default=1, choices=[0, 1])
    parser.add_argument("--n_loops_train", type=int, default=None,
                        help="训练时的循环圈数；默认等于 max_loop_iters")
    parser.add_argument("--from_weight", type=str, default="none", help="初始化权重名（none=从零训练）")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="断点续训")
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = BaiZeTokenizer.from_pretrained(args.tokenizer)
    n_loops_train = args.n_loops_train or args.max_loop_iters
    config = BaiZeConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        prelude_layers=args.prelude_layers,
        coda_layers=args.coda_layers,
        max_loop_iters=args.max_loop_iters,
        max_position_embeddings=max(args.max_seq_len * 4, 4096),
        use_moe=bool(args.use_moe),
        n_experts=args.n_experts,
        attn_type=args.attn_type,
        use_act=bool(args.use_act),
    )
    model = BaiZeForCausalLM(config)
    if args.from_weight != "none":
        wp = f"{args.save_dir}/{args.from_weight}.safetensors"
        if os.path.exists(wp) or os.path.exists(os.path.splitext(wp)[0] + ".pth"):
            n_loaded = load_weights(model, wp)
            Logger(f"从 {wp} 加载 {n_loaded} 个权重初始化")
    if args.use_compile:
        model = torch.compile(model)
    model = model.to(args.device)
    log_model_params(model)
    Logger(f"config: loops={config.max_loop_iters}(train {n_loops_train}), "
           f"prelude={config.prelude_layers}, coda={config.coda_layers}, "
           f"attn={config.attn_type}, moe={config.use_moe}, act={config.use_act}")

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    autocast_ctx = torch.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    scaler = build_scaler(enabled=(args.dtype == "float16"))

    files = sorted(glob.glob(args.data))
    if not files:
        raise FileNotFoundError(f"未找到语料: {args.data}")
    train_ds = PretrainDataset(files, tokenizer, args.max_seq_len)
    sampler = torch.utils.data.distributed.DistributedSampler(train_ds) if dist.is_initialized() else None
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    iters_per_epoch = math.ceil(len(loader) / args.accumulation_steps)
    total_steps = args.epochs * iters_per_epoch

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)

    # ---- 断点续训 ----
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
                out = model(x, labels=y, n_loops=n_loops_train)
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
                    raw = model.module if isinstance(model, DistributedDataParallel) else model
                    rho = raw.model.recurrent.injection.get_A().max().item()
                    Logger(f"step:{global_step}/{total_steps} loss:{loss.item() * args.accumulation_steps:.4f} "
                           f"aux:{out.aux_loss.item():.4f} lr:{lr:.2e} ρ(A):{rho:.3f} eta:{eta:.1f}min")
                if global_step % args.save_interval == 0 and is_main_process():
                    save_ckpt(model, optimizer, scaler, global_step, config, args, ckpt_path)

    if is_main_process():
        save_ckpt(model, optimizer, scaler, global_step, config, args, ckpt_path)
        raw = model.module if isinstance(model, DistributedDataParallel) else model
        raw = getattr(raw, "_orig_mod", raw)
        final = save_weights(raw, f"{args.save_dir}/{args.save_weight}.safetensors")
        config.save_pretrained(args.save_dir)
        Logger(f"训练完成，权重保存至 {final}")

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
