#!/usr/bin/env python3
"""
BaiZe 指令微调（SFT）。

数据格式：jsonl，每行一个多轮对话：
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    支持 system 角色。prompt 部分 label 置 -100，只对回复计算 loss。

用法：
    python scripts/sft.py --data data/sft.jsonl --from_weight pretrain
    torchrun --nproc_per_node=4 scripts/sft.py --data "data/*.jsonl" --fsdp 1

训练循环（warmup、断点续训、DDP/FSDP、激活重计算）见 baize/trainer.py。
"""

import argparse
import glob
import json
import os
import sys

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baize import BaiZeConfig, BaiZeForCausalLM, BaiZeTokenizer
from baize.trainer import Trainer, add_train_args, lm_loss_fn
from baize.trainer_utils import Logger, load_weights, log_model_params


class SFTDataset(Dataset):
    """读取一个或多个 jsonl 对话文件；max_samples 在读取循环中截断有效样本数。"""

    def __init__(self, files, tokenizer, max_length, max_samples=None):
        self.samples = []
        skipped = 0
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break
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


def load_base_model(args, tokenizer):
    """从 save_dir/config.json + from_weight 恢复模型；from_weight=none 时用默认配置从零训练。"""
    if args.from_weight == "none":
        config = BaiZeConfig(vocab_size=tokenizer.vocab_size)
        return BaiZeForCausalLM(config), config
    wp = os.path.join(args.save_dir, f"{args.from_weight}.safetensors")
    cfg_dir = args.config_dir or args.save_dir
    if not os.path.exists(os.path.join(cfg_dir, "config.json")):
        raise FileNotFoundError(f"{cfg_dir}/config.json 不存在，无法恢复 {wp} 对应的模型结构（可用 --config_dir 指定）")
    config = BaiZeConfig.from_pretrained(cfg_dir)
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(f"分词器 vocab_size={tokenizer.vocab_size} 与预训练模型 {config.vocab_size} 不一致，"
                         f"请使用预训练时的分词器（--tokenizer）")
    model = BaiZeForCausalLM(config)
    n_loaded = load_weights(model, wp)
    Logger(f"从 {wp} 加载 {n_loaded} 个权重")
    return model, config


def main():
    parser = argparse.ArgumentParser(description="BaiZe SFT")
    d = parser.add_argument_group("数据")
    d.add_argument("--data", type=str, default="data/sft.jsonl", help="SFT jsonl 文件 glob，可用逗号分隔多个")
    d.add_argument("--tokenizer", type=str, default="tokenizer")
    d.add_argument("--max_samples", type=int, default=None, help="最多使用多少条有效对话（控制数据量）")
    d.add_argument("--max_seq_len", type=int, default=512)
    d.add_argument("--n_loops_train", type=int, default=None)
    d.add_argument("--save_weight", type=str, default="sft")
    d.add_argument("--from_weight", type=str, default="pretrain", help="预训练权重名（none=从零）")
    d.add_argument("--config_dir", type=str, default=None, help="模型 config.json 所在目录，默认同 save_dir")
    add_train_args(parser, learning_rate=1e-4, epochs=3)
    args = parser.parse_args()

    tokenizer = BaiZeTokenizer.from_pretrained(args.tokenizer)
    model, config = load_base_model(args, tokenizer)
    log_model_params(model)

    trainer = Trainer(args, model, config, save_weight=args.save_weight)
    files = sorted(f for pat in args.data.split(",") for f in glob.glob(pat.strip()))
    if not files:
        raise FileNotFoundError(f"未找到 SFT 数据: {args.data}")
    train_ds = SFTDataset(files, tokenizer, args.max_seq_len, max_samples=args.max_samples)
    trainer.fit(train_ds, lm_loss_fn(args.n_loops_train or config.max_loop_iters), collate_fn=collate_fn)


if __name__ == "__main__":
    main()
