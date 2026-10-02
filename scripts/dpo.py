#!/usr/bin/env python3
"""
BaiZe 偏好对齐（DPO, Rafailov et al. 2023）。

数据：jsonl，每行一对偏好样本（prepare_allenai.py --task dpo 的输出即此格式）：
    {"chosen":   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "好的回答"}],
     "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "差的回答"}]}
    也接受 {"prompt": "...", "chosen": "回答", "rejected": "回答"} 的字符串形式。

损失：L = -log σ(β · [(log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x))])
    π 为训练中的策略模型（从 SFT 权重初始化），π_ref 为冻结的同一份 SFT 权重。
    只对最后一条 assistant 回复的 token 求 log 概率（prompt 部分 mask 掉）。

用法：
    python scripts/dpo.py --data data/allenai_dpo.jsonl --from_weight sft --beta 0.1
    torchrun --nproc_per_node=4 scripts/dpo.py --data data/allenai_dpo.jsonl --from_weight sft

注意：参考模型在每个 rank 上完整保存一份（不参与 FSDP 切分），显存约为模型权重的 1 倍。
"""

import argparse
import copy
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baize import BaiZeTokenizer
from baize.trainer import Trainer, add_train_args
from baize.trainer_utils import Logger, log_model_params
from sft import load_base_model


def to_messages(record, side):
    value = record[side]
    if isinstance(value, str):
        return [{"role": "user", "content": record["prompt"]}, {"role": "assistant", "content": value}]
    return value


class DPODataset(Dataset):
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
                    rec = json.loads(line)
                    c_ids, c_lab = tokenizer.encode_chat(to_messages(rec, "chosen"), max_length)
                    r_ids, r_lab = tokenizer.encode_chat(to_messages(rec, "rejected"), max_length)
                    # 截断后回复部分为空的样本无法比较，丢弃
                    if sum(t != -100 for t in c_lab) < 1 or sum(t != -100 for t in r_lab) < 1:
                        skipped += 1
                        continue
                    self.samples.append((c_ids, c_lab, r_ids, r_lab))
        Logger(f"DPO 样本: {len(self.samples)} 对（跳过无效 {skipped} 对）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def dpo_collate(batch, pad_id=0):
    """chosen 与 rejected 拼成一个 [2B, L] 批次，一次前向同时算两者。"""
    seqs = [(c, cl) for c, cl, _, _ in batch] + [(r, rl) for _, _, r, rl in batch]
    maxlen = max(len(x) for x, _ in seqs)
    ids = torch.tensor([x + [pad_id] * (maxlen - len(x)) for x, _ in seqs], dtype=torch.long)
    labels = torch.tensor([y + [-100] * (maxlen - len(y)) for _, y in seqs], dtype=torch.long)
    return ids, labels


def sequence_logps(logits, labels):
    """每条序列在 label 位置上的 log 概率之和（labels 与 input 对齐，内部做 shift）。"""
    logits = logits[:, :-1, :].float()
    labels = labels[:, 1:]
    mask = labels != -100
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)


def dpo_loss_fn(ref_model, beta, n_loops=None):
    def fn(model, batch):
        ids, labels = batch
        out = model(ids, n_loops=n_loops)
        logps = sequence_logps(out.logits, labels)
        with torch.no_grad():
            ref_logps = sequence_logps(ref_model(ids, n_loops=n_loops).logits, labels)
        b = ids.shape[0] // 2
        chosen_reward = beta * (logps[:b] - ref_logps[:b])
        rejected_reward = beta * (logps[b:] - ref_logps[b:])
        margin = chosen_reward - rejected_reward
        loss = -F.logsigmoid(margin).mean()
        logs = {
            "dpo": loss.detach(),
            "acc": (margin > 0).float().mean(),
            "margin": margin.detach().mean(),
            "aux": out.aux_loss.detach(),
        }
        return loss + out.aux_loss, logs
    return fn


def main():
    parser = argparse.ArgumentParser(description="BaiZe DPO")
    d = parser.add_argument_group("数据 / DPO")
    d.add_argument("--data", type=str, default="data/allenai_dpo.jsonl", help="偏好数据 jsonl glob，逗号分隔多个")
    d.add_argument("--tokenizer", type=str, default="tokenizer")
    d.add_argument("--max_samples", type=int, default=None, help="最多使用多少对偏好样本（控制数据量）")
    d.add_argument("--max_seq_len", type=int, default=512)
    d.add_argument("--n_loops_train", type=int, default=None)
    d.add_argument("--beta", type=float, default=0.1, help="DPO 温度 β：越大越贴近参考模型")
    d.add_argument("--save_weight", type=str, default="dpo")
    d.add_argument("--from_weight", type=str, default="sft", help="策略与参考模型的初始权重（通常是 SFT）")
    d.add_argument("--config_dir", type=str, default=None)
    add_train_args(parser, learning_rate=1e-6, epochs=1)
    args = parser.parse_args()
    if args.from_weight == "none":
        raise ValueError("DPO 需要从 SFT 权重开始（--from_weight sft）")

    tokenizer = BaiZeTokenizer.from_pretrained(args.tokenizer)
    model, config = load_base_model(args, tokenizer)
    log_model_params(model)
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    trainer = Trainer(args, model, config, save_weight=args.save_weight)
    ref_model.to(trainer.device)
    files = sorted(f for pat in args.data.split(",") for f in glob.glob(pat.strip()))
    if not files:
        raise FileNotFoundError(f"未找到 DPO 数据: {args.data}")
    ds = DPODataset(files, tokenizer, args.max_seq_len, max_samples=args.max_samples)
    n_loops = args.n_loops_train or config.max_loop_iters
    trainer.fit(ds, dpo_loss_fn(ref_model, args.beta, n_loops), collate_fn=dpo_collate)


if __name__ == "__main__":
    main()
