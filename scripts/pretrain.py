#!/usr/bin/env python3
"""
BaiZe 预训练（next-token LM）。

用法：
    python scripts/pretrain.py --data data/corpus.txt                       # 小语料：直接读文本
    python scripts/pretrain.py --data data/pretrain.bin                     # 大语料：先 tokenize_corpus.py
    torchrun --nproc_per_node=2 scripts/pretrain.py --data data/pretrain.bin
    torchrun --nproc_per_node=8 scripts/pretrain.py --data data/pretrain.bin --fsdp 1 --grad_checkpoint 1

数据：
    .bin    —— scripts/tokenize_corpus.py 预分词的 memmap 文件，不占内存、启动零等待（推荐）
    .txt    —— 每个非空行一篇；.jsonl —— "text" 字段（启动时整体分词进内存，适合小语料）
--max_docs / --max_tokens 控制读入的数据量；训练循环见 baize/trainer.py。
"""

import argparse
import glob
import os
import sys
from array import array

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baize import BaiZeConfig, BaiZeForCausalLM, BaiZeTokenizer
from baize.data import TokenBinDataset, iter_documents
from baize.trainer import Trainer, add_train_args, lm_loss_fn
from baize.trainer_utils import Logger, load_weights, log_model_params


class PretrainDataset(Dataset):
    """把文本语料（.txt/.jsonl）分词进内存并打包成 (input_ids, labels) 定长样本。

    max_docs / max_tokens 在读取循环中截断数据量（None 表示全部读入）。
    token 以 uint32 紧凑存储，1 亿 token 约占 400MB 内存；更大的语料请用 .bin。
    """

    def __init__(self, files, tokenizer, seq_len, max_docs=None, max_tokens=None):
        self.seq_len = seq_len
        ids = array("I")
        n_docs = 0
        for text in iter_documents(files, max_docs=max_docs):
            ids.extend(tokenizer.encode(text))
            ids.append(tokenizer.eos_token_id)
            n_docs += 1
            if max_tokens is not None and len(ids) >= max_tokens:
                del ids[max_tokens:]
                break
        self.data = ids
        self.n_samples = max(1, len(self.data) // seq_len)
        Logger(f"预训练语料: {n_docs:,} 篇 / {len(self.data):,} tokens → "
               f"{self.n_samples:,} 个样本(seq_len={seq_len})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        # 注意：BaiZeForCausalLM.forward 内部会做 next-token shift，
        # 这里 input_ids 与 labels 必须对齐返回，否则会错位两次（变成预测 t+2）
        start = i * self.seq_len
        chunk = self.data[start : start + self.seq_len].tolist()
        n_pad = self.seq_len - len(chunk)
        input_ids = torch.tensor(chunk + [0] * n_pad, dtype=torch.long)
        labels = torch.tensor(chunk + [-100] * n_pad, dtype=torch.long)  # padding 不计 loss
        return input_ids, labels


def build_dataset(args, tokenizer):
    files = sorted(f for pat in args.data.split(",") for f in glob.glob(pat.strip()))
    if not files:
        raise FileNotFoundError(f"未找到语料: {args.data}")
    if all(f.endswith(".bin") for f in files):
        ds = TokenBinDataset(files, args.max_seq_len, max_tokens=args.max_tokens,
                             vocab_size=tokenizer.vocab_size)
        Logger(f"预训练语料(memmap): {ds.n_tokens:,} tokens → {len(ds):,} 个样本(seq_len={args.max_seq_len})")
        return ds
    if any(f.endswith(".bin") for f in files):
        raise ValueError(".bin 与文本语料不能混用，请把文本也用 tokenize_corpus.py 转成 .bin")
    return PretrainDataset(files, tokenizer, args.max_seq_len, max_docs=args.max_docs, max_tokens=args.max_tokens)


def main():
    parser = argparse.ArgumentParser(description="BaiZe Pretrain")
    d = parser.add_argument_group("数据")
    d.add_argument("--data", type=str, default="data/corpus*.txt",
                   help="语料 glob：.bin（推荐）或 .txt/.jsonl，可用逗号分隔多个")
    d.add_argument("--tokenizer", type=str, default="tokenizer")
    d.add_argument("--max_docs", type=int, default=None, help="[文本语料] 最多读入多少篇文档")
    d.add_argument("--max_tokens", type=int, default=None, help="最多使用多少 token（控制数据量）")
    d.add_argument("--max_seq_len", type=int, default=512)
    d.add_argument("--save_weight", type=str, default="pretrain")
    d.add_argument("--from_weight", type=str, default="none", help="初始化权重名（none=从零训练）")
    m = parser.add_argument_group("模型")
    m.add_argument("--hidden_size", type=int, default=512)
    m.add_argument("--num_attention_heads", type=int, default=8)
    m.add_argument("--num_key_value_heads", type=int, default=2)
    m.add_argument("--head_dim", type=int, default=64)
    m.add_argument("--intermediate_size", type=int, default=1024)
    m.add_argument("--prelude_layers", type=int, default=2)
    m.add_argument("--coda_layers", type=int, default=2)
    m.add_argument("--max_loop_iters", type=int, default=8)
    m.add_argument("--n_loops_train", type=int, default=None, help="训练时的循环圈数；默认等于 max_loop_iters")
    m.add_argument("--attn_type", type=str, default="gqa", choices=["gqa", "mla"])
    m.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="Prelude/Coda 也用 MoE（循环块恒为 MoE）")
    m.add_argument("--n_experts", type=int, default=8)
    m.add_argument("--n_experts_per_tok", type=int, default=2)
    m.add_argument("--moe_intermediate_size", type=int, default=512)
    m.add_argument("--moe_capacity_factor", type=float, default=0.0)
    m.add_argument("--use_act", type=int, default=1, choices=[0, 1])
    m.add_argument("--act_init_bias", type=float, default=-3.0)
    m.add_argument("--act_ponder_coef", type=float, default=1e-3)
    add_train_args(parser, learning_rate=5e-4, epochs=2)
    args = parser.parse_args()

    tokenizer = BaiZeTokenizer.from_pretrained(args.tokenizer)
    n_loops_train = args.n_loops_train or args.max_loop_iters
    config = BaiZeConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate_size,
        prelude_layers=args.prelude_layers,
        coda_layers=args.coda_layers,
        max_loop_iters=args.max_loop_iters,
        max_position_embeddings=max(args.max_seq_len * 4, 4096),
        attn_type=args.attn_type,
        use_moe=bool(args.use_moe),
        n_experts=args.n_experts,
        n_experts_per_tok=args.n_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
        moe_capacity_factor=args.moe_capacity_factor,
        use_act=bool(args.use_act),
        act_init_bias=args.act_init_bias,
        act_ponder_coef=args.act_ponder_coef,
    )
    model = BaiZeForCausalLM(config)
    if args.from_weight != "none":
        wp = os.path.join(args.save_dir, f"{args.from_weight}.safetensors")
        n_loaded = load_weights(model, wp)
        Logger(f"从 {wp} 加载 {n_loaded} 个权重初始化")
    log_model_params(model)
    Logger(f"config: loops={config.max_loop_iters}(train {n_loops_train}), prelude={config.prelude_layers}, "
           f"coda={config.coda_layers}, attn={config.attn_type}, moe={config.use_moe}, act={config.use_act}")

    trainer = Trainer(args, model, config, save_weight=args.save_weight)
    dataset = build_dataset(args, tokenizer)
    trainer.fit(dataset, lm_loss_fn(n_loops_train))


if __name__ == "__main__":
    main()
