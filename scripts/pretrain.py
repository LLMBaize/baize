#!/usr/bin/env python3
"""
BaiZe 预训练（next-token LM）—— 统一入口。

训练后端由 --backend 选择：
    native    原版训练循环（baize/trainer.py，单卡 / DDP / FSDP2）      默认
    megatron  Megatron-core 训练循环（baize/mcore/train.py，TP / EP / PP / MLA / 分布式优化器）

用法：
    python scripts/pretrain.py --data data/corpus.txt                       # 小语料：直接读文本
    python scripts/pretrain.py --data data/pretrain.bin                     # 大语料：先 tokenize_corpus.py
    torchrun --nproc_per_node=2 scripts/pretrain.py --data data/pretrain.bin
    torchrun --nproc_per_node=8 scripts/pretrain.py --data data/pretrain.bin --fsdp 1 --grad_checkpoint 1
    torchrun --nproc_per_node=8 scripts/pretrain.py --backend megatron --data data/pretrain.bin \\
        --tensor_parallel_size 2 --use_distributed_optimizer 1

数据：
    .bin    —— scripts/tokenize_corpus.py 预分词的 memmap 文件，不占内存、启动零等待（推荐）
    .txt    —— 每个非空行一篇；.jsonl —— "text" 字段（启动时整体分词进内存，适合小语料）
--max_docs / --max_tokens 控制读入的数据量；原版训练循环见 baize/trainer.py。
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baize import BaiZeForCausalLM, BaiZeTokenizer  # noqa: E402
from baize.mcore.args import add_megatron_args  # noqa: E402
from baize.pretrain_common import (  # noqa: E402
    PretrainDataset, add_data_model_args, build_dataset, config_from_args,
)
from baize.trainer import Trainer, add_train_args, lm_loss_fn  # noqa: E402
from baize.trainer_utils import Logger, load_weights, log_model_params  # noqa: E402

__all__ = ["PretrainDataset", "add_data_model_args", "build_dataset", "config_from_args", "main"]


def run_native(args):
    tokenizer = BaiZeTokenizer.from_pretrained(args.tokenizer)
    config = config_from_args(args, tokenizer.vocab_size)
    n_loops_train = config.n_loops_train
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


def main():
    parser = argparse.ArgumentParser(description="BaiZe Pretrain")
    parser.add_argument("--backend", type=str, default="native", choices=["native", "megatron"],
                        help="训练后端：native（原版循环）或 megatron（Megatron-core，需 pip install megatron-core）")
    add_data_model_args(parser)
    add_train_args(parser, learning_rate=5e-4, epochs=2)
    add_megatron_args(parser)  # 只对 --backend megatron 生效
    args = parser.parse_args()

    if args.backend == "megatron":
        from baize.mcore.train import run_megatron  # 延迟导入：原版后端不需要 megatron-core
        run_megatron(args)
    else:
        run_native(args)


if __name__ == "__main__":
    main()
