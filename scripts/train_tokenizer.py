#!/usr/bin/env python3
"""
从零训练 BaiZe BPE 分词器。

用法：
    python scripts/train_tokenizer.py --corpus data/corpus.txt --vocab_size 6400

    python scripts/train_tokenizer.py --corpus data/allenai_pretrain.jsonl --max_docs 200000

语料准备：.txt（每行一篇）或 .jsonl（"text" 字段，prepare_allenai.py 的输出），utf-8 编码。
--max_docs 控制参与训练的文档数（大语料只取一部分即可得到稳定的词表）。
小模型建议 vocab_size 3200~6400（词表小 → embedding 占比低，适合参数预算有限的场景）。
"""

import argparse
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baize.data import iter_documents
from baize.tokenizer import train_bpe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/corpus*.txt",
                        help="语料文件 glob，支持 .txt/.jsonl，可用逗号分隔多个")
    parser.add_argument("--max_docs", type=int, default=None, help="最多使用多少篇文档训练分词器")
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--save_dir", type=str, default="tokenizer")
    parser.add_argument("--min_frequency", type=int, default=2)
    args = parser.parse_args()

    files = sorted(f for pat in args.corpus.split(",") for f in glob.glob(pat.strip()))
    if not files:
        raise FileNotFoundError(f"未找到语料文件: {args.corpus}（请先准备纯文本语料）")
    print(f"训练语料: {files}")
    texts = None
    if args.max_docs is not None or any(f.endswith((".jsonl", ".json")) for f in files):
        texts = iter_documents(files, max_docs=args.max_docs)
    tok = train_bpe(files, args.vocab_size, args.save_dir, args.min_frequency, texts=texts)

    # 自检
    sample = "白泽知道天下万物之情理。\nThe quick brown fox jumps over the lazy dog."
    ids = tok.encode(sample)
    print(f"encode: {ids[:20]}{'...' if len(ids) > 20 else ''}")
    print(f"decode: {tok.decode(ids)}")
    print(f"vocab_size = {tok.vocab_size}  ← 训练模型时把 config.vocab_size 设为该值")


if __name__ == "__main__":
    main()
