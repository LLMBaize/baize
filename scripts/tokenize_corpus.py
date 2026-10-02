#!/usr/bin/env python3
"""
把预训练语料一次性分词，写成 memmap 二进制文件，供 pretrain.py 零拷贝读取。

    python scripts/tokenize_corpus.py --data data/allenai_pretrain.jsonl \
        --tokenizer tokenizer --out data/pretrain.bin --max_tokens 2_000_000_000

输出：
    data/pretrain.bin        扁平的 token 序列（每篇文档后接 eos），uint16（词表 < 65536）或 uint32
    data/pretrain.meta.json  dtype、n_tokens、n_docs、vocab_size 等元信息

分词用 tokenizers 的 encode_batch（Rust 多线程），边读边写，内存占用与语料大小无关。
训练时：python scripts/pretrain.py --data data/pretrain.bin
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baize import BaiZeTokenizer
from baize.data import bin_meta_path, iter_documents


def tokenize_to_bin(files, tokenizer, out_path, max_docs=None, max_tokens=None, batch_docs=1000, log_every=100_000):
    dtype = np.uint16 if tokenizer.vocab_size < 2**16 else np.uint32
    eos = tokenizer.eos_token_id
    n_tokens = n_docs = 0
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp_path = out_path + ".tmp"

    def flush(batch, f):
        nonlocal n_tokens, n_docs
        encs = tokenizer.tokenizer.encode_batch(batch, add_special_tokens=False)
        ids = []
        for enc in encs:
            ids.extend(enc.ids)
            ids.append(eos)
        if max_tokens is not None:
            ids = ids[: max(0, max_tokens - n_tokens)]
        np.asarray(ids, dtype=dtype).tofile(f)
        n_tokens += len(ids)
        n_docs += len(batch)

    with open(tmp_path, "wb") as f:
        batch = []
        last_log = 0
        for text in iter_documents(files, max_docs=max_docs):
            batch.append(text)
            if len(batch) >= batch_docs:
                flush(batch, f)
                batch = []
                if max_tokens is not None and n_tokens >= max_tokens:
                    break
                if n_docs - last_log >= log_every:
                    last_log = n_docs
                    rate = n_tokens / max(time.time() - t0, 1e-6)
                    print(f"[{n_docs:,} 篇] {n_tokens:,} tokens  {rate / 1e6:.2f}M tok/s", flush=True)
        if batch and (max_tokens is None or n_tokens < max_tokens):
            flush(batch, f)
    os.replace(tmp_path, out_path)

    meta = dict(dtype=np.dtype(dtype).name, n_tokens=n_tokens, n_docs=n_docs,
                vocab_size=tokenizer.vocab_size, eos_token_id=eos, sources=list(files),
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(bin_meta_path(out_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"完成：{n_docs:,} 篇 / {n_tokens:,} tokens → {out_path}"
          f"（{os.path.getsize(out_path) / 2**30:.2f} GiB，{time.time() - t0:.0f}s）")
    return meta


def main():
    p = argparse.ArgumentParser(description="预分词语料 → memmap .bin")
    p.add_argument("--data", required=True, help="语料 glob（.txt / .jsonl），逗号分隔多个")
    p.add_argument("--tokenizer", default="tokenizer")
    p.add_argument("--out", default="data/pretrain.bin")
    p.add_argument("--max_docs", type=int, default=None, help="最多分词多少篇")
    p.add_argument("--max_tokens", type=int, default=None, help="最多写出多少 token")
    p.add_argument("--batch_docs", type=int, default=1000, help="每批送入 encode_batch 的文档数")
    args = p.parse_args()
    files = sorted(f for pat in args.data.split(",") for f in glob.glob(pat.strip()))
    if not files:
        raise FileNotFoundError(f"未找到语料: {args.data}")
    tok = BaiZeTokenizer.from_pretrained(args.tokenizer)
    tokenize_to_bin(files, tok, args.out, args.max_docs, args.max_tokens, args.batch_docs)


if __name__ == "__main__":
    main()
