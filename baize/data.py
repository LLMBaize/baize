"""
BaiZe — 本地语料读取
====================
预训练 / 分词器训练 / PPL 评估共用的文档迭代器，支持两种格式：

    *.txt    —— 每个非空行视为一篇文档（原有 toy 语料格式）
    *.jsonl  —— 每行一个 JSON，取 "text" 字段为一篇文档（文档内可含换行），
                scripts/prepare_allenai.py 输出的就是这种格式

另含大规模训练用的预分词 memmap 数据集（TokenBinDataset）与可续训采样器（ResumableSampler）。

max_docs 用来在读取循环中截断数据量（None 表示不限）。
"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def iter_documents(files, max_docs=None, text_field="text"):
    """按顺序遍历多个语料文件，逐篇 yield 文档文本。"""
    n = 0
    for fp in files:
        is_jsonl = fp.endswith(".jsonl") or fp.endswith(".json")
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if max_docs is not None and n >= max_docs:
                    return
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get(text_field, "") if is_jsonl else line
                if not text:
                    continue
                yield text
                n += 1


# ---------------------------------------------------------------------------
# 预分词二进制语料（memmap）
# ---------------------------------------------------------------------------
# scripts/tokenize_corpus.py 把语料一次性分词写成 <prefix>.bin（扁平 token 序列）
# 与 <prefix>.meta.json（dtype、token 数等）。训练时用 np.memmap 按需读取：
# 不占内存、启动零等待、多卡多进程共享同一份页缓存。


def bin_meta_path(bin_path: str) -> str:
    return bin_path[: -len(".bin")] + ".meta.json" if bin_path.endswith(".bin") else bin_path + ".meta.json"


def read_bin_meta(bin_path: str) -> dict:
    with open(bin_meta_path(bin_path), encoding="utf-8") as f:
        return json.load(f)


class TokenBinDataset(Dataset):
    """一个或多个 .bin 文件上的定长样本（样本不跨文件）。

    返回 (input_ids, labels)，两者对齐 —— BaiZeForCausalLM.forward 内部做 next-token shift。
    max_tokens 截断总使用量（按文件顺序累计）。
    """

    def __init__(self, bin_files, seq_len: int, max_tokens=None, vocab_size=None):
        self.seq_len = seq_len
        self.arrays, self.offsets = [], [0]
        used = 0
        for fp in bin_files:
            meta = read_bin_meta(fp)
            if vocab_size is not None and meta.get("vocab_size") not in (None, vocab_size):
                raise ValueError(f"{fp} 由 vocab_size={meta['vocab_size']} 的分词器生成，"
                                 f"与当前分词器 vocab_size={vocab_size} 不一致，请重新分词")
            arr = np.memmap(fp, dtype=np.dtype(meta["dtype"]), mode="r", shape=(meta["n_tokens"],))
            if max_tokens is not None:
                arr = arr[: max(0, max_tokens - used)]
            used += len(arr)
            n = len(arr) // seq_len
            if n > 0:
                self.arrays.append(arr)
                self.offsets.append(self.offsets[-1] + n)
        self.n_tokens = used

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, i):
        f = int(np.searchsorted(self.offsets, i, side="right")) - 1
        start = (i - self.offsets[f]) * self.seq_len
        chunk = torch.from_numpy(self.arrays[f][start: start + self.seq_len].astype(np.int64))
        return chunk, chunk.clone()


# ---------------------------------------------------------------------------
# 可续训的采样器
# ---------------------------------------------------------------------------


class ResumableSampler(Sampler):
    """确定性打乱 + 按 rank 切分 + 断点跳过。

    - 每个 epoch 用 seed + epoch 生成固定排列：单卡也会打乱，且续训后顺序完全一致；
    - 多卡时各 rank 取排列中互不重叠的一份（尾部不足一份的样本丢弃，保证各卡步数一致）；
    - set_epoch(epoch, skip) 跳过本 epoch 本 rank 已消费的前 skip 个样本，续训不重复也不遗漏。
    """

    def __init__(self, n_samples: int, shuffle: bool = True, seed: int = 42, rank: int = 0, world_size: int = 1):
        self.n = n_samples
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.num_samples = n_samples // world_size  # 每个 rank 每 epoch 的样本数
        self.epoch = 0
        self.skip = 0

    def set_epoch(self, epoch: int, skip: int = 0):
        self.epoch = epoch
        self.skip = skip

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(self.n, generator=g).tolist()
        else:
            perm = list(range(self.n))
        mine = perm[self.rank: self.num_samples * self.world_size: self.world_size]
        return iter(mine[self.skip:])

    def __len__(self):
        return max(0, self.num_samples - self.skip)
