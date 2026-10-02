"""
BaiZe — 本地语料读取
====================
预训练 / 分词器训练 / PPL 评估共用的文档迭代器，支持两种格式：

    *.txt    —— 每个非空行视为一篇文档（原有 toy 语料格式）
    *.jsonl  —— 每行一个 JSON，取 "text" 字段为一篇文档（文档内可含换行），
                scripts/prepare_allenai.py 输出的就是这种格式

max_docs 用来在读取循环中截断数据量（None 表示不限）。
"""

import json


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
