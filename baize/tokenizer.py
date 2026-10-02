"""
BaiZe — Tokenizer
=================
两部分：
1. train_bpe()    —— 用 HuggingFace `tokenizers` 库在自有语料上从头训练 BPE，
                     不依赖任何外部词表，词表大小完全由语料决定。
2. BaiZeTokenizer —— 统一封装（encode/decode/chat template），
                     from_pretrained 时自动读出真实词表大小，
                     供 BaiZeConfig(vocab_size=...) 使用，保证两端严格一致。

特殊符号（与训练脚本约定一致）：
    <unk>=0  <s>=1  </s>=2  <im_start>=3  <im_end>=4

v2 改动：encode_chat 的 prompt 边界定位
    原版用 ±1 硬编码探测 ByteLevel BPE 的边界偏差，有残留漏洞。
    v2 改为扫描 <im_start>assistant 的 token id 序列作为定界符，
    找到最后一次出现位置后加上该序列长度，定位精确且与 BPE 合并无关。
"""

import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<im_start>", "<im_end>"]
UNK_ID, BOS_ID, EOS_ID, IM_START_ID, IM_END_ID = 0, 1, 2, 3, 4


def train_bpe(
    corpus_files,
    vocab_size: int = 6400,
    save_dir: str = "tokenizer",
    min_frequency: int = 2,
    texts=None,
):
    """在语料上训练 BPE 分词器并保存。

    Args:
        corpus_files: 文本文件路径列表（utf-8，纯文本，一行或多行均可）
        texts:        可选，文档文本迭代器；给出时忽略 corpus_files，
                      用于 jsonl 语料或只取部分数据训练
        vocab_size:   目标词表大小（含特殊符号）
        save_dir:     输出目录，写入 tokenizer.json 与 tokenizer_config.json
        min_frequency: BPE 合并的最小频次
    Returns:
        BaiZeTokenizer
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        # 256 个字节级基础 token 全部保留，保证任何输入都能无损 roundtrip
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    if texts is not None:
        tokenizer.train_from_iterator(texts, trainer)
    else:
        tokenizer.train(corpus_files, trainer)

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": tokenizer.get_vocab_size(),
                "special_tokens": SPECIAL_TOKENS,
                "chat_template": "im_start",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[tokenizer] vocab_size={tokenizer.get_vocab_size()} -> {save_dir}")
    return BaiZeTokenizer(save_dir)


class BaiZeTokenizer:
    """BaiZe 分词器封装。

    同时提供：
        - 预训练用的纯文本 encode/decode
        - SFT 用的对话模板（<im_start>role\\n...<im_end>）
        - vocab_size 属性（配置模型时务必使用它，保证两端一致）
    """

    def __init__(self, tokenizer_dir: str):
        self.tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=10**9)  # 关闭长度限制，由训练侧控制
        self.tokenizer.no_padding()
        self._vocab_size = self.tokenizer.get_vocab_size()
        # 预计算 <im_start>assistant\n 的 token id 序列，用于 prompt 边界定位
        self._assistant_prefix_ids = self.encode("<im_start>assistant\n")

    # ---- 工厂方法 ----
    @classmethod
    def from_pretrained(cls, path: str):
        return cls(path)

    # ---- 基本属性 ----
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_token_id(self) -> int:
        return BOS_ID

    @property
    def eos_token_id(self) -> int:
        return EOS_ID

    @property
    def im_start_id(self) -> int:
        return IM_START_ID

    @property
    def im_end_id(self) -> int:
        return IM_END_ID

    # ---- 预训练接口 ----
    def encode(self, text: str, add_special_tokens: bool = False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    # ---- SFT 对话接口 ----
    def build_chat(self, messages, add_generation_prompt: bool = False):
        """把 [{'role': ..., 'content': ...}, ...] 渲染成对话文本。"""
        text = "<s>"
        for m in messages:
            text += f"<im_start>{m['role']}\n{m['content']}<im_end>"
        if add_generation_prompt:
            text += "<im_start>assistant\n"
        return text

    def _find_last_subseq(self, ids: list, subseq: list) -> int:
        """在 ids 中找 subseq 最后一次出现的结束位置（不含）。
        返回结束索引；找不到时返回 0。

        v2 新增：替代原版 ±1 硬编码探测，定位 <im_start>assistant\\n 序列
        的末尾作为 prompt 边界，与 BPE 合并策略无关。
        """
        n, m = len(ids), len(subseq)
        last = 0
        for i in range(n - m + 1):
            if ids[i: i + m] == subseq:
                last = i + m
        return last

    def encode_chat(self, messages, max_length: int, add_generation_prompt: bool = False):
        """编码对话，返回 (input_ids, labels)。

        labels 中 prompt 部分（最后一个 assistant 回复之前）置为 -100，
        只对 assistant 的回复计算 loss。

        v2 边界定位改动：
            扫描 <im_start>assistant\\n 的 token id 序列在全文中最后出现的位置，
            以此作为 prompt 结束点，比原版 ±1 枚举更准确。
        """
        full_text = self.build_chat(messages)
        full_ids = self.encode(full_text)

        # 精确定位最后一个 assistant turn 的起点（即 prompt 结束点）
        prompt_end = self._find_last_subseq(full_ids, self._assistant_prefix_ids)
        # 如果最后消息是 assistant，prompt_end 已含前缀；否则用整段作 prompt
        if prompt_end == 0:
            # fallback：整段都是 prompt（不应发生，但保底）
            prompt_end = len(full_ids)

        input_ids = full_ids[:max_length]
        labels = list(input_ids)
        for i in range(min(prompt_end, len(labels))):
            labels[i] = -100

        # 确保末尾有 eos
        if input_ids and input_ids[-1] != EOS_ID and len(input_ids) < max_length:
            input_ids = input_ids + [EOS_ID]
            labels = labels + [EOS_ID]

        return input_ids, labels
