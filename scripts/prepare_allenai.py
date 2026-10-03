#!/usr/bin/env python3
"""
从 HuggingFace 流式拉取 AllenAI 的预训练 / 后训练数据集，按比例混合、过滤、
控制数据量后写成本地 jsonl，供 pretrain.py / sft.py / train_tokenizer.py 直接使用。

流式读取（streaming=True）：不会把整个数据集下载到本地，读够配额即停止，
所以即便是 TB 级的 C4 / OLMo-mix，也只会下载实际用到的那部分分片。

用法示例（详见 docs/allenai_data.md）：

    # 预训练：中文 mC4 70% + 英文 C4 30%，总共 2 亿 token
    python scripts/prepare_allenai.py --task pretrain \
        --sources c4-zh:0.7,c4-en:0.3 --max_tokens 200_000_000 \
        --tokenizer tokenizer --out data/allenai_pretrain.jsonl

    # 后训练（SFT）：Tulu 3 SFT mixture 抽 5 万条
    python scripts/prepare_allenai.py --task sft \
        --sources tulu3 --max_docs 50000 --out data/allenai_sft.jsonl

    # 偏好对齐（DPO）：Tulu 3 偏好数据 2 万对
    python scripts/prepare_allenai.py --task dpo \
        --sources tulu3-pref --max_docs 20000 --out data/allenai_dpo.jsonl

    # 查看所有内置数据源
    python scripts/prepare_allenai.py --list
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# 内置数据源
#   path       —— HF 数据集仓库
#   name       —— 数据集 config（子集）名，可选
#   data_files —— 仓库内文件 glob，可选（用它可只拉指定语言/分片）
#   split      —— 默认 train
# 不在列表中的数据集可直接写 "仓库名#config" 或 "仓库名@文件glob"，见 parse_sources。
# ---------------------------------------------------------------------------

PRETRAIN_PRESETS = {
    # ---- C4 / mC4（allenai/c4，ODC-BY）----
    "c4-en": dict(path="allenai/c4", data_files="en/c4-train.*.json.gz",
                  desc="C4 英文清洗版 Common Crawl（~3.65 亿篇，~156B token）"),
    "c4-zh": dict(path="allenai/c4", data_files="multilingual/c4-zh.*.json.gz",
                  desc="mC4 中文（allenai/c4 的 multilingual 目录）"),
    "c4-realnewslike": dict(path="allenai/c4", data_files="realnewslike/c4-train.*.json.gz",
                            desc="C4 新闻类子集（~1300 万篇）"),
    "c4-en-val": dict(path="allenai/c4", data_files="en/c4-validation.*.json.gz",
                      desc="C4 英文验证集（做 PPL 评估用）"),
    "c4-zh-val": dict(path="allenai/c4", data_files="multilingual/c4-zh-validation.*.json.gz",
                      desc="mC4 中文验证集（做 PPL 评估用）"),
    # ---- OLMo 2 预训练混合（allenai/olmo-mix-1124，ODC-BY）----
    "olmo-mix-dclm": dict(path="allenai/olmo-mix-1124", name="dclm",
                          desc="OLMo 2 预训练主体：DCLM 高质量网页（英文）"),
    "olmo-mix-wiki": dict(path="allenai/olmo-mix-1124", name="wiki",
                          desc="OLMo 2 预训练：Wikipedia + Wikibooks"),
    "olmo-mix-pes2o": dict(path="allenai/olmo-mix-1124", name="pes2o",
                           desc="OLMo 2 预训练：peS2o 学术论文"),
    "olmo-mix-arxiv": dict(path="allenai/olmo-mix-1124", name="arxiv",
                           desc="OLMo 2 预训练：arXiv"),
    "olmo-mix-starcoder": dict(path="allenai/olmo-mix-1124", name="starcoder",
                               desc="OLMo 2 预训练：代码"),
    "olmo-mix-open-web-math": dict(path="allenai/olmo-mix-1124", name="open-web-math",
                                   desc="OLMo 2 预训练：数学网页"),
    # ---- OLMo 2 中期训练（退火）高质量混合（allenai/dolmino-mix-1124）----
    "dolmino-wiki": dict(path="allenai/dolmino-mix-1124", name="wiki",
                         desc="Dolmino：Wikipedia"),
    "dolmino-flan": dict(path="allenai/dolmino-mix-1124", name="flan",
                         desc="Dolmino：FLAN 指令改写文本"),
    "dolmino-math": dict(path="allenai/dolmino-mix-1124", name="math",
                         desc="Dolmino：数学合成/精选数据"),
    "dolmino-stackexchange": dict(path="allenai/dolmino-mix-1124", name="stackexchange",
                                  desc="Dolmino：StackExchange 问答"),
}

SFT_PRESETS = {
    # ---- Tulu 3 后训练数据（ODC-BY，子集各有许可）----
    "tulu3": dict(path="allenai/tulu-3-sft-mixture",
                  desc="Tulu 3 SFT 混合（~94 万条，多源，含 source 字段）"),
    "tulu3-olmo2": dict(path="allenai/tulu-3-sft-olmo-2-mixture",
                        desc="OLMo 2 Instruct 使用的 SFT 混合"),
    "tulu3-personas-math": dict(path="allenai/tulu-3-sft-personas-math",
                                desc="Tulu 3 persona 合成数学题"),
    "tulu3-personas-code": dict(path="allenai/tulu-3-sft-personas-code",
                                desc="Tulu 3 persona 合成代码题"),
    "tulu3-personas-if": dict(path="allenai/tulu-3-sft-personas-instruction-following",
                              desc="Tulu 3 persona 精确指令遵循"),
    # ---- 真实用户对话（含大量中文；可用 --language Chinese 过滤）----
    "wildchat": dict(path="allenai/WildChat-1M", messages_field="conversation",
                     desc="WildChat-1M 真实用户与 ChatGPT 对话（有 language 字段）"),
}

DPO_PRESETS = {
    # ---- 偏好数据（chosen / rejected 两条完整对话）----
    "tulu3-pref": dict(path="allenai/llama-3.1-tulu-3-8b-preference-mixture",
                       desc="Tulu 3 偏好混合（Llama-3.1-Tulu-3-8B DPO 用，~27 万对）"),
    "tulu3-pref-olmo2": dict(path="allenai/olmo-2-1124-7b-preference-mix",
                             desc="OLMo 2 7B DPO 使用的偏好混合"),
    "ultrafeedback": dict(path="allenai/ultrafeedback_binarized_cleaned", split="train_prefs",
                          desc="UltraFeedback 二值化清洗版（~6 万对）"),
}

VALID_ROLES = {"system", "user", "assistant"}


# ---------------------------------------------------------------------------
# 数据源解析
# ---------------------------------------------------------------------------


def parse_sources(spec: str, presets: dict):
    """解析 "名称[:权重],名称[:权重],..."。

    名称可以是内置预设，也可以是自定义 HF 数据集：
        allenai/dolma3_mix#config        —— 指定 config
        allenai/c4@multilingual/c4-ja.*.json.gz  —— 指定文件 glob
    """
    sources = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        weight = 1.0
        head, sep, tail = item.rpartition(":")
        if sep:
            try:
                weight = float(tail)
                item = head
            except ValueError:
                pass
        if weight <= 0:
            raise ValueError(f"权重必须 > 0: {item}")
        if item in presets:
            src = dict(presets[item])
        elif "/" in item:
            src = {}
            path = item
            if "@" in path:
                path, src["data_files"] = path.split("@", 1)
            if "#" in path:
                path, src["name"] = path.split("#", 1)
            src["path"] = path
        else:
            raise ValueError(f"未知数据源 '{item}'，用 --list 查看内置预设，或写完整的 HF 仓库名")
        src["key"] = item
        src["weight"] = weight
        sources.append(src)
    if not sources:
        raise ValueError("--sources 为空")
    return sources


NETWORK_HINT = """        无法连接 HuggingFace。可按顺序尝试（详见 docs/allenai_data.md「网络问题」）：
          1) 使用镜像：加参数 --hf_endpoint https://hf-mirror.com（或 export HF_ENDPOINT=...，须在启动 Python 前 export）
          2) 报 'Network is unreachable'（errno 101）多为服务器无 IPv6 路由却解析到了 IPv6 地址：
             在 /etc/gai.conf 加一行 'precedence ::ffff:0:0/96 100' 让系统优先走 IPv4
          3) 走代理：export HTTPS_PROXY=http://<代理地址>:<端口>
          4) 离线：先用 huggingface-cli download 把文件下到本地，再用 --sources "json@<本地glob>" 读取"""


def is_network_error(exc) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    keys = ("network is unreachable", "connection", "timed out", "timeout", "name resolution",
            "temporary failure", "max retries", "errno 101", "errno 110", "errno 111", "ssl")
    return any(k in text for k in keys)


def open_stream(src, seed, shuffle_buffer, skip):
    """以 streaming 方式打开一个 HF 数据集，返回样本迭代器。"""
    from datasets import load_dataset

    kwargs = dict(split=src.get("split", "train"), streaming=True)
    if src.get("data_files"):
        kwargs["data_files"] = src["data_files"]
    ds = load_dataset(src["path"], src.get("name"), **kwargs)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if skip > 0:
        ds = ds.skip(skip)
    return iter(ds)


# ---------------------------------------------------------------------------
# 样本清洗
# ---------------------------------------------------------------------------


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return cjk / len(text)


def clean_pretrain(example, src, args):
    """返回清洗后的文本，或 None（被过滤）。"""
    text = example.get(src.get("text_field", args.text_field))
    if not isinstance(text, str):
        return None
    text = text.strip()
    if len(text) < args.min_chars:
        return None
    if args.max_doc_chars > 0:
        text = text[: args.max_doc_chars]
    if args.min_cjk_ratio > 0 and cjk_ratio(text) < args.min_cjk_ratio:
        return None
    return text


def normalize_messages(raw):
    """规范化对话：只保留 system/user/assistant，去掉末尾非 assistant 轮次；不合法返回 None。"""
    if not isinstance(raw, list):
        return None
    messages = []
    for m in raw:
        if not isinstance(m, dict):
            return None
        role, content = m.get("role"), m.get("content")
        if role not in VALID_ROLES or not isinstance(content, str) or not content.strip():
            return None  # 含工具调用等非标准轮次的对话整条丢弃
        messages.append({"role": role, "content": content.strip()})
    # 只对最后一个 assistant 回复计算 loss，末尾的非 assistant 轮次去掉
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not any(m["role"] == "user" for m in messages):
        return None
    return messages


def _meta_filters_pass(example, args):
    if args.language and "language" in example and example["language"] != args.language:
        return False
    if example.get("toxic") is True:  # WildChat 自带的毒性标注
        return False
    if args.source_filter:
        source = str(example.get("source", ""))
        if not any(s in source for s in args.source_filter):
            return False
    return True


def clean_sft(example, src, args):
    """返回规范化的 messages 列表，或 None（被过滤）。"""
    if not _meta_filters_pass(example, args):
        return None
    messages = normalize_messages(example.get(src.get("messages_field", args.messages_field)))
    if messages is None:
        return None
    if args.max_turns > 0 and sum(m["role"] == "assistant" for m in messages) > args.max_turns:
        return None
    if args.min_cjk_ratio > 0:
        if cjk_ratio("".join(m["content"] for m in messages)) < args.min_cjk_ratio:
            return None
    return messages


def clean_dpo(example, src, args):
    """返回 {"chosen": messages, "rejected": messages}，两者共享相同的 prompt 前缀；或 None。

    兼容两种格式：chosen/rejected 为完整对话列表（Tulu 3 / UltraFeedback），
    或 prompt 为字符串、chosen/rejected 为回复字符串。
    """
    if not _meta_filters_pass(example, args):
        return None
    pair = {}
    for side in ("chosen", "rejected"):
        raw = example.get(side)
        if isinstance(raw, str):
            prompt = example.get("prompt")
            if not isinstance(prompt, str):
                return None
            raw = [{"role": "user", "content": prompt}, {"role": "assistant", "content": raw}]
        msgs = normalize_messages(raw)
        if msgs is None:
            return None
        pair[side] = msgs
    if pair["chosen"][:-1] != pair["rejected"][:-1]:  # prompt 必须一致，只有最后一条回复不同
        return None
    if pair["chosen"][-1]["content"] == pair["rejected"][-1]["content"]:
        return None
    if args.max_turns > 0 and sum(m["role"] == "assistant" for m in pair["chosen"]) > args.max_turns:
        return None
    if args.min_cjk_ratio > 0:
        if cjk_ratio("".join(m["content"] for m in pair["chosen"])) < args.min_cjk_ratio:
            return None
    return pair


# ---------------------------------------------------------------------------
# 主流程：按权重交错采样各数据源，直到配额用完
# ---------------------------------------------------------------------------


def split_budget(total, sources):
    """把总配额按权重分给各数据源；total 为 None 表示不限。"""
    if total is None:
        return [None] * len(sources)
    wsum = sum(s["weight"] for s in sources)
    return [int(total * s["weight"] / wsum) for s in sources]


def prepare(args, stream_fn=open_stream):
    presets = {"pretrain": PRETRAIN_PRESETS, "sft": SFT_PRESETS, "dpo": DPO_PRESETS}[args.task]
    sources = parse_sources(args.sources, presets)
    if args.max_tokens is not None and not args.tokenizer:
        raise ValueError("--max_tokens 需要 --tokenizer 来统计 token 数（或改用 --max_docs / --max_chars）")
    if args.max_docs is None and args.max_tokens is None and args.max_chars is None:
        raise ValueError("请至少指定一个数据量上限：--max_docs / --max_tokens / --max_chars")

    tokenizer = None
    if args.tokenizer:
        from baize import BaiZeTokenizer
        tokenizer = BaiZeTokenizer.from_pretrained(args.tokenizer)

    quota = {
        "docs": split_budget(args.max_docs, sources),
        "tokens": split_budget(args.max_tokens, sources),
        "chars": split_budget(args.max_chars, sources),
    }
    stats = [dict(source=s["key"], hf_path=s["path"], weight=s["weight"],
                  docs=0, tokens=0, chars=0, scanned=0, filtered=0, duplicates=0,
                  exhausted=False, error=None) for s in sources]
    streams = [None] * len(sources)

    def is_done(i):
        st = stats[i]
        if st["exhausted"]:
            return True
        for key in ("docs", "tokens", "chars"):
            q = quota[key][i]
            if q is not None and st[key] >= q:
                return True
        return False

    rng = random.Random(args.seed)
    seen = set()
    clean = {"pretrain": clean_pretrain, "sft": clean_sft, "dpo": clean_dpo}[args.task]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    val_path = os.path.splitext(args.out)[0] + ".val.jsonl"
    n_val = 0
    t0 = time.time()
    n_written = 0
    fout = open(args.out, "w", encoding="utf-8")
    fval = open(val_path, "w", encoding="utf-8") if args.val_docs > 0 else None
    try:
        while True:
            active = [i for i in range(len(sources)) if not is_done(i)]
            if not active:
                break
            i = rng.choices(active, weights=[sources[j]["weight"] for j in active])[0]
            src, st = sources[i], stats[i]

            # ---- 取下一条原始样本 ----
            try:
                if streams[i] is None:
                    print(f"[open] {src['key']} ← {src['path']}"
                          f"{'#' + src['name'] if src.get('name') else ''}"
                          f"{'@' + src['data_files'] if src.get('data_files') else ''}", flush=True)
                    streams[i] = stream_fn(src, args.seed, args.shuffle_buffer, args.skip)
                example = next(streams[i])
            except StopIteration:
                st["exhausted"] = True
                print(f"[done] {src['key']} 数据已读完", flush=True)
                continue
            except Exception as exc:  # 网络错误 / config 名错误等：停掉该源，保留已写数据
                st["exhausted"], st["error"] = True, f"{type(exc).__name__}: {exc}"
                print(f"[error] {src['key']}: {st['error']}", flush=True)
                if is_network_error(exc):
                    print(NETWORK_HINT, flush=True)
                else:
                    print(f"        请到 https://huggingface.co/datasets/{src['path']} 确认 config/文件名，"
                          f"或改用 '仓库#config' / '仓库@文件glob' 自定义写法", flush=True)
                continue

            st["scanned"] += 1
            if args.max_scan > 0 and st["scanned"] >= args.max_scan:
                st["exhausted"] = True
                print(f"[stop] {src['key']} 已扫描 {st['scanned']} 条（--max_scan），停止", flush=True)

            # ---- 清洗 / 去重 ----
            item = clean(example, src, args)
            if item is None:
                st["filtered"] += 1
                continue
            if args.dedup:
                h = hashlib.md5(json.dumps(item, ensure_ascii=False).encode()).digest()
                if h in seen:
                    st["duplicates"] += 1
                    continue
                seen.add(h)

            if args.task == "pretrain":
                record = {"text": item, "source": src["key"]}
                n_chars = len(item)
                n_tokens = len(tokenizer.encode(item)) + 1 if tokenizer else 0
            elif args.task == "sft":
                record = {"messages": item, "source": src["key"]}
                n_chars = sum(len(m["content"]) for m in item)
                n_tokens = len(tokenizer.encode(tokenizer.build_chat(item))) if tokenizer else 0
            else:
                record = {"chosen": item["chosen"], "rejected": item["rejected"], "source": src["key"]}
                n_chars = sum(len(m["content"]) for m in item["chosen"]) + len(item["rejected"][-1]["content"])
                n_tokens = (len(tokenizer.encode(tokenizer.build_chat(item["chosen"])))
                            + len(tokenizer.encode(tokenizer.build_chat(item["rejected"])))) if tokenizer else 0
            line = json.dumps(record, ensure_ascii=False) + "\n"

            # ---- 先填满验证集，不计入训练配额 ----
            if fval is not None and n_val < args.val_docs:
                fval.write(line)
                n_val += 1
                continue

            fout.write(line)
            n_written += 1
            st["docs"] += 1
            st["chars"] += n_chars
            st["tokens"] += n_tokens

            if n_written % args.log_every == 0:
                speed = n_written / max(time.time() - t0, 1e-6)
                summary = "  ".join(f"{s['source']}={s['docs']}" for s in stats)
                print(f"[{n_written:,}] {speed:.0f} 条/s  {summary}", flush=True)
    finally:
        fout.close()
        if fval is not None:
            fval.close()

    manifest = dict(
        task=args.task, out=args.out, val_out=val_path if args.val_docs > 0 else None,
        val_docs=n_val, total_docs=n_written,
        total_tokens=sum(s["tokens"] for s in stats) if tokenizer else None,
        total_chars=sum(s["chars"] for s in stats),
        budget=dict(max_docs=args.max_docs, max_tokens=args.max_tokens, max_chars=args.max_chars),
        sources=stats, args=vars(args), created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(os.path.splitext(args.out)[0] + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n==== 汇总 ====")
    for s in stats:
        tok = f" tokens={s['tokens']:,}" if tokenizer else ""
        err = f"  ERROR: {s['error']}" if s["error"] else ""
        print(f"{s['source']:<24} docs={s['docs']:,}{tok} chars={s['chars']:,} "
              f"(扫描 {s['scanned']:,}，过滤 {s['filtered']:,}，重复 {s['duplicates']:,}){err}")
    print(f"训练集 → {args.out}（{n_written:,} 条）")
    if args.val_docs > 0:
        print(f"验证集 → {val_path}（{n_val:,} 条）")
    return manifest


def build_parser():
    p = argparse.ArgumentParser(description="下载并混合 AllenAI 预训练 / 后训练数据",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", choices=["pretrain", "sft", "dpo"], default="pretrain")
    p.add_argument("--sources", type=str, default=None,
                   help="数据源及权重，如 c4-zh:0.7,c4-en:0.3；默认 pretrain=c4-zh,c4-en / sft=tulu3 / dpo=tulu3-pref")
    p.add_argument("--out", type=str, default=None,
                   help="输出 jsonl；默认 data/allenai_<task>.jsonl")
    p.add_argument("--list", action="store_true", help="列出内置数据源后退出")
    # ---- 数据量控制（任一达到即停；按权重分配到各数据源）----
    g = p.add_argument_group("数据量控制")
    g.add_argument("--max_docs", type=int, default=None, help="总文档 / 对话条数上限")
    g.add_argument("--max_tokens", type=int, default=None, help="总 token 数上限（需 --tokenizer）")
    g.add_argument("--max_chars", type=int, default=None, help="总字符数上限（无需分词器）")
    g.add_argument("--max_scan", type=int, default=0,
                   help="每个数据源最多扫描多少条原始样本（防止过滤太狠时一直读），0=不限")
    g.add_argument("--val_docs", type=int, default=0, help="额外写出多少条到 <out>.val.jsonl 作验证集")
    g.add_argument("--skip", type=int, default=0, help="每个数据源先跳过前 N 条（用来取不重叠的切片）")
    g.add_argument("--shuffle_buffer", type=int, default=10_000, help="流式 shuffle 缓冲大小，0=不打乱")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--tokenizer", type=str, default=None, help="分词器目录，用于精确统计 token 数")
    # ---- 过滤 ----
    f = p.add_argument_group("过滤")
    f.add_argument("--min_chars", type=int, default=50, help="[pretrain] 文档最少字符数")
    f.add_argument("--max_doc_chars", type=int, default=0, help="[pretrain] 单篇文档截断长度，0=不截断")
    f.add_argument("--min_cjk_ratio", type=float, default=0.0, help="中文字符占比下限，如 0.3 只留中文为主的样本")
    f.add_argument("--language", type=str, default=None, help="[sft/dpo] 按样本的 language 字段过滤，如 Chinese")
    f.add_argument("--source_filter", type=str, nargs="*", default=None,
                   help="[sft/dpo] 只保留 source 字段包含这些子串的样本（Tulu 3 mixture 适用）")
    f.add_argument("--max_turns", type=int, default=0, help="[sft/dpo] 最多 assistant 轮数，0=不限")
    f.add_argument("--dedup", type=int, default=1, choices=[0, 1], help="精确去重")
    f.add_argument("--text_field", type=str, default="text")
    f.add_argument("--messages_field", type=str, default="messages")
    p.add_argument("--log_every", type=int, default=10_000)
    p.add_argument("--hf_endpoint", type=str, default=None,
                   help="HuggingFace 访问地址，如 https://hf-mirror.com；默认取环境变量 HF_ENDPOINT")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list:
        for title, presets in (("预训练（--task pretrain）", PRETRAIN_PRESETS),
                               ("后训练 SFT（--task sft）", SFT_PRESETS),
                               ("偏好对齐 DPO（--task dpo）", DPO_PRESETS)):
            print(f"\n{title}")
            for k, v in presets.items():
                loc = v["path"] + (f"#{v['name']}" if v.get("name") else "") + \
                    (f"@{v['data_files']}" if v.get("data_files") else "")
                print(f"  {k:<24} {v['desc']}\n  {'':<24} {loc}")
        return
    from baize.hub import configure_hf_endpoint
    print(f"[hf] endpoint = {configure_hf_endpoint(args.hf_endpoint)}", flush=True)
    if args.sources is None:
        args.sources = {"pretrain": "c4-zh,c4-en", "sft": "tulu3", "dpo": "tulu3-pref"}[args.task]
    if args.out is None:
        args.out = f"data/allenai_{args.task}.jsonl"
    prepare(args)


if __name__ == "__main__":
    main()
