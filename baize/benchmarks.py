"""
BaiZe — 标准评测集
==================
两类评测：

    选择题（对数似然打分）：对每个选项计算 log P(选项 | 题干)，取最大者为预测
        acc       —— 直接比较总 log 概率
        acc_norm  —— 按选项字符数归一化后比较（对长短不一的选项更公平，HellaSwag/ARC 常用）
    生成题（GSM8K）：贪心生成，抽取最后一个数字与标准答案比较

内置评测集（HuggingFace，需联网；字段以数据集页面为准）：
    arc-easy / arc-challenge  allenai/ai2_arc            英文科学选择题
    mmlu                      cais/mmlu (all)            英文 57 学科选择题（字母打分）
    ceval[:学科]              ceval/ceval-exam (val)     中文 52 学科选择题（字母打分）
    hellaswag                 Rowan/hellaswag            英文常识续写
    gsm8k                     openai/gsm8k (main)        英文小学数学（生成）
    jsonl:<路径>              本地文件，每行 {"question", "choices": [...], "answer": 下标或字母}
"""

import json
import math
import re

import torch

LETTERS = "ABCDEFGHIJ"

CEVAL_SUBJECTS = [
    "computer_network", "operating_system", "computer_architecture", "college_programming",
    "college_physics", "college_chemistry", "advanced_mathematics", "probability_and_statistics",
    "discrete_mathematics", "electrical_engineer", "metrology_engineer", "high_school_mathematics",
    "high_school_physics", "high_school_chemistry", "high_school_biology", "middle_school_mathematics",
    "middle_school_biology", "middle_school_physics", "middle_school_chemistry", "veterinary_medicine",
    "college_economics", "business_administration", "marxism", "mao_zedong_thought", "education_science",
    "teacher_qualification", "high_school_politics", "high_school_geography", "middle_school_politics",
    "middle_school_geography", "modern_chinese_history", "ideological_and_moral_cultivation", "logic",
    "law", "chinese_language_and_literature", "art_studies", "professional_tour_guide",
    "legal_professional", "high_school_chinese", "high_school_history", "middle_school_history",
    "civil_servant", "sports_science", "plant_protection", "basic_medicine", "clinical_medicine",
    "urban_and_rural_planner", "accountant", "fire_engineer", "environmental_impact_assessment_engineer",
    "tax_accountant", "physician",
]


# ---------------------------------------------------------------------------
# 适配器：原始样本 → 统一格式
#   选择题 {"prompt", "choices", "answer"(下标)}；生成题 {"prompt", "answer"(字符串)}
# ---------------------------------------------------------------------------


def _letter_mc(question, options, answer_idx, zh=False):
    body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    if zh:
        prompt = f"以下是单项选择题，请选出正确答案。\n\n{question}\n{body}\n答案："
        choices = [LETTERS[i] for i in range(len(options))]
    else:
        prompt = f"The following is a multiple choice question.\n\n{question}\n{body}\nAnswer:"
        choices = [f" {LETTERS[i]}" for i in range(len(options))]
    return {"prompt": prompt, "choices": choices, "answer": answer_idx}


def adapt_arc(ex):
    labels = list(ex["choices"]["label"])
    return {"prompt": f"Question: {ex['question']}\nAnswer:",
            "choices": [" " + t for t in ex["choices"]["text"]],
            "answer": labels.index(ex["answerKey"])}


def adapt_mmlu(ex):
    return _letter_mc(ex["question"], ex["choices"], int(ex["answer"]))


def adapt_ceval(ex):
    options = [ex[c] for c in "ABCD"]
    return _letter_mc(ex["question"], options, "ABCD".index(ex["answer"]), zh=True)


def _hellaswag_clean(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def adapt_hellaswag(ex):
    ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    return {"prompt": _hellaswag_clean(ex["activity_label"] + ": " + ctx),
            "choices": [" " + _hellaswag_clean(e) for e in ex["endings"]],
            "answer": int(ex["label"])}


def adapt_gsm8k(ex):
    return {"prompt": f"Question: {ex['question']}\nAnswer: Let's think step by step.",
            "answer": ex["answer"].split("####")[-1].strip().replace(",", "")}


def adapt_local(ex):
    ans = ex["answer"]
    if isinstance(ans, str):
        ans = LETTERS.index(ans.strip().upper())
    zh = any("一" <= c <= "鿿" for c in ex["question"])
    if ex.get("format", "text") == "letter":
        return _letter_mc(ex["question"], ex["choices"], ans, zh=zh)
    sep = "" if zh else " "
    prompt = f"问题：{ex['question']}\n答案：" if zh else f"Question: {ex['question']}\nAnswer:"
    return {"prompt": prompt, "choices": [sep + c for c in ex["choices"]], "answer": ans}


BENCHMARKS = {
    "arc-easy": dict(path="allenai/ai2_arc", name="ARC-Easy", split="test", adapter=adapt_arc, kind="mc"),
    "arc-challenge": dict(path="allenai/ai2_arc", name="ARC-Challenge", split="test", adapter=adapt_arc, kind="mc"),
    "mmlu": dict(path="cais/mmlu", name="all", split="test", adapter=adapt_mmlu, kind="mc"),
    "ceval": dict(path="ceval/ceval-exam", split="val", adapter=adapt_ceval, kind="mc"),
    "hellaswag": dict(path="Rowan/hellaswag", split="validation", adapter=adapt_hellaswag, kind="mc"),
    "gsm8k": dict(path="openai/gsm8k", name="main", split="test", adapter=adapt_gsm8k, kind="gen"),
}


def load_benchmark(spec: str, limit=None, loader=None):
    """返回 (kind, items)。spec：内置名、ceval:学科、或 jsonl:路径。"""
    if spec.startswith("jsonl:"):
        with open(spec[len("jsonl:"):], encoding="utf-8") as f:
            items = [adapt_local(json.loads(line)) for line in f if line.strip()]
        return "mc", items[:limit] if limit else items

    name, _, sub = spec.partition(":")
    if name not in BENCHMARKS:
        raise ValueError(f"未知评测集 {spec}，可选: {', '.join(BENCHMARKS)} 或 jsonl:<路径>")
    cfg = BENCHMARKS[name]
    if loader is None:
        from datasets import load_dataset as loader
    configs = [sub] if sub else (CEVAL_SUBJECTS if name == "ceval" else [cfg.get("name")])
    items = []
    for c in configs:
        ds = loader(cfg["path"], c, split=cfg["split"])
        for ex in ds:
            items.append(cfg["adapter"](ex))
            if limit and len(items) >= limit:
                return cfg["kind"], items
    return cfg["kind"], items


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------


def _wrap_prompt(tokenizer, prompt, chat):
    if chat:
        return tokenizer.build_chat([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return prompt


@torch.inference_mode()
def score_choices(model, tokenizer, prompt, choices, device, n_loops=None, chat=False):
    """返回每个选项续写的 log 概率之和（选项 token 单独编码后拼接在 prompt 之后）。"""
    ctx = tokenizer.encode(_wrap_prompt(tokenizer, prompt, chat))
    seqs, spans = [], []
    for c in choices:
        cont = tokenizer.encode(c)
        seqs.append(ctx + cont)
        spans.append((len(ctx), len(ctx) + len(cont)))
    maxlen = max(len(s) for s in seqs)
    ids = torch.tensor([s + [0] * (maxlen - len(s)) for s in seqs], dtype=torch.long, device=device)
    logp = torch.log_softmax(model(ids, n_loops=n_loops).logits.float(), dim=-1)
    scores = []
    for b, (start, end) in enumerate(spans):
        pos = torch.arange(start, end, device=device)
        scores.append(logp[b, pos - 1, ids[b, pos]].sum().item())
    return scores


def _extract_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def evaluate(model, tokenizer, kind, items, device, n_loops=None, chat=False, max_new_tokens=256, log_every=200):
    model.eval()
    if kind == "mc":
        correct = correct_norm = 0
        for i, it in enumerate(items, 1):
            scores = score_choices(model, tokenizer, it["prompt"], it["choices"], device, n_loops, chat)
            pred = max(range(len(scores)), key=lambda j: scores[j])
            norm = [s / max(1, len(c.strip())) for s, c in zip(scores, it["choices"])]
            pred_norm = max(range(len(norm)), key=lambda j: norm[j])
            correct += pred == it["answer"]
            correct_norm += pred_norm == it["answer"]
            if log_every and i % log_every == 0:
                print(f"  [{i}/{len(items)}] acc={correct / i:.4f}", flush=True)
        n = max(1, len(items))
        return {"n": len(items), "acc": correct / n, "acc_norm": correct_norm / n,
                "random_baseline": sum(1 / len(it["choices"]) for it in items) / n}

    correct = 0
    for i, it in enumerate(items, 1):
        ids = torch.tensor([tokenizer.encode(_wrap_prompt(tokenizer, it["prompt"], chat))],
                           dtype=torch.long, device=device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, n_loops=n_loops, temperature=0,
                             eos_token_id=tokenizer.im_end_id if chat else tokenizer.eos_token_id)
        text = tokenizer.decode(out[0, ids.shape[1]:].tolist())
        text = text.split("Question:")[0]  # 基座模型常会继续编下一题
        pred = _extract_number(text)
        try:
            correct += pred is not None and math.isclose(float(pred), float(it["answer"]))
        except ValueError:
            pass
        if log_every and i % log_every == 0:
            print(f"  [{i}/{len(items)}] acc={correct / i:.4f}", flush=True)
    return {"n": len(items), "acc": correct / max(1, len(items))}
