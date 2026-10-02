#!/usr/bin/env python3
"""
BaiZe 评估与生成。

用法：
    python scripts/eval.py --weight pretrain --mode ppl --data data/corpus.txt
    python scripts/eval.py --weight sft --mode chat
    python scripts/eval.py --weight pretrain --mode chat --loops 16   # 深度外推
    python scripts/eval.py --weight pretrain --mode bench --bench arc-easy,ceval,mmlu --bench_limit 500
    python scripts/eval.py --weight sft --mode bench --bench gsm8k --chat 1
    python scripts/eval.py --weight sft --mode bench --bench jsonl:data/my_bench.jsonl
"""

import argparse
import glob
import json
import math
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baize import BaiZeConfig, BaiZeForCausalLM, BaiZeTokenizer
from baize.benchmarks import evaluate, load_benchmark
from baize.data import iter_documents
from baize.trainer_utils import load_weights


@torch.inference_mode()
def eval_ppl(model, tokenizer, files, device, seq_len=512, stride=256, max_docs=None):
    """滑动窗口困惑度（stride < seq_len，窗口重叠以覆盖长程依赖）。"""
    ids = []
    for text in iter_documents(files, max_docs=max_docs):
        ids.extend(tokenizer.encode(text))
        ids.append(tokenizer.eos_token_id)
    nll, count = 0.0, 0
    for i in range(0, len(ids) - seq_len, stride):
        chunk = torch.tensor(ids[i : i + seq_len + 1], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(chunk[:, :-1]).logits
        loss = torch.nn.functional.cross_entropy(
            logits[0].float(), chunk[0, 1:], reduction="sum"
        )
        nll += loss.item()
        count += seq_len
    ppl = math.exp(nll / count)
    print(f"tokens={count:,}  mean NLL={nll / count:.4f}  PPL={ppl:.2f}")
    return ppl


@torch.inference_mode()
def chat(model, tokenizer, device, loops=None, max_new_tokens=256):
    print("进入对话模式（输入 exit 退出）")
    history = []
    while True:
        try:
            text = input("用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("exit", "quit"):
            break
        history.append({"role": "user", "content": text})
        prompt = tokenizer.build_chat(history, add_generation_prompt=True)
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, n_loops=loops,
            eos_token_id=tokenizer.im_end_id if hasattr(tokenizer, "im_end_id") else 4,
            repetition_penalty=1.05,
        )
        reply = tokenizer.decode(out[0][ids.shape[1]:].tolist())
        print(f"白泽: {reply}")
        history.append({"role": "assistant", "content": reply})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="pretrain")
    parser.add_argument("--mode", type=str, default="chat", choices=["ppl", "chat", "bench"])
    parser.add_argument("--data", type=str, default="data/corpus*.txt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer")
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--max_docs", type=int, default=None, help="[ppl] 最多评估多少篇文档")
    parser.add_argument("--bench", type=str, default="arc-easy",
                        help="[bench] 逗号分隔：arc-easy,arc-challenge,mmlu,ceval[:学科],hellaswag,gsm8k,jsonl:<路径>")
    parser.add_argument("--bench_limit", type=int, default=None, help="[bench] 每个评测集最多评多少题")
    parser.add_argument("--chat", type=int, default=0, choices=[0, 1], help="[bench] 用对话模板包裹题目（SFT 模型）")
    parser.add_argument("--bench_out", type=str, default=None, help="[bench] 结果写入 json 文件")
    parser.add_argument("--loops", type=int, default=None, help="推理循环圈数（可大于训练值做深度外推）")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = BaiZeTokenizer.from_pretrained(args.tokenizer)
    config = BaiZeConfig.from_pretrained(args.save_dir)
    config.vocab_size = tokenizer.vocab_size
    model = BaiZeForCausalLM(config)
    wp = f"{args.save_dir}/{args.weight}.safetensors"
    n_loaded = load_weights(model, wp)
    print(f"从 {wp} 加载 {n_loaded} 个权重", flush=True)
    model = model.to(args.device).eval()

    A = model.model.recurrent.injection.get_A()
    print(f"谱半径 ρ(A) = {A.max().item():.4f}（须 < 1）")
    print(f"推理循环圈数: {args.loops or config.max_loop_iters}")

    if args.mode == "bench":
        results = {}
        for spec in [b.strip() for b in args.bench.split(",") if b.strip()]:
            kind, items = load_benchmark(spec, limit=args.bench_limit)
            print(f"[{spec}] {len(items)} 题（{'选择题' if kind == 'mc' else '生成题'}）", flush=True)
            results[spec] = evaluate(model, tokenizer, kind, items, args.device, n_loops=args.loops,
                                     chat=bool(args.chat))
            print(f"[{spec}] {results[spec]}", flush=True)
        if args.bench_out:
            with open(args.bench_out, "w", encoding="utf-8") as f:
                json.dump({"weight": args.weight, "loops": args.loops, "results": results}, f,
                          ensure_ascii=False, indent=2)
    elif args.mode == "ppl":
        files = sorted(f for pat in args.data.split(",") for f in glob.glob(pat.strip()))
        assert files, f"未找到评估语料: {args.data}"
        eval_ppl(model, tokenizer, files, args.device, max_docs=args.max_docs)
    else:
        chat(model, tokenizer, args.device, loops=args.loops, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
