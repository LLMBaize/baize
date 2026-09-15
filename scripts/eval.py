#!/usr/bin/env python3
"""
BaiZe 评估与生成。

用法：
    python scripts/eval.py --weight pretrain --mode ppl --data data/corpus.txt
    python scripts/eval.py --weight sft --mode chat
    python scripts/eval.py --weight pretrain --mode chat --loops 16   # 深度外推
"""

import argparse
import glob
import math
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baize import BaiZeConfig, BaiZeForCausalLM, BaiZeTokenizer
from baize.trainer_utils import load_weights


@torch.inference_mode()
def eval_ppl(model, tokenizer, files, device, seq_len=512, stride=256):
    """滑动窗口困惑度（stride < seq_len，窗口重叠以覆盖长程依赖）。"""
    ids = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.extend(tokenizer.encode(line))
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
    parser.add_argument("--mode", type=str, default="chat", choices=["ppl", "chat"])
    parser.add_argument("--data", type=str, default="data/corpus*.txt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer")
    parser.add_argument("--save_dir", type=str, default="out")
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

    if args.mode == "ppl":
        files = sorted(glob.glob(args.data))
        assert files, f"未找到评估语料: {args.data}"
        eval_ppl(model, tokenizer, files, args.device)
    else:
        chat(model, tokenizer, args.device, loops=args.loops, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
