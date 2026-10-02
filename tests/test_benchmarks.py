"""评测集适配与打分测试（不联网：用模拟样本代替 HF 数据集）。

运行：python tests/test_benchmarks.py   （也兼容 pytest）
"""

import json
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from baize import BaiZeConfig, BaiZeForCausalLM, BaiZeTokenizer
from baize import benchmarks as B


def test_adapters():
    arc = B.adapt_arc({"question": "Which is a gas?", "answerKey": "B",
                       "choices": {"text": ["rock", "steam"], "label": ["A", "B"]}})
    assert arc["answer"] == 1 and arc["choices"] == [" rock", " steam"]
    mmlu = B.adapt_mmlu({"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1})
    assert mmlu["choices"] == [" A", " B", " C", " D"] and "B. 4" in mmlu["prompt"]
    ceval = B.adapt_ceval({"question": "1+1=?", "A": "1", "B": "2", "C": "3", "D": "4", "answer": "B"})
    assert ceval["answer"] == 1 and ceval["choices"] == ["A", "B", "C", "D"] and "答案：" in ceval["prompt"]
    hs = B.adapt_hellaswag({"activity_label": "Cooking", "ctx_a": "He cracks an egg.", "ctx_b": "then he",
                            "endings": ["fries it.", "eats the pan."], "label": "0"})
    assert hs["answer"] == 0 and hs["prompt"].startswith("Cooking: He cracks an egg. Then he")
    gsm = B.adapt_gsm8k({"question": "q", "answer": "work...\n#### 1,234"})
    assert gsm["answer"] == "1234"
    assert B._extract_number("so the total is 1,234.") == "1234"


def test_load_benchmark_with_mock_loader():
    calls = []

    def loader(path, name, split):
        calls.append((path, name, split))
        return [{"question": "q", "A": "a", "B": "b", "C": "c", "D": "d", "answer": "A"}] * 3
    kind, items = B.load_benchmark("ceval", limit=5, loader=loader)
    assert kind == "mc" and len(items) == 5
    assert calls[0] == ("ceval/ceval-exam", "computer_network", "val") and len(calls) == 2
    _, items = B.load_benchmark("ceval:logic", loader=loader)
    assert calls[-1][1] == "logic" and len(items) == 3


def test_score_choices_matches_manual():
    tok = BaiZeTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer"))
    torch.manual_seed(0)
    model = BaiZeForCausalLM(BaiZeConfig(vocab_size=tok.vocab_size, hidden_size=64, num_attention_heads=4,
                                         num_key_value_heads=2, head_dim=16, max_loop_iters=2,
                                         intermediate_size=128, moe_intermediate_size=64)).eval()
    prompt, choices = "问题：白泽是什么？\n答案：", ["神兽", "一种很长很长的答案"]
    scores = B.score_choices(model, tok, prompt, choices, "cpu")
    for c, s in zip(choices, scores):
        ctx, cont = tok.encode(prompt), tok.encode(c)
        ids = torch.tensor([ctx + cont])
        with torch.no_grad():
            logp = torch.log_softmax(model(ids).logits.float(), -1)[0]
        manual = sum(logp[len(ctx) + j - 1, t].item() for j, t in enumerate(cont))
        assert abs(manual - s) < 1e-3


def test_eval_script_on_local_jsonl():
    rows = [{"question": "白泽是什么？", "choices": ["中国古代神话中的神兽", "一种水果"], "answer": 0},
            {"question": "What is 2+2?", "choices": ["4", "5"], "answer": "A"},
            {"question": "Pick", "choices": ["x", "y", "z"], "answer": 2, "format": "letter"}]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "b.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out = os.path.join(d, "res.json")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "eval.py"), "--mode", "bench",
                            "--save_dir", os.path.join(ROOT, "demo_weights"), "--weight", "model",
                            "--tokenizer", os.path.join(ROOT, "demo_weights"), "--device", "cpu",
                            "--bench", f"jsonl:{path}", "--bench_out", out],
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        res = json.load(open(out))["results"][f"jsonl:{path}"]
        assert res["n"] == 3 and 0 <= res["acc"] <= 1 and abs(res["random_baseline"] - (0.5 + 0.5 + 1 / 3) / 3) < 1e-9


def test_gsm8k_generation_path():
    tok = BaiZeTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer"))
    torch.manual_seed(0)
    model = BaiZeForCausalLM(BaiZeConfig(vocab_size=tok.vocab_size, hidden_size=64, num_attention_heads=4,
                                         num_key_value_heads=2, head_dim=16, max_loop_iters=2,
                                         intermediate_size=128, moe_intermediate_size=64)).eval()
    res = B.evaluate(model, tok, "gen", [{"prompt": "Question: 1+1?\nAnswer:", "answer": "2"}], "cpu",
                     max_new_tokens=5)
    assert res["n"] == 1 and res["acc"] in (0.0, 1.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
