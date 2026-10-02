"""回归测试：覆盖已修复的若干 bug。

运行：python tests/test_regressions.py   （也兼容 pytest）
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from baize import BaiZeConfig, BaiZeForCausalLM
from baize.model import MoEFFN, RecurrentBlock


def small_config(**kw):
    base = dict(
        vocab_size=100, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_loop_iters=3, intermediate_size=128, moe_intermediate_size=64,
        max_position_embeddings=256, kv_lora_rank=32, q_lora_rank=32,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16,
    )
    base.update(kw)
    return BaiZeConfig(**base)


def _check_kv_cache_consistency(attn_type):
    torch.manual_seed(0)
    model = BaiZeForCausalLM(small_config(attn_type=attn_type)).eval()
    x = torch.randint(5, 100, (2, 10))
    with torch.no_grad():
        full = model(x).logits
        # prefill 带 cache 必须与无 cache 一致（因果 mask）
        prefill = model(x, kv_cache={}).logits
        assert torch.allclose(full, prefill, atol=1e-4), (full - prefill).abs().max()
        # 分块增量解码必须与整段前向一致
        cache = {}
        model(x[:, :4], kv_cache=cache)
        chunk = model(x[:, 4:7], kv_cache=cache, start_pos=4).logits
        step = model(x[:, 7:8], kv_cache=cache, start_pos=7).logits
        assert torch.allclose(full[:, 4:7], chunk, atol=1e-4), (full[:, 4:7] - chunk).abs().max()
        assert torch.allclose(full[:, 7:8], step, atol=1e-4), (full[:, 7:8] - step).abs().max()


def test_kv_cache_causal_gqa():
    _check_kv_cache_consistency("gqa")


def test_kv_cache_causal_mla():
    _check_kv_cache_consistency("mla")


def test_moe_routes_to_selected_experts():
    torch.manual_seed(0)
    cfg = small_config()
    moe = MoEFFN(cfg).eval()
    with torch.no_grad():
        # 让所有 token 只选专家 1、2（专家 0 不出现）
        moe.gate.weight.zero_()
        moe.gate.weight[1, 0] = 50.0
        moe.gate.weight[2, 0] = 49.0
        h = torch.randn(1, 4, cfg.hidden_size) * 0.01
        h[..., 0] = 1.0
        out = moe(h).view(-1, cfg.hidden_size)

        flat = h.view(-1, cfg.hidden_size)
        w, idx = torch.softmax(moe.gate(flat), -1).topk(cfg.n_experts_per_tok)
        w = w / w.sum(-1, keepdim=True)
        ref = moe.shared_experts(flat).clone()
        for n in range(flat.shape[0]):
            for j in range(idx.shape[1]):
                ref[n] += w[n, j] * moe.experts[idx[n, j].item()](flat[n])
        assert idx[0].tolist() == [1, 2] or sorted(idx[0].tolist()) == [1, 2]
        assert torch.allclose(out, ref, atol=1e-6), (out - ref).abs().max()


def test_aux_loss_covers_all_loops_and_moe_layers():
    torch.manual_seed(0)
    model = BaiZeForCausalLM(small_config(use_moe=True)).train()
    x = torch.randint(5, 100, (2, 8))
    out = model(x, labels=x, n_loops=3)
    expected = model.model.recurrent.aux_loss
    for layer in list(model.model.prelude) + list(model.model.coda):
        expected = expected + layer.ffn.aux_loss
    assert torch.allclose(out.aux_loss, expected)
    assert out.aux_loss.item() > model.model.recurrent.aux_loss.item()


def test_act_weights_sum_to_one():
    torch.manual_seed(0)
    cfg = small_config(use_act=True)
    block = RecurrentBlock(cfg).eval()
    with torch.no_grad():
        # 停机概率很小 → 跑满所有圈仍未停机；剩余概率须在最后一圈补齐
        block.act.halt.weight.zero_()
        block.act.halt.bias.fill_(-6.0)
        # 把 h 替换为常数 1，输出即为各圈权重之和
        block.injection.forward = lambda h, e, f: torch.ones_like(h)
        h = torch.randn(1, 5, cfg.hidden_size)
        cos = torch.ones(1, 5, 1, cfg.head_dim)
        sin = torch.zeros(1, 5, 1, cfg.head_dim)
        out = block(h, h, cos, sin, n_loops=3)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5), out.mean()


def test_pretrain_labels_aligned_with_inputs():
    from pretrain import PretrainDataset

    class Tok:
        eos_token_id = 2

        def encode(self, s):
            return [int(c) for c in s.split()]

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(" ".join(str(i) for i in range(10, 40)) + "\n")
        path = f.name
    try:
        ds = PretrainDataset([path], Tok(), 8)
        for i in range(len(ds)):
            x, y = ds[i]
            mask = y != -100
            # 模型内部做 shift，这里 labels 必须与 input_ids 对齐
            assert torch.equal(x[mask], y[mask])
        x0, _ = ds[0]
        assert x0.tolist() == list(range(10, 18))
    finally:
        os.remove(path)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
