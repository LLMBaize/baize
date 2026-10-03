"""模型层改进的测试：ACT 初始化 / ponder cost、MoE 批量计算、激活重计算、旧权重兼容、批量生成。

运行：python tests/test_model_features.py   （也兼容 pytest）
"""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from baize import BaiZeConfig, BaiZeForCausalLM
from baize.model import MoEFFN, convert_legacy_moe_state_dict
from baize.trainer_utils import load_weights


def small_config(**kw):
    base = dict(vocab_size=100, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                head_dim=16, max_loop_iters=6, intermediate_size=128, moe_intermediate_size=64,
                max_position_embeddings=256)
    base.update(kw)
    return BaiZeConfig(**base)


def count_loops(model, x):
    model(x)
    return model.model.recurrent.avg_loops.item()


def test_act_init_runs_all_loops():
    torch.manual_seed(0)
    model = BaiZeForCausalLM(small_config()).eval()
    assert abs(model.model.recurrent.act.halt.bias.item() - (-3.0)) < 1e-6
    x = torch.randint(5, 100, (2, 16))
    with torch.no_grad():
        assert count_loops(model, x) == 6  # 初始停机概率 ≈0.05，跑满所有圈
        # 对照：偏置为 0 时（旧行为）两三圈就停
        model.model.recurrent.act.halt.bias.fill_(0.0)
        assert count_loops(model, x) <= 3


def test_ponder_cost_gradient_pushes_toward_halting():
    torch.manual_seed(0)
    model = BaiZeForCausalLM(small_config(act_ponder_coef=1.0)).train()
    x = torch.randint(5, 100, (2, 16))
    out = model(x)
    rec = model.model.recurrent
    assert 6.0 <= rec.ponder_cost.item() <= 7.0  # N = 6 圈 + 剩余概率 R
    out.aux_loss.backward()
    # 最小化 ponder cost → 提高停机概率 → bias 的梯度为负（梯度下降会增大 bias）
    assert rec.act.halt.bias.grad.item() < 0


def test_grad_checkpointing_matches():
    torch.manual_seed(0)
    cfg = small_config(use_moe=True)
    x = torch.randint(5, 100, (2, 16))
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        model = BaiZeForCausalLM(cfg).train()
        model.enable_grad_checkpointing(ckpt)
        out = model(x, labels=x)
        (out.loss + out.aux_loss).backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
        if ckpt:
            loss_ckpt = out.loss.item()
        else:
            loss_ref = out.loss.item()
    assert abs(loss_ref - loss_ckpt) < 1e-5
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], atol=1e-5), n


def test_all_expert_params_receive_grad():
    """批量 bmm 计算下所有专家参数每步都在计算图里（DDP 不会报未使用参数）。"""
    torch.manual_seed(0)
    moe = MoEFFN(small_config()).train()
    moe(torch.randn(1, 3, 64)).sum().backward()
    for w in (moe.w_gate, moe.w_up, moe.w_down):
        assert w.grad is not None


def test_capacity_factor_drops_overflow():
    torch.manual_seed(0)
    cfg = small_config(moe_capacity_factor=0.5)
    moe = MoEFFN(cfg).eval()
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[1, 0] = 50.0
        moe.gate.weight[2, 0] = 49.0  # 所有 token 挤向专家 1、2
        h = torch.zeros(1, 32, 64)
        h[..., 0] = 1.0
        out_drop = moe(h)
        cfg.moe_capacity_factor = 0.0
        out_full = moe(h)
    shared = moe.shared_experts(h)
    # 容量 = ceil(0.5·32·2/8) = 4：只有前 4 个 token 被专家处理，其余只剩共享专家输出
    routed_drop = (out_drop - shared).abs().sum(-1)[0]
    assert (routed_drop[:4] > 0).all() and torch.allclose(routed_drop[4:], torch.zeros(28), atol=1e-7)
    assert ((out_full - shared).abs().sum(-1) > 0).all()


def test_legacy_moe_weights_convert():
    torch.manual_seed(0)
    cfg = small_config()
    moe = MoEFFN(cfg)
    legacy = {"model.recurrent.ffn.gate.weight": moe.gate.weight.data}
    for i in range(cfg.n_experts):
        legacy[f"model.recurrent.ffn.experts.{i}.gate_proj.weight"] = moe.w_gate.data[i].t().clone()
        legacy[f"model.recurrent.ffn.experts.{i}.up_proj.weight"] = moe.w_up.data[i].t().clone()
        legacy[f"model.recurrent.ffn.experts.{i}.down_proj.weight"] = moe.w_down.data[i].t().clone()
    new = convert_legacy_moe_state_dict(legacy)
    assert torch.equal(new["model.recurrent.ffn.w_gate"], moe.w_gate.data)
    assert torch.equal(new["model.recurrent.ffn.w_down"], moe.w_down.data)
    assert not any(".experts." in k for k in new)


def test_demo_weights_load_strict():
    cfg = BaiZeConfig.from_pretrained(os.path.join(ROOT, "demo_weights"))
    model = BaiZeForCausalLM(cfg)
    n = load_weights(model, os.path.join(ROOT, "demo_weights", "model.safetensors"))
    assert n == len(list(model.parameters()))


def test_load_weights_rejects_mismatch():
    import tempfile
    from baize.trainer_utils import save_weights
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        path = save_weights(BaiZeForCausalLM(small_config()), os.path.join(d, "w.safetensors"))
        other = BaiZeForCausalLM(small_config(vocab_size=120))
        try:
            load_weights(other, path)
            raise AssertionError("vocab_size 不一致时应报错")
        except ValueError as exc:
            assert "embed_tokens" in str(exc)
        load_weights(other, path, strict=False)  # 非严格模式只警告


def test_batched_generate_matches_single():
    torch.manual_seed(0)
    model = BaiZeForCausalLM(small_config()).eval()
    prompts = torch.randint(5, 100, (3, 7))
    batch = model.generate(prompts, max_new_tokens=6, temperature=0, top_k=10,
                           repetition_penalty=1.2)
    for b in range(3):
        single = model.generate(prompts[b:b + 1], max_new_tokens=6, temperature=0, top_k=10,
                                repetition_penalty=1.2)
        assert torch.equal(batch[b], single[0]), b


def test_generate_pads_finished_with_eos():
    torch.manual_seed(0)
    model = BaiZeForCausalLM(small_config()).eval()
    prompts = torch.randint(5, 100, (2, 5))
    first = model.generate(prompts, max_new_tokens=1, temperature=0)[:, -1]
    eos = int(first[0])  # 让第 0 个样本第一步就生成 eos
    out = model.generate(prompts, max_new_tokens=4, temperature=0, eos_token_id=eos)
    assert (out[0, 5:] == eos).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
