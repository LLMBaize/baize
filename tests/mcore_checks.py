#!/usr/bin/env python3
"""Megatron-core 版的 CPU 冒烟检查（由 tests/test_mcore.py 在子进程中调用，避免 CPU 垫片影响其它测试）。

    python tests/mcore_checks.py equivalence   # 原版模型 → 转换 → mcore 模型：logits / loss / 梯度一致，往返无损
    python tests/mcore_checks.py tp_convert    # TP 切分 / 合并往返无损
"""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from baize import BaiZeConfig, BaiZeForCausalLM  # noqa: E402


def tiny_config(**kw):
    base = dict(vocab_size=100, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                intermediate_size=64, moe_intermediate_size=16, n_experts=4, n_experts_per_tok=2,
                prelude_layers=2, coda_layers=1, max_loop_iters=4, max_position_embeddings=64)
    base.update(kw)
    return BaiZeConfig(**base)


def equivalence():
    from baize.mcore import build_model, init_parallel
    from baize.mcore.convert import hf_to_mcore, mcore_to_hf, model_tensors

    init_parallel(cpu=True)
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1))
    # 跑满所有圈（含 MoE aux-loss）/ ACT 提前停机（提前退出循环时 aux 的均摊方式与原版不同，见 README，这里关掉 aux）
    for use_moe, bias, aux in [(True, -3.0, 1e-3), (False, 3.0, 0.0), (True, 3.0, 0.0)]:
        cfg = tiny_config(use_moe=use_moe, act_init_bias=bias, router_aux_loss_coef=aux)
        torch.manual_seed(0)
        hf = BaiZeForCausalLM(cfg)
        with torch.no_grad():
            hf.model.recurrent.injection.log_A.normal_()
            hf.model.recurrent.lora.B.normal_(std=0.1)
        mc = build_model(cfg, n_loops_train=4)
        res = mc.load_state_dict(hf_to_mcore(hf.state_dict(), cfg, vocab_padded=mc.vocab_padded), strict=False)
        assert not [k for k in res.missing_keys if not k.endswith("_extra_state")] and not res.unexpected_keys

        x = torch.randint(0, 100, (2, 12))
        hf.eval(), mc.eval()
        with torch.no_grad():
            for loops in (1, 4):
                diff = (hf(x, n_loops=loops).logits - mc(x, n_loops=loops)).abs().max().item()
                assert diff < 1e-5, f"logits 不一致 moe={use_moe} loops={loops}: {diff}"
                assert hf.model.recurrent.avg_loops.item() == mc.recurrent.avg_loops.item()

        # 训练模式下的 loss 与梯度
        hf.train(), mc.train()
        out = hf(x, labels=x, n_loops=4)
        (out.loss + out.aux_loss).backward()
        tl, mask = mc(x, labels=x, n_loops=4)
        mc_loss = (tl * mask).sum() / mask.sum()
        (mc_loss + cfg.act_ponder_coef * mc.recurrent.ponder_cost).backward()
        assert abs(out.loss.item() - mc_loss.item()) < 1e-5
        grads = mcore_to_hf({k: (p.grad if p.grad is not None else torch.zeros_like(p))
                             for k, p in mc.named_parameters()}, cfg)
        for k, p in hf.named_parameters():
            if p.grad is not None:
                rel = ((p.grad - grads[k]).abs().max() / (p.grad.abs().max() + 1e-12)).item()
                assert rel < 1e-3, f"梯度不一致 {k}: {rel}"

        back = mcore_to_hf(model_tensors(mc), cfg)
        ref = hf.state_dict()
        assert set(back) == set(ref) and all(torch.equal(ref[k], back[k]) for k in ref)
    print("OK equivalence")


def tp_convert():
    from baize.mcore.convert import hf_to_mcore, merge_tp, shard_tp

    for use_moe in (False, True):
        cfg = tiny_config(use_moe=use_moe)
        torch.manual_seed(0)
        full = hf_to_mcore(BaiZeForCausalLM(cfg).state_dict(), cfg, vocab_padded=256)
        shards = [shard_tp(full, cfg, r, 2) for r in range(2)]
        assert shards[0]["embedding.word_embeddings.weight"].shape[0] == 128
        merged = merge_tp(shards, cfg)
        assert set(merged) == set(full) and all(torch.equal(full[k], merged[k]) for k in full)
    print("OK tp_convert")


if __name__ == "__main__":
    {"equivalence": equivalence, "tp_convert": tp_convert}[sys.argv[1]]()
