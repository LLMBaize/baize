"""
权重互转：原版 BaiZeForCausalLM（HF 格式，baize/model.py）⇄ Megatron-core 版（baize/mcore/model.py）
====================================================================================================

    hf_to_mcore(hf_sd, cfg)         → 完整（未切分）的 mcore state dict
    mcore_to_hf(mcore_sd, cfg)      → HF state dict（可直接 save_weights 成 .safetensors，供 eval/demo/sft 使用）
    shard_tp(full_sd, cfg, r, n)    → 张量并行第 r 个分片
    merge_tp([sd_0 … sd_{n-1}], cfg)→ 合并回完整 state dict

布局要点：
    - Megatron 的 linear_qkv 按 query group 分组排列：[q(该组所有 head) | k | v] × num_query_groups
    - SwiGLU 的 linear_fc1 = [gate; up] 按行拼接；张量并行时 gate、up 各自切分后再拼
    - 原版 MoE 专家合并存成 w_gate/w_up [E, H, I]、w_down [E, I, H]（x @ W），Megatron 为逐专家 Linear（[out, in]）
"""

import re

import torch


# ---------------------------------------------------------------------------
# 单层映射
# ---------------------------------------------------------------------------


def _qkv_fuse(q, k, v, cfg):
    hd = cfg.head_dim
    ng = cfg.num_key_value_heads or cfg.num_attention_heads
    per = cfg.num_attention_heads // ng
    H = q.shape[1]
    q = q.view(ng, per * hd, H)
    k = k.view(ng, hd, H)
    v = v.view(ng, hd, H)
    return torch.cat([q, k, v], dim=1).reshape(-1, H)


def _qkv_split(qkv, cfg):
    hd = cfg.head_dim
    ng = cfg.num_key_value_heads or cfg.num_attention_heads
    per = cfg.num_attention_heads // ng
    H = qkv.shape[1]
    g = qkv.view(ng, (per + 2) * hd, H)
    q = g[:, : per * hd].reshape(-1, H)
    k = g[:, per * hd: (per + 1) * hd].reshape(-1, H)
    v = g[:, (per + 1) * hd:].reshape(-1, H)
    return q, k, v


def _rope_perm(r):
    """Megatron 的 MLA 在 RoPE 前把 rope 通道重排为 [偶数下标 | 奇数下标]（DeepSeek 约定），
    原版用半分（rotate_half）RoPE。要让两者等价，rope 部分权重行需按 idx 重排：
    W_mc[idx[j]] = W_orig[j]，idx = [0,2,4,…, 1,3,5,…]。返回 (idx, 逆 idx)。"""
    idx = torch.cat([torch.arange(0, r, 2), torch.arange(1, r, 2)])
    return idx, torch.argsort(idx)


def _mla_q_up_to_mcore(q_nope, q_rope, cfg):
    """原版 q_up_nope [H·nope, q_lora] 与 q_up_rope [H·rope, q_lora] → Megatron linear_q_up_proj [H·(nope+rope), q_lora]，
    每个 head 的行排列为 [nope | rope]，rope 行按 _rope_perm 重排。"""
    H, n, r = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim
    _, inv = _rope_perm(r)
    q_rope = q_rope.reshape(H, r, -1)[:, inv]
    per_head = torch.cat([q_nope.reshape(H, n, -1), q_rope], dim=1)
    return per_head.reshape(H * (n + r), -1)


def _mla_q_up_to_hf(w, cfg):
    H, n, r = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim
    idx, _ = _rope_perm(r)
    per_head = w.reshape(H, n + r, -1)
    q_rope = per_head[:, n:][:, idx]
    return per_head[:, :n].reshape(H * n, -1), q_rope.reshape(H * r, -1)


def _mla_kv_down_to_mcore(w, cfg):
    """kv_down 输出 = [c_kv (kv_lora) | k_rope (rope)]；rope 行同样按 _rope_perm 重排。"""
    k, r = cfg.kv_lora_rank, cfg.qk_rope_head_dim
    _, inv = _rope_perm(r)
    return torch.cat([w[:k], w[k:][inv]], dim=0)


def _mla_kv_down_to_hf(w, cfg):
    k, r = cfg.kv_lora_rank, cfg.qk_rope_head_dim
    idx, _ = _rope_perm(r)
    return torch.cat([w[:k], w[k:][idx]], dim=0)


def _attn_to_mcore(hf, src, dst, cfg, out):
    if cfg.attn_type == "mla":
        out[f"{dst}.self_attention.linear_q_down_proj.weight"] = hf[f"{src}.attn.q_down.weight"]
        out[f"{dst}.self_attention.q_layernorm.weight"] = hf[f"{src}.attn.q_norm.weight"]
        out[f"{dst}.self_attention.linear_q_up_proj.weight"] = _mla_q_up_to_mcore(
            hf[f"{src}.attn.q_up_nope.weight"], hf[f"{src}.attn.q_up_rope.weight"], cfg)
        out[f"{dst}.self_attention.linear_kv_down_proj.weight"] = _mla_kv_down_to_mcore(hf[f"{src}.attn.kv_down.weight"], cfg)
        out[f"{dst}.self_attention.kv_layernorm.weight"] = hf[f"{src}.attn.kv_norm.weight"]
        out[f"{dst}.self_attention.linear_kv_up_proj.weight"] = hf[f"{src}.attn.kv_up.weight"]
        out[f"{dst}.self_attention.linear_proj.weight"] = hf[f"{src}.attn.o_proj.weight"]
        return
    out[f"{dst}.self_attention.linear_qkv.weight"] = _qkv_fuse(
        hf[f"{src}.attn.q_proj.weight"], hf[f"{src}.attn.k_proj.weight"], hf[f"{src}.attn.v_proj.weight"], cfg)
    out[f"{dst}.self_attention.linear_proj.weight"] = hf[f"{src}.attn.o_proj.weight"]
    out[f"{dst}.self_attention.q_layernorm.weight"] = hf[f"{src}.attn.q_norm.weight"]
    out[f"{dst}.self_attention.k_layernorm.weight"] = hf[f"{src}.attn.k_norm.weight"]


def _attn_to_hf(mc, src, dst, cfg, out):
    if cfg.attn_type == "mla":
        out[f"{dst}.attn.q_down.weight"] = mc[f"{src}.self_attention.linear_q_down_proj.weight"]
        out[f"{dst}.attn.q_norm.weight"] = mc[f"{src}.self_attention.q_layernorm.weight"]
        out[f"{dst}.attn.q_up_nope.weight"], out[f"{dst}.attn.q_up_rope.weight"] = _mla_q_up_to_hf(
            mc[f"{src}.self_attention.linear_q_up_proj.weight"], cfg)
        out[f"{dst}.attn.kv_down.weight"] = _mla_kv_down_to_hf(mc[f"{src}.self_attention.linear_kv_down_proj.weight"], cfg)
        out[f"{dst}.attn.kv_norm.weight"] = mc[f"{src}.self_attention.kv_layernorm.weight"]
        out[f"{dst}.attn.kv_up.weight"] = mc[f"{src}.self_attention.linear_kv_up_proj.weight"]
        out[f"{dst}.attn.o_proj.weight"] = mc[f"{src}.self_attention.linear_proj.weight"]
        return
    q, k, v = _qkv_split(mc[f"{src}.self_attention.linear_qkv.weight"], cfg)
    out[f"{dst}.attn.q_proj.weight"], out[f"{dst}.attn.k_proj.weight"], out[f"{dst}.attn.v_proj.weight"] = q, k, v
    out[f"{dst}.attn.o_proj.weight"] = mc[f"{src}.self_attention.linear_proj.weight"]
    out[f"{dst}.attn.q_norm.weight"] = mc[f"{src}.self_attention.q_layernorm.weight"]
    out[f"{dst}.attn.k_norm.weight"] = mc[f"{src}.self_attention.k_layernorm.weight"]


def _dense_to_mcore(hf, src, dst, out):
    out[f"{dst}.linear_fc1.weight"] = torch.cat([hf[f"{src}.gate_proj.weight"], hf[f"{src}.up_proj.weight"]], 0)
    out[f"{dst}.linear_fc2.weight"] = hf[f"{src}.down_proj.weight"]


def _dense_to_hf(mc, src, dst, out):
    gate, up = mc[f"{src}.linear_fc1.weight"].chunk(2, dim=0)
    out[f"{dst}.gate_proj.weight"], out[f"{dst}.up_proj.weight"] = gate, up
    out[f"{dst}.down_proj.weight"] = mc[f"{src}.linear_fc2.weight"]


def _moe_to_mcore(hf, src, dst, cfg, out):
    out[f"{dst}.router.weight"] = hf[f"{src}.gate.weight"]
    for e in range(cfg.n_experts):
        out[f"{dst}.experts.local_experts.{e}.linear_fc1.weight"] = torch.cat(
            [hf[f"{src}.w_gate"][e].t(), hf[f"{src}.w_up"][e].t()], 0)
        out[f"{dst}.experts.local_experts.{e}.linear_fc2.weight"] = hf[f"{src}.w_down"][e].t()
    if cfg.n_shared_experts:
        _dense_to_mcore(hf, f"{src}.shared_experts", f"{dst}.shared_experts", out)


def _moe_to_hf(mc, src, dst, cfg, out):
    out[f"{dst}.gate.weight"] = mc[f"{src}.router.weight"]
    gates, ups, downs = [], [], []
    for e in range(cfg.n_experts):
        g, u = mc[f"{src}.experts.local_experts.{e}.linear_fc1.weight"].chunk(2, dim=0)
        gates.append(g.t())
        ups.append(u.t())
        downs.append(mc[f"{src}.experts.local_experts.{e}.linear_fc2.weight"].t())
    out[f"{dst}.w_gate"] = torch.stack(gates).contiguous()
    out[f"{dst}.w_up"] = torch.stack(ups).contiguous()
    out[f"{dst}.w_down"] = torch.stack(downs).contiguous()
    if cfg.n_shared_experts:
        _dense_to_hf(mc, f"{src}.shared_experts", f"{dst}.shared_experts", out)


_RDT_EXTRA = ["norm.weight", "injection.log_A", "injection.log_dt", "injection.B", "act.halt.weight",
              "act.halt.bias", "lora.B", "lora.down.weight", "lora.scale.weight"]


# ---------------------------------------------------------------------------
# 整模型
# ---------------------------------------------------------------------------


def _pad_rows(w, n):
    return w if n is None or n == w.shape[0] else torch.cat([w, w.new_zeros(n - w.shape[0], *w.shape[1:])], 0)


def hf_to_mcore(hf, cfg, vocab_padded=None):
    """vocab_padded：mcore 模型的 padding 后词表大小（model.vocab_padded），embedding / 输出层补零行。"""
    hf = {k: v for k, v in hf.items()}
    out = {"embedding.word_embeddings.weight": _pad_rows(hf["model.embed_tokens.weight"], vocab_padded)}
    if not cfg.tie_word_embeddings:
        out["output_layer.weight"] = _pad_rows(hf["lm_head.weight"], vocab_padded)
    for part, n in (("prelude", cfg.prelude_layers), ("coda", cfg.coda_layers)):
        for i in range(n):
            src, dst = f"model.{part}.{i}", f"{part}.layers.{i}"
            out[f"{dst}.input_layernorm.weight"] = hf[f"{src}.attn_norm.weight"]
            out[f"{dst}.pre_mlp_layernorm.weight"] = hf[f"{src}.ffn_norm.weight"]
            _attn_to_mcore(hf, src, dst, cfg, out)
            if cfg.use_moe:
                _moe_to_mcore(hf, f"{src}.ffn", f"{dst}.mlp", cfg, out)
            else:
                _dense_to_mcore(hf, f"{src}.ffn", f"{dst}.mlp", out)
    if cfg.coda_layers > 0:
        out["coda.final_layernorm.weight"] = hf["model.norm.weight"]
    else:
        out["final_norm.weight"] = hf["model.norm.weight"]
    rs, rd = "model.recurrent", "recurrent"
    out[f"{rd}.attn_norm.weight"] = hf[f"{rs}.attn_norm.weight"]
    out[f"{rd}.ffn_norm.weight"] = hf[f"{rs}.ffn_norm.weight"]
    _attn_to_mcore(hf, rs, rd, cfg, out)
    _moe_to_mcore(hf, f"{rs}.ffn", f"{rd}.mlp", cfg, out)
    for k in _RDT_EXTRA:
        out[f"{rd}.{k}"] = hf[f"{rs}.{k}"]
    return {k: v.contiguous() for k, v in out.items()}


def mcore_to_hf(mc, cfg):
    mc = {k: v for k, v in mc.items() if torch.is_tensor(v)}
    V = cfg.vocab_size  # 去掉词表 padding
    out = {"model.embed_tokens.weight": mc["embedding.word_embeddings.weight"][:V]}
    out["lm_head.weight"] = (out["model.embed_tokens.weight"] if cfg.tie_word_embeddings
                             else mc["output_layer.weight"][:V])
    for part, n in (("prelude", cfg.prelude_layers), ("coda", cfg.coda_layers)):
        for i in range(n):
            src, dst = f"{part}.layers.{i}", f"model.{part}.{i}"
            out[f"{dst}.attn_norm.weight"] = mc[f"{src}.input_layernorm.weight"]
            out[f"{dst}.ffn_norm.weight"] = mc[f"{src}.pre_mlp_layernorm.weight"]
            _attn_to_hf(mc, src, dst, cfg, out)
            if cfg.use_moe:
                _moe_to_hf(mc, f"{src}.mlp", f"{dst}.ffn", cfg, out)
            else:
                _dense_to_hf(mc, f"{src}.mlp", f"{dst}.ffn", out)
    out["model.norm.weight"] = mc["coda.final_layernorm.weight" if cfg.coda_layers > 0 else "final_norm.weight"]
    rs, rd = "recurrent", "model.recurrent"
    out[f"{rd}.attn_norm.weight"] = mc[f"{rs}.attn_norm.weight"]
    out[f"{rd}.ffn_norm.weight"] = mc[f"{rs}.ffn_norm.weight"]
    _attn_to_hf(mc, rs, rd, cfg, out)
    _moe_to_hf(mc, f"{rs}.mlp", f"{rd}.ffn", cfg, out)
    for k in _RDT_EXTRA:
        out[f"{rd}.{k}"] = mc[f"{rs}.{k}"]
    return {k: v.contiguous() for k, v in out.items()}


# ---------------------------------------------------------------------------
# 张量并行切分 / 合并
# ---------------------------------------------------------------------------

_COL = re.compile(r"(linear_qkv|word_embeddings|output_layer|linear_q_up_proj|linear_kv_up_proj"
                  r"|linear_q_down_proj|linear_kv_down_proj)\.weight$")   # 按行（输出维）切；MLA 下投影 gather_output=False
_GLU = re.compile(r"linear_fc1\.weight$")                                  # [gate; up] 各自按行切
_ROW = re.compile(r"(linear_proj|linear_fc2)\.weight$")                    # 按列（输入维）切


def _check_divisible(cfg, n):
    ng = cfg.num_key_value_heads or cfg.num_attention_heads
    if ng % n:
        raise ValueError(f"张量并行 {n} 需要整除 num_key_value_heads={ng}")


def shard_tp(full, cfg, rank, size):
    if size == 1:
        return dict(full)
    _check_divisible(cfg, size)
    out = {}
    for k, v in full.items():
        if _COL.search(k):
            out[k] = v.chunk(size, dim=0)[rank]
        elif _GLU.search(k):
            g, u = v.chunk(2, dim=0)
            out[k] = torch.cat([g.chunk(size, 0)[rank], u.chunk(size, 0)[rank]], 0)
        elif _ROW.search(k):
            out[k] = v.chunk(size, dim=1)[rank]
        else:
            out[k] = v
    return {k: v.contiguous() for k, v in out.items()}


def merge_tp(shards, cfg):
    if len(shards) == 1:
        return dict(shards[0])
    out = {}
    for k in shards[0]:
        vs = [s[k] for s in shards]
        if _COL.search(k):
            out[k] = torch.cat(vs, 0)
        elif _GLU.search(k):
            gs, us = zip(*(v.chunk(2, 0) for v in vs))
            out[k] = torch.cat([torch.cat(gs, 0), torch.cat(us, 0)], 0)
        elif _ROW.search(k):
            out[k] = torch.cat(vs, 1)
        else:
            out[k] = vs[0]
    return out


# ---------------------------------------------------------------------------
# 专家并行：每个 EP rank 只持有 n_experts / EP 个专家，本地下标从 0 开始
# ---------------------------------------------------------------------------

_EXPERT = re.compile(r"(mlp\.experts\.local_experts\.)(\d+)(\.)")


def shard_ep(full, cfg, rank, size):
    if size == 1:
        return dict(full)
    ne = cfg.n_experts
    if ne % size:
        raise ValueError(f"专家并行 {size} 需要整除 n_experts={ne}")
    per = ne // size
    out = {}
    for k, v in full.items():
        m = _EXPERT.search(k)
        if m is None:
            out[k] = v
            continue
        g = int(m.group(2))
        if g // per == rank:  # 全局专家 g 归属本 rank，改写为本地下标
            out[_EXPERT.sub(lambda mm: f"{mm.group(1)}{g - rank * per}{mm.group(3)}", k, count=1)] = v
    return out


def merge_ep(shards, cfg):
    if len(shards) == 1:
        return dict(shards[0])
    per = cfg.n_experts // len(shards)
    out = {}
    for r, sd in enumerate(shards):
        for k, v in sd.items():
            m = _EXPERT.search(k)
            if m is None:
                out.setdefault(k, v)
            else:
                g = int(m.group(2)) + r * per
                out[_EXPERT.sub(lambda mm: f"{mm.group(1)}{g}{mm.group(3)}", k, count=1)] = v
    return out


def shard_parallel(full, cfg, tp_rank, tp_size, ep_rank, ep_size):
    """完整（未切分）state dict → 第 (tp_rank, ep_rank) 个分片。"""
    return shard_ep(shard_tp(full, cfg, tp_rank, tp_size), cfg, ep_rank, ep_size)


def merge_parallel(grid, cfg):
    """grid[tp][ep] = 同一流水线段内各分片 → 该段的 state dict（未合并流水线）。"""
    per_tp = [merge_ep(row, cfg) for row in grid]
    return merge_tp(per_tp, cfg)


# ---------------------------------------------------------------------------
# 流水线并行（2 段）：第一段 = 词嵌入 + Prelude；第二段 = 循环块 + Coda + 输出层
# ---------------------------------------------------------------------------

_FIRST_STAGE_PREFIXES = ("embedding.", "prelude.")


def split_pp(full, cfg, pp_rank, pp_size):
    """完整 state dict → 第 pp_rank 段的 state dict。

    绑定词嵌入（tie）时，第二段的输出层权重是词嵌入的副本（Megatron 的做法），这里直接给它同一份数据。"""
    if pp_size == 1:
        return dict(full)
    if pp_size != 2:
        raise ValueError("流水线并行目前只支持 1 或 2 段（循环块不能跨段拆分）")
    if pp_rank == 0:
        return {k: v for k, v in full.items() if k.startswith(_FIRST_STAGE_PREFIXES)}
    out = {k: v for k, v in full.items()
           if not k.startswith(_FIRST_STAGE_PREFIXES) and not k.startswith("output_layer.")}
    out["output_layer.weight"] = (full["embedding.word_embeddings.weight"] if cfg.tie_word_embeddings
                                  else full["output_layer.weight"])
    return out


def merge_pp(stages, cfg):
    """各段（已合并 TP/EP）的 state dict → 完整 state dict。绑定词嵌入时丢弃第二段的副本。"""
    if len(stages) == 1:
        return dict(stages[0])
    out = dict(stages[0])
    for k, v in stages[1].items():
        if cfg.tie_word_embeddings and k == "output_layer.weight":
            continue
        out[k] = v
    return out


def model_tensors(model):
    """mcore 模型的参数（去掉 Megatron 的 _extra_state 等非张量项），已 detach 到 CPU。"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if torch.is_tensor(v) and not k.endswith("_extra_state")}
