"""
BaiZe RDT —— Megatron-core 实现
===============================

与 ``baize/model.py`` 同一个架构、同一套数学，换成 Megatron-core 的并行组件：

    tokens → [Prelude × P] → [Recurrent Block × T 圈] → [Coda × C] → logits

=========================  ==============================================================
原版（baize/model.py）       Megatron-core 版（本文件）
=========================  ==============================================================
nn.Embedding / lm_head     LanguageModelEmbedding（VocabParallelEmbedding）/ ColumnParallelLinear，
                           权重共享；loss 用 vocab_parallel_cross_entropy（词表切分也能算）
Prelude / Coda Block       TransformerBlock（local spec：GQA + QK-Norm + SwiGLU / MoE）
循环块 attention / MoE      同一个 TransformerLayer 的 SelfAttention、MoELayer 子模块，
                           按 RDT 公式组装（attn 与 FFN 并行作用在 norm(h+e) 上，再经 LTI 注入）
LTI / LoRA / ACT / 圈数嵌入  直接复用原版模块（参数在张量并行组内复制）
激活重计算                  TransformerBlock 的 full recompute + 循环块每圈 tensor_parallel.checkpoint
=========================  ==============================================================

张量布局沿用 Megatron 约定：隐状态为 [s, b, h]。
支持：张量并行（TP，自动开启序列并行）、数据并行（DP，Megatron DDP / 分布式优化器）。
不支持：流水线并行（循环块要在同一 stage 内迭代）、MLA（attn_type="mla"）。
"""

import dataclasses
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compat import apply_compat_patches, cpu_smoke_mode

apply_compat_patches()

from megatron.core import tensor_parallel  # noqa: E402
from megatron.core.enums import ModelType  # noqa: E402
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec  # noqa: E402
from megatron.core.transformer.module import MegatronModule  # noqa: E402
from megatron.core.transformer.spec_utils import build_module  # noqa: E402
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from ..model import ACTHalting, LoRAAdapter, LTIInjection, loop_index_embedding  # noqa: E402


# ---------------------------------------------------------------------------
# 配置映射：BaiZeConfig → Megatron TransformerConfig
# ---------------------------------------------------------------------------


def build_transformer_configs(cfg, *, tensor_parallel_size=1, expert_parallel_size=1, bf16=False,
                              n_loops_train=None, grad_checkpoint=False, moe_dispatcher="allgather"):
    """返回 (dense_cfg, moe_cfg)。

    dense_cfg —— Prelude / Coda（use_moe=False 时为稠密 SwiGLU）
    moe_cfg   —— 循环块（恒为 MoE）；Prelude/Coda 的 use_moe=True 时也用它

    MoE aux-loss：Megatron 的 MoELayer 每调用一次就把 coeff·aux 挂到反向图上，循环块每圈调用一次，
    所以系数除以训练圈数，等价于原版"各圈 aux 取平均再乘系数"。
    """
    if cfg.attn_type != "gqa":
        raise NotImplementedError("Megatron-core 版目前只实现 GQA（attn_type='gqa'）；MLA 请用原版训练")
    if cfg.hidden_act != "silu":
        raise NotImplementedError("Megatron-core 版只实现 SwiGLU（hidden_act='silu'）")
    n_loops = n_loops_train or cfg.max_loop_iters
    dtype = torch.bfloat16 if bf16 else torch.float32
    common = dict(
        num_layers=1,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_query_groups=cfg.num_key_value_heads or cfg.num_attention_heads,
        kv_channels=cfg.head_dim,
        ffn_hidden_size=cfg.intermediate_size,
        normalization="RMSNorm",
        layernorm_epsilon=cfg.rms_norm_eps,
        qk_layernorm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        hidden_dropout=0.0,
        attention_dropout=cfg.dropout,
        init_method_std=getattr(cfg, "initializer_range", None) or 0.02,
        bf16=bf16,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        tensor_model_parallel_size=tensor_parallel_size,
        expert_model_parallel_size=expert_parallel_size,
        sequence_parallel=tensor_parallel_size > 1,  # Megatron 要求 MoE + TP 时开启；层间隐状态沿序列切分
        use_cpu_initialization=cpu_smoke_mode(),
        gradient_accumulation_fusion=False,
        bias_activation_fusion=False,
        masked_softmax_fusion=False,
        persist_layer_norm=False,
        apply_rope_fusion=False,
    )
    if grad_checkpoint:
        common.update(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
    moe = dict(
        num_moe_experts=cfg.n_experts,
        moe_router_topk=cfg.n_experts_per_tok,
        moe_ffn_hidden_size=cfg.moe_intermediate_size,
        moe_shared_expert_intermediate_size=cfg.moe_intermediate_size * cfg.n_shared_experts
        if cfg.n_shared_experts else None,
        moe_router_load_balancing_type="aux_loss" if cfg.router_aux_loss_coef > 0 else "none",
        moe_aux_loss_coeff=cfg.router_aux_loss_coef / n_loops,
        moe_token_dispatcher_type=moe_dispatcher,
        moe_grouped_gemm=False,
        moe_router_dtype="fp32",
        moe_expert_capacity_factor=cfg.moe_capacity_factor if cfg.moe_capacity_factor > 0 else None,
    )
    dense_cfg = TransformerConfig(**common)
    moe_cfg = TransformerConfig(**common, **moe)
    return dense_cfg, moe_cfg


def padded_vocab_size(vocab_size: int, tensor_parallel_size: int, multiple: int = 128) -> int:
    """词表 padding 到 multiple × TP 的整数倍（Megatron 的 make_vocab_size_divisible_by 做法），
    让 VocabParallelEmbedding / 输出层能按 TP 均分。padding 出来的 logits 在 loss 前置为 -inf，不影响数值。"""
    m = multiple * tensor_parallel_size
    return (vocab_size + m - 1) // m * m


class RMSNormSP(nn.RMSNorm):
    """Megatron spec 里的 norm：即 torch.nn.RMSNorm（与 megatron 无 apex/TE 时的回退实现数值相同），
    额外支持序列并行——megatron 的 WrappedTorchNorm 在 sequence_parallel=True 时直接断言失败。"""

    def __init__(self, config, hidden_size, eps=1e-5, **kwargs):
        super().__init__(hidden_size, eps=eps)
        if config.sequence_parallel:
            self.weight.sequence_parallel = True  # 梯度在 TP 组内求和（finalize_model_grads）


def _layer_spec(use_moe: bool, num_experts: int):
    spec = get_gpt_layer_local_spec(num_experts=num_experts if use_moe else None, moe_grouped_gemm=False,
                                    qk_layernorm=True, normalization="RMSNorm")
    sub = spec.submodules
    attn = sub.self_attention
    attn = dataclasses.replace(attn, submodules=dataclasses.replace(
        attn.submodules, q_layernorm=RMSNormSP, k_layernorm=RMSNormSP))
    sub = dataclasses.replace(sub, input_layernorm=RMSNormSP, pre_mlp_layernorm=RMSNormSP, self_attention=attn)
    return dataclasses.replace(spec, submodules=sub)


def _block_spec(use_moe: bool, num_experts: int, num_layers: int):
    return TransformerBlockSubmodules(layer_specs=[_layer_spec(use_moe, num_experts)] * num_layers,
                                      layer_norm=RMSNormSP)


def rope_freqs(seq_len: int, dim: int, theta: float, device, offset: int = 0):
    """Megatron 格式的 RoPE 频率 [s, 1, 1, dim]（与原版 precompute_freqs_cis 同一组频率、同为非交错式）。
    不用 megatron 的 RotaryEmbedding：它的 forward 写死了 CUDA 设备。"""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(offset, offset + seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv)
    return torch.cat([freqs, freqs], dim=-1)[:, None, None, :]


def causal_mask(seq_len: int, device):
    """Megatron 约定：True = 屏蔽。形状 [1, 1, s, s]。"""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)[None, None]


# ---------------------------------------------------------------------------
# 循环块
# ---------------------------------------------------------------------------


class MCoreRecurrentBlock(nn.Module):
    """单个 Transformer 层循环 T 圈（语义与 baize.model.RecurrentBlock 一致）。

    每圈：圈数嵌入 → x = norm(h+e) → f = attn(attn_norm(x)) + moe(ffn_norm(x)) + LoRA_t(x)
          → h' = A·h + B·e + f → ACT 累积
    attention / MoE / 两个 norm 取自 Megatron TransformerLayer 的子模块（张量 / 专家并行）。
    """

    def __init__(self, cfg, moe_cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        layer = build_module(_layer_spec(True, cfg.n_experts), config=moe_cfg, layer_number=1)
        self.attn_norm = layer.input_layernorm
        self.self_attention = layer.self_attention
        self.ffn_norm = layer.pre_mlp_layernorm
        self.mlp = layer.mlp
        self.norm = RMSNormSP(moe_cfg, cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.injection = LTIInjection(cfg.hidden_size)
        self.act = ACTHalting(cfg.hidden_size)
        self.lora = LoRAAdapter(cfg.hidden_size, cfg.lora_rank, cfg.max_loop_iters)
        self.loop_dim = max(2, int(cfg.hidden_size * cfg.loop_emb_frac))
        self.sequence_parallel = moe_cfg.sequence_parallel
        self.grad_checkpointing = False
        self.ponder_cost = None   # 计入 loss 的 ponder（序列并行时已按 TP 均摊，见 forward）
        self.ponder_log = None    # 全局 ponder 均值（只用于日志）
        self.avg_loops = None

    def _loop_step(self, h, e, rotary, mask, t: int):
        h_loop = loop_index_embedding(h, t, self.loop_dim)
        x = self.norm(h_loop + e)
        attn_out, _ = self.self_attention(self.attn_norm(x), attention_mask=mask, rotary_pos_emb=rotary)
        mlp_out, _ = self.mlp(self.ffn_norm(x))
        f = attn_out + mlp_out + self.lora(x, t)
        return self.injection(h, e, f)

    def forward(self, h, e, rotary, mask, n_loops=None):
        n_loops = n_loops or self.cfg.max_loop_iters
        use_act = self.cfg.use_act
        s, b, _ = h.shape
        halted = torch.zeros(s, b, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(s, b, device=h.device)
        n_updates = torch.zeros(s, b, device=h.device)
        remainders = torch.zeros(s, b, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            if self.grad_checkpointing and self.training:
                h = tensor_parallel.checkpoint(functools.partial(self._loop_step, t=t), False, h, e, rotary, mask)
            else:
                h = self._loop_step(h, e, rotary, mask, t)
            if use_act:
                p = self.act(h)
                still_running = (~halted).float()
                remainder = (1.0 - cumulative_p).clamp(min=0)
                stop_now = (cumulative_p + p >= self.cfg.act_threshold) | (t == n_loops - 1)
                weight = torch.where(stop_now, remainder, p) * still_running
                h_out = h_out + weight.unsqueeze(-1).to(h.dtype) * h
                n_updates = n_updates + still_running
                remainders = remainders + torch.where(stop_now, remainder, torch.zeros_like(remainder)) * still_running
                cumulative_p = cumulative_p + p * still_running
                halted = halted | stop_now
                if self._all_halted(halted):
                    break
            else:
                h_out = h
                n_updates = n_updates + 1

        ponder = (n_updates + remainders).mean() if use_act else h.new_zeros(())
        avg_loops = n_updates.mean().detach()
        if self.sequence_parallel:
            # 每个 TP rank 只持有一段序列：复制参数的梯度会在 TP 组内求和（sequence_parallel 标记），
            # 所以 loss 里的 ponder 先除以 TP，求和后恰好是全局均值
            from megatron.core import parallel_state

            tp = parallel_state.get_tensor_model_parallel_world_size()
            stats = torch.stack([ponder.detach(), avg_loops]).float()
            torch.distributed.all_reduce(stats, group=parallel_state.get_tensor_model_parallel_group())
            self.ponder_cost, self.ponder_log, self.avg_loops = ponder / tp, stats[0] / tp, stats[1] / tp
        else:
            self.ponder_cost, self.ponder_log, self.avg_loops = ponder, ponder.detach(), avg_loops
        return h_out

    def _all_halted(self, halted):
        """序列并行时各 rank 只看到自己那段序列，必须在 TP 组内取一致的结论，否则各 rank 圈数不同会在集合通信处死锁。"""
        done = halted.all()
        if not self.sequence_parallel:
            return bool(done)
        from megatron.core import parallel_state

        flag = done.to(torch.int32).reshape(1)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN,
                                     group=parallel_state.get_tensor_model_parallel_group())
        return bool(flag.item())


# ---------------------------------------------------------------------------
# 完整模型
# ---------------------------------------------------------------------------


class BaiZeMCoreModel(MegatronModule):
    """Megatron-core 版 BaiZe 因果语言模型。

    forward(input_ids [b, s], labels [b, s] = None, n_loops=None)
        labels 为空：返回 logits [b, s, vocab]（TP>1 时为本 rank 的词表分片，已去掉 padding 列）
        labels 非空：返回 (token_loss [b, s], loss_mask [b, s])；labels 与 input 对齐（模型内部做 shift），
                     -100 处不计 loss
    """

    def __init__(self, cfg, dense_cfg: TransformerConfig, moe_cfg: TransformerConfig):
        super().__init__(config=moe_cfg)  # schedules 通过 model.config 读取 MoE / 梯度同步设置
        self.model_type = ModelType.encoder_or_decoder
        self.cfg = cfg
        self.vocab_padded = padded_vocab_size(cfg.vocab_size, dense_cfg.tensor_model_parallel_size)
        self.embedding = LanguageModelEmbedding(
            config=dense_cfg, vocab_size=self.vocab_padded, max_sequence_length=cfg.max_position_embeddings,
            position_embedding_type="rope")
        # Prelude / Coda 的 MoE 每次前向只调用一次：aux 系数用原值（循环块的 moe_cfg 已按圈数均摊）
        side_cfg = (dataclasses.replace(moe_cfg, moe_aux_loss_coeff=cfg.router_aux_loss_coef) if cfg.use_moe
                    else dense_cfg)
        self.prelude = (TransformerBlock(dataclasses.replace(side_cfg, num_layers=cfg.prelude_layers),
                                         _block_spec(cfg.use_moe, cfg.n_experts, cfg.prelude_layers),
                                         post_layer_norm=False) if cfg.prelude_layers > 0 else None)
        self.recurrent = MCoreRecurrentBlock(cfg, moe_cfg)
        # Coda 的 final_layernorm 即原版 BaiZeModel.norm（coda_layers=0 时单独建一个 norm）
        self.coda = (TransformerBlock(dataclasses.replace(side_cfg, num_layers=cfg.coda_layers),
                                      _block_spec(cfg.use_moe, cfg.n_experts, cfg.coda_layers),
                                      post_layer_norm=True) if cfg.coda_layers > 0 else None)
        self.final_norm = None if self.coda is not None else RMSNormSP(dense_cfg, cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.output_layer = tensor_parallel.ColumnParallelLinear(
            cfg.hidden_size, self.vocab_padded, config=dense_cfg, init_method=dense_cfg.init_method, bias=False,
            skip_bias_add=False, gather_output=False,
            skip_weight_param_allocation=cfg.tie_word_embeddings)
        self._init_rdt_params()
        if moe_cfg.sequence_parallel:
            # 这些参数在 TP 组内复制、作用在本 rank 的序列分片上：梯度需要在 TP 组内求和
            for mod in (self.recurrent.injection, self.recurrent.act, self.recurrent.lora):
                for p in mod.parameters():
                    p.sequence_parallel = True

    @torch.no_grad()
    def _init_rdt_params(self):
        """原版 HF 初始化：Linear / Embedding ~ N(0, 0.02)，bias 置 0；ACT 停机偏置单独设为负值。"""
        std = self.config.init_method_std
        for m in (self.recurrent.lora.down, self.recurrent.lora.scale, self.recurrent.act.halt):
            nn.init.normal_(m.weight, mean=0.0, std=std)
        self.recurrent.act.halt.bias.fill_(self.cfg.act_init_bias)

    def set_input_tensor(self, input_tensor):
        """流水线并行接口（本模型只支持 PP=1，无需处理）。"""

    def enable_grad_checkpointing(self, enabled: bool = True):
        self.recurrent.grad_checkpointing = enabled

    def shared_embedding_or_output_weight(self):
        return self.embedding.word_embeddings.weight

    def forward(self, input_ids, labels=None, n_loops=None):
        b, s = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        h = self.embedding(input_ids=input_ids, position_ids=position_ids)          # [s, b, h]
        rotary = rope_freqs(s, self.cfg.head_dim, self.cfg.rope_theta, device)
        mask = causal_mask(s, device)

        if self.prelude is not None:
            h = self.prelude(h, attention_mask=mask, rotary_pos_emb=rotary)
        e = h
        h = self.recurrent(h, e, rotary, mask, n_loops=n_loops)
        if self.coda is not None:
            h = self.coda(h, attention_mask=mask, rotary_pos_emb=rotary)
        else:
            h = self.final_norm(h)

        weight = self.shared_embedding_or_output_weight() if self.cfg.tie_word_embeddings else None
        logits, _ = self.output_layer(h, weight=weight)                             # [s, b, v_pad/TP]
        logits, n_real = self._mask_vocab_padding(logits)
        if labels is None:
            return logits[..., :n_real].transpose(0, 1).contiguous()

        # next-token：位置 t 预测 labels[t+1]；最后一个位置和 -100 不计 loss
        shifted = torch.cat([labels[:, 1:], labels.new_full((b, 1), -100)], dim=1)
        loss_mask = shifted != -100
        target = torch.where(loss_mask, shifted, torch.zeros_like(shifted))
        token_loss = tensor_parallel.vocab_parallel_cross_entropy(logits.float(), target.t().contiguous())
        return token_loss.t().contiguous(), loss_mask

    def _mask_vocab_padding(self, logits):
        """把本 rank 词表分片里超出真实词表的列置为 -inf。返回 (logits, 本分片真实词表列数)。"""
        from megatron.core import parallel_state

        local = logits.shape[-1]
        start = parallel_state.get_tensor_model_parallel_rank() * local
        n_real = max(0, min(local, self.cfg.vocab_size - start))
        if n_real < local:
            pad = torch.arange(local, device=logits.device) >= n_real
            logits = logits.masked_fill(pad, float("-inf"))
        return logits, n_real
