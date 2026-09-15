"""
BaiZe — 配置定义
=================
BaiZeConfig 继承 transformers.PretrainedConfig，
可直接使用 AutoConfig / save_pretrained / from_pretrained 等 HF 生态接口。

架构：Recurrent-Depth Transformer（RDT）
    tokens → [Prelude × P] → [Recurrent Block × T 圈] → [Coda × C] → logits

设计要点：
    1. tokenizer 词表与 config.vocab_size 强绑定：BaiZeTokenizer.from_pretrained
       读出真实词表大小后直接传入 config，杜绝两端不一致导致的 embedding 越界。
    2. MoE 负载均衡采用显式 aux-loss（L = Σ f_i·P_i），训练时真实接入优化器。
    3. 默认 GQA + SDPA，可选 MLA（attn_type="mla"）。
"""

from transformers import PretrainedConfig


class BaiZeConfig(PretrainedConfig):
    model_type = "baize"

    def __init__(
        self,
        # ---- 词表 / 尺寸 ----
        vocab_size: int = 6400,
        hidden_size: int = 512,
        max_position_embeddings: int = 32768,
        # ---- RDT 三段式 ----
        prelude_layers: int = 2,        # 循环前的一次性层
        coda_layers: int = 2,           # 循环后的一次性层
        max_loop_iters: int = 8,        # 默认循环圈数 T
        # ---- 注意力 ----
        attn_type: str = "gqa",         # "gqa" | "mla"
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,   # GQA 分组
        head_dim: int = 64,
        rope_theta: float = 1e6,
        # MLA 参数（attn_type="mla" 时生效）
        kv_lora_rank: int = 128,
        q_lora_rank: int = 256,
        qk_rope_head_dim: int = 32,
        qk_nope_head_dim: int = 32,
        v_head_dim: int = 32,
        # ---- MoE（use_moe=True 时替换所有 FFN；循环块内恒为 MoE）----
        use_moe: bool = False,
        n_experts: int = 8,
        n_shared_experts: int = 1,
        n_experts_per_tok: int = 2,
        moe_intermediate_size: int = 512,
        router_aux_loss_coef: float = 1e-3,
        # ---- 循环块机制 ----
        lora_rank: int = 8,             # 深度 LoRA 秩
        act_threshold: float = 0.99,    # ACT 停机阈值
        use_act: bool = True,           # False 时退化为"跑满 T 圈取加权平均"
        loop_emb_frac: float = 0.125,   # 接受圈数嵌入的通道比例（D/8）
        # ---- 常规 ----
        intermediate_size: int = 1024,  # 稠密 SwiGLU 中间维
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings

        self.prelude_layers = prelude_layers
        self.coda_layers = coda_layers
        self.max_loop_iters = max_loop_iters

        self.attn_type = attn_type
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.rope_theta = rope_theta
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim

        self.use_moe = use_moe
        self.n_experts = n_experts
        self.n_shared_experts = n_shared_experts
        self.n_experts_per_tok = n_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.router_aux_loss_coef = router_aux_loss_coef

        self.lora_rank = lora_rank
        self.act_threshold = act_threshold
        self.use_act = use_act
        self.loop_emb_frac = loop_emb_frac

        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
