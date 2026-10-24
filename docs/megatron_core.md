# Megatron-core 版训练

`scripts/pretrain_mcore.py` 用 [Megatron-core](https://github.com/NVIDIA/Megatron-LM)（`megatron-core`）重写了 BaiZe 的预训练。
模型结构、数据、超参语义都和原版 `scripts/pretrain.py` 完全一致。训练结束后导出的权重是原版格式，
`eval.py`、`demo.py`、`sft.py` 可以直接使用。

原版训练（`baize/trainer.py`，DDP / FSDP2）继续保留。小规模训练和 SFT / DPO 仍然用原版；需要张量并行的大规模预训练用 Megatron-core 版。

---

## 安装

```bash
pip install "megatron-core==0.16.1"
# 可选，GPU 上加速：Transformer Engine / apex（本实现不依赖它们）
```

兼容补丁（`baize/mcore/compat.py`）是针对 megatron-core **0.16.1** 编写并测试的，换版本前请先跑一遍 `python tests/test_mcore.py`。

---

## 用法

```bash
# 单卡
python scripts/pretrain_mcore.py --data data/pretrain.bin --tokenizer tokenizer --save_dir out \
    --hidden_size 1024 --num_attention_heads 16 --num_key_value_heads 4 --head_dim 64 \
    --intermediate_size 2816 --prelude_layers 4 --coda_layers 4 \
    --max_loop_iters 8 --n_loops_train 4 --moe_intermediate_size 704 \
    --max_seq_len 512 --batch_size 32 --accumulation_steps 16 --learning_rate 6e-4 --bf16 1

# 8 卡：张量并行 2 × 数据并行 4，分布式优化器
torchrun --nproc_per_node 8 scripts/pretrain_mcore.py ...同上... \
    --tensor_parallel_size 2 --use_distributed_optimizer 1

# 8 卡：专家并行 4 × 数据并行 2（MoE 专家切到 4 个 rank 上）
torchrun --nproc_per_node 8 scripts/pretrain_mcore.py ...同上... --expert_parallel_size 4

# 4 卡：流水线并行 2 × 数据并行 2（第一段：词嵌入 + Prelude；第二段：循环块 + Coda + 输出层）
torchrun --nproc_per_node 4 scripts/pretrain_mcore.py ...同上... --pipeline_parallel_size 2

# MLA 注意力（与原版 pretrain.py 的 --attn_type mla 相同），可与上述并行方式任意组合
torchrun --nproc_per_node 8 scripts/pretrain_mcore.py ...同上... --attn_type mla --expert_parallel_size 4

# 中断后续训（原命令加 --from_resume 1）
torchrun --nproc_per_node 8 scripts/pretrain_mcore.py ...同上... --from_resume 1

# 从原版权重继续训练（自动按 TP 切分）
python scripts/pretrain_mcore.py ... --from_weight pretrain
```

数据参数和模型参数与 `pretrain.py` 完全相同（两个脚本共用 `add_data_model_args`）。Megatron-core 版特有的参数如下：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--tensor_parallel_size` | 1 | 张量并行度。>1 时自动开启**序列并行**（Megatron 要求 MoE + TP 必须开）；要求能整除 `num_key_value_heads` |
| `--expert_parallel_size` | 1 | 专家并行度。每个 EP rank 持有 `n_experts / EP` 个专家（要求整除）；dense 部分仍在 DP 组内复制 |
| `--pipeline_parallel_size` | 1 | 流水线并行度，**只支持 1 或 2**。两段：词嵌入 + Prelude | 循环块 + Coda + 输出层；循环块的全部圈数都在同一段内完成。词嵌入与输出层权重在两段之间绑定并同步梯度 |
| `--moe_dispatcher` | allgather | MoE token 分发方式：`allgather`（CPU 冒烟可测）或 `alltoall`（EP 通信更省；仅 GPU 上可用，CPU 冒烟未测） |
| `--attn_type` | gqa | `gqa` 或 `mla`（与原版相同）。MLA 的 `kv_lora_rank` / `q_lora_rank` / `qk_*_head_dim` / `v_head_dim` 使用 `BaiZeConfig` 的默认值 |
| `--use_distributed_optimizer` | 0 | 优化器状态在 DP 组内切分（ZeRO-1） |
| `--batch_size` | 8 | 每个 DP rank 的 micro-batch 大小；每步样本数 = batch_size × accumulation_steps × DP |
| `--bf16` | 1 | bf16 计算 + fp32 主权重；`--cpu_smoke 1` 时强制关闭 |
| `--grad_checkpoint` | 0 | 激活重计算：Prelude/Coda 用 Megatron full recompute，循环块每圈用 `tensor_parallel.checkpoint` |
| `--exit_interval` | — | 到达该步数的倍数时存断点并退出（同 Megatron-LM `--exit-interval`），用于分段调度 |
| `--save_interval` | 500 | 断点间隔，只保留最新一份 |
| `--cpu_smoke` | 0 | 无 GPU 的冒烟测试：CPU + gloo（见下文） |
| `--use_wandb` 等 | — | 与原版相同 |

并行参数会在启动前检查：`pipeline_parallel_size` 只能是 1 或 2；`tensor_parallel_size` 要能整除 `num_key_value_heads`；
`expert_parallel_size` 要能整除 `n_experts`；`WORLD_SIZE` 要能被 `tp × pp × ep` 整除。

断点与导出都按 `(TP, EP, PP)` 坐标拆分：每个坐标只由一个 rank 写文件（专家参数在 expert-DP 组内复制，所以取 expert-DP rank 0），
因此 EP / PP 的断点和导出权重都是完整的。续训时并行方式必须与保存时一致（`latest.json` 里记录了 tp / ep / pp）。

产物：

```
out/mcore_ckpt_pretrain/latest.json              # 指向最新的完整断点
out/mcore_ckpt_pretrain/step_0001000/tp00_ep00_pp00.pt   # 每个 (TP, EP, PP) 坐标一份（模型 + 优化器 + 步数）
out/mcore_ckpt_pretrain/step_0001000/tp00_ep00_pp00_distopt.pt   # 分布式优化器的参数状态
out/pretrain.safetensors + out/config.json       # 训练结束导出的原版格式权重（合并 TP 分片、去掉词表 padding）
```

---

## 实现对照

| 原版（`baize/model.py`、`baize/trainer.py`） | Megatron-core 版（`baize/mcore/`、`scripts/pretrain_mcore.py`） |
|---|---|
| `nn.Embedding` / `lm_head`（共享权重） | `LanguageModelEmbedding`（`VocabParallelEmbedding`）/ `ColumnParallelLinear`，共享权重；词表 padding 到 128×TP 的倍数，padding 列在 loss 前置为 −inf |
| 交叉熵 | `tensor_parallel.vocab_parallel_cross_entropy`（词表切分也能算） |
| Prelude / Coda 的 `Block` | `TransformerBlock`（local spec：GQA + QK-Norm + SwiGLU / MoE）；Coda 的 `final_layernorm` 就是原版的 `model.norm` |
| 循环块的 attention / MoE | 同一个 `TransformerLayer` 的 `SelfAttention`、`MoELayer` 子模块，按 RDT 公式组装：`f = attn(norm₁(x)) + moe(norm₂(x)) + LoRA_t(x)`，`h' = A·h + B·e + f` |
| LTI 注入 / 深度 LoRA / ACT / 圈数嵌入 | 直接复用原版模块（参数在 TP 组内复制；序列并行时梯度在 TP 组内求和） |
| `MoEFFN`（合并专家张量） | `MoELayer`（router + 逐专家 MLP + 共享专家，allgather 分发） |
| DDP / FSDP2 | `megatron.core.distributed.DistributedDataParallel`（梯度 bucket）+ 可选分布式优化器 |
| `torch.optim.AdamW` | `get_megatron_optimizer`（AdamW，norm / bias / 1 维参数不做 decay，梯度裁剪） |
| 手写梯度累积 | `get_forward_backward_func`（micro-batch 调度，DDP no_sync） |
| warmup + 余弦 | 同一个 `get_lr`，每步按 Megatron 的 `lr_mult` / `wd_mult` 约定写入参数组 |
| `ResumableSampler` | 同一个采样器，按 DP rank 切分；续训从中断处下一个 batch 继续 |

权重互转见 `baize/mcore/convert.py`：
- `hf_to_mcore` / `mcore_to_hf`：原版 HF 格式与 Megatron 格式互转；
- `shard_tp` / `merge_tp`：张量并行切分与合并。QKV 按 query group 切；GLU 的 fc1 把 gate、up 分别切；词表按行切。

---

## 验证

`python tests/test_mcore.py` 运行以下全部检查（GQA 与 MLA 两种注意力），CI 里也有单独的 `megatron-core` job。

| 检查 | 结果 |
|---|---|
| 原版模型随机初始化 → 转换 → Megatron 模型：logits | 最大误差约 1e-7（fp32 舍入），dense / MoE Prelude、ACT 跑满 / 提前停机都覆盖 |
| 训练模式下的 loss 与全部参数梯度 | 相对误差 < 1e-5 |
| 权重往返（原版 → Megatron → 原版）、TP 切分 / 合并往返 | 逐位相等 |
| 同一初始权重、同一数据顺序训练，对比原版 `pretrain.py` 的 loss 轨迹 | TP=1、TP=2（序列并行）、DP=2、激活重计算、中断续训（TP=1 / TP=2）：每步 loss 与原版相同（4 位小数） |
| ACT 延后启用 + 提前停机，TP=2 | TP=1 与 TP=2 完全一致；与原版相差约 1e-4，原因见下方"已知差异" |
| TP=2 训练后导出的权重 | 与 TP=1 导出的一致（fp16 存盘误差约 3e-5），可严格加载进原版模型，`eval.py` 正常运行 |
| 流水线并行 PP=2 | loss 与原版逐步一致；导出权重与 TP=1 一致（3e-5）；中断续训与不中断一致 |
| 专家并行 EP=2 | loss 与 DP=2（同 batch）一致；导出权重与 DP=2 一致（3e-5）；中断续训与不中断一致 |
| MLA（`--attn_type mla`） | TP=1 与原版 loss 一致；TP=2 / PP=2 / EP=2 与 TP=1 一致（权重 3e-5）；与原版的层级输出误差约 1e-7 |

### 已知差异

**MoE aux-loss 在循环提前退出时的均摊方式。** 训练时，如果一个 micro-batch 的所有 token 都提前停机，循环会提前退出：
- 原版：按**实际执行的圈数**对各圈 aux 取平均；
- Megatron 版：每圈的系数固定为 `router_aux_loss_coef / n_loops_train`。

这个系数在前向时就固化在 Megatron 的 `MoEAuxLossAutoScaler` 里。不提前退出时两者完全相同；提前退出时 loss 差约 1e-4 量级。

---

## 限制与注意事项

- **没有在 GPU 上运行过**：开发环境没有 GPU，上面的验证都是在 CPU + gloo 的冒烟模式下完成的。以下几项在 CPU 上无法测试，请在 GPU 上先用 `--max_steps 50` 小规模确认：
  - bf16 训练路径；
  - 分布式优化器（Megatron 的 `DistributedOptimizer` 只接受 CUDA 张量）；
  - NCCL 通信；
  - `--moe_dispatcher alltoall`：CPU 上 torch 的 `all_to_all_single` 与 megatron-core 0.16.1 的参数类型不兼容，只能在 GPU 上用。
- **CPU 冒烟模式**（`--cpu_smoke 1`）：megatron-core 有约 50 处写死了 CUDA 设备、RNG 和梯度裁剪的断言。冒烟模式把这些调用重定向到 CPU，只用于测试和 CI。GPU 上不要开启，检测到 GPU 时会直接报错。
- **流水线并行只支持 2 段**：`--pipeline_parallel_size` 为 1 或 2。循环块在一段内完整迭代 T 圈，所以循环块不能被切开；更多段需要把 Coda 也拆分，目前没有实现。
- **MLA 的维度不能通过命令行设置**：`kv_lora_rank`、`q_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim` 取 `BaiZeConfig` 的默认值（与原版相同）。
- **MLA 的 RoPE 行顺序**：Megatron 对 RoPE 通道做了去交错，转换时会对 `linear_q_up_proj` 与 `linear_kv_down_proj` 的 RoPE 行做相应的置换（`baize/mcore/convert.py` 的 `_rope_perm`），导出时反置换，所以导出的权重是原版格式。
- 张量并行度必须能整除 `num_key_value_heads`。
- 训练步的 MoE 辅助 loss 统计（Megatron 的全局 aux-loss 追踪表）只用于 Megatron 内部日志，训练脚本不读取它；`baize/mcore/model.py` 按最大层数预先定尺寸，避免各子块共用同一张表时越界。
- SFT / DPO 仍然使用原版 Trainer，可以直接接 Megatron 版导出的 `pretrain.safetensors`。
