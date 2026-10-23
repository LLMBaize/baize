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

# 中断后续训（原命令加 --from_resume 1）
torchrun --nproc_per_node 8 scripts/pretrain_mcore.py ...同上... --from_resume 1

# 从原版权重继续训练（自动按 TP 切分）
python scripts/pretrain_mcore.py ... --from_weight pretrain
```

数据参数和模型参数与 `pretrain.py` 完全相同（两个脚本共用 `add_data_model_args`）。Megatron-core 版特有的参数如下：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--tensor_parallel_size` | 1 | 张量并行度。>1 时自动开启**序列并行**（Megatron 要求 MoE + TP 必须开）；要求能整除 `num_key_value_heads` |
| `--use_distributed_optimizer` | 0 | 优化器状态在 DP 组内切分（ZeRO-1） |
| `--batch_size` | 8 | 每个 DP rank 的 micro-batch 大小；每步样本数 = batch_size × accumulation_steps × DP |
| `--bf16` | 1 | bf16 计算 + fp32 主权重；`--cpu_smoke 1` 时强制关闭 |
| `--grad_checkpoint` | 0 | 激活重计算：Prelude/Coda 用 Megatron full recompute，循环块每圈用 `tensor_parallel.checkpoint` |
| `--exit_interval` | — | 到达该步数的倍数时存断点并退出（同 Megatron-LM `--exit-interval`），用于分段调度 |
| `--save_interval` | 500 | 断点间隔，只保留最新一份 |
| `--cpu_smoke` | 0 | 无 GPU 的冒烟测试：CPU + gloo（见下文） |
| `--use_wandb` 等 | — | 与原版相同 |

产物：

```
out/mcore_ckpt_pretrain/latest.json              # 指向最新的完整断点
out/mcore_ckpt_pretrain/step_0001000/tp00.pt     # 每个 TP 分片一份（模型 + 优化器 + 步数）
out/mcore_ckpt_pretrain/step_0001000/tp00_distopt.pt   # 分布式优化器的参数状态
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

`python tests/test_mcore.py` 运行以下全部检查，CI 里也有单独的 `megatron-core` job。

| 检查 | 结果 |
|---|---|
| 原版模型随机初始化 → 转换 → Megatron 模型：logits | 最大误差约 1e-7（fp32 舍入），dense / MoE Prelude、ACT 跑满 / 提前停机都覆盖 |
| 训练模式下的 loss 与全部参数梯度 | 相对误差 < 1e-5 |
| 权重往返（原版 → Megatron → 原版）、TP 切分 / 合并往返 | 逐位相等 |
| 同一初始权重、同一数据顺序训练，对比原版 `pretrain.py` 的 loss 轨迹 | TP=1、TP=2（序列并行）、DP=2、激活重计算、中断续训（TP=1 / TP=2）：每步 loss 与原版相同（4 位小数） |
| ACT 延后启用 + 提前停机，TP=2 | TP=1 与 TP=2 完全一致；与原版相差约 1e-4，原因见下方"已知差异" |
| TP=2 训练后导出的权重 | 与 TP=1 导出的一致（fp16 存盘误差约 3e-5），可严格加载进原版模型，`eval.py` 正常运行 |

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
  - NCCL 通信。
- **CPU 冒烟模式**（`--cpu_smoke 1`）：megatron-core 有约 50 处写死了 CUDA 设备、RNG 和梯度裁剪的断言。冒烟模式把这些调用重定向到 CPU，只用于测试和 CI。GPU 上不要开启，检测到 GPU 时会直接报错。
- **不支持流水线并行**：循环块需要在同一个 stage 内迭代 T 圈。
- **不支持 MLA**（`--attn_type mla`）：只实现了 GQA。需要 MLA 请用原版训练。
- **没有开放专家并行（EP）**：Megatron 支持，但断点保存和权重导出还没有处理专家分片。
- 张量并行度必须能整除 `num_key_value_heads`。
- SFT / DPO 仍然使用原版 Trainer，可以直接接 Megatron 版导出的 `pretrain.safetensors`。
