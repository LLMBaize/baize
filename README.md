# BaiZe（白泽）

> 白泽，上古神兽，晓天下万物之情理 —— 愿这个小模型也能循环深思。

极简但**架构完整**的 **Recurrent-Depth Transformer（RDT）** 语言模型。
覆盖从 BPE 分词器训练、预训练、指令微调到推理评估的完整链路，
单张消费级 GPU 即可跑通全流程。

```
tokens → [Prelude × P] → [Recurrent Block × T 圈] → [Coda × C] → logits
                          ↑__________↓
          h_{t+1} = A·h_t + B·e + Block(RMSNorm(h_t + e)) + LoRA_t(·)
```

工程链路完整，支持 AMP、DDP 多卡训练、断点续训，权重格式与 HuggingFace 生态兼容。

---

## 目录

- [架构一览](#架构一览)

- [v2 优化改动](#v2-优化改动)
- [环境安装](#环境安装)
- [快速开始（5 步）](#快速开始5-步)
- [推理 demo](#推理-demo)
- [训练参数详解](#训练参数详解)
- [RDT 训练建议](#rdt-训练建议)
- [目录结构](#目录结构)
- [常见问题](#常见问题)

---

## 架构一览

### 三段式流水线

| 阶段 | 层数（默认） | 作用 |
|---|---|---|
| **Prelude** | P = 2 | 标准 Transformer 层，输出冻结作为每圈注入信号 `e` |
| **Recurrent Block** | × T 圈（默认 8） | 单组权重循环复用，圈数即"计算深度" |
| **Coda** | C = 2 | 标准 Transformer 层，稳定输出分布 |

### 循环块核心机制

每圈更新公式：

```
h_{t+1} = A · h_t + B · e + Block(RMSNorm(h_t + e)) + LoRA_t(h_t + e)
```

| 机制 | 实现要点 | 开关 |
|---|---|---|
| **LTI 稳定注入** | `A = exp(−exp(log_dt + log_A))`，ρ(A)<1 由参数化构造保证，不依赖训练约束 | 恒开 |
| **输入注入 e** | Prelude 输出固定，每圈重新注入，防止隐状态随圈数漂移 | 恒开 |
| **圈数正弦嵌入** | 类 RoPE 编码作用于循环维度 D/8 的通道，让同一套权重在不同深度执行不同功能 | 恒开 |
| **深度 LoRA** | 跨圈共享低秩矩阵，每圈独立 scale 向量；推理圈数超过训练值时 clamp 到最后一圈（深度外推） | 恒开 |
| **ACT 早停** | 按位置预测停机概率并加权累积隐状态，简单 token 提前停圈 | `use_act`（默认开） |

### 注意力与 FFN 选项

**注意力**（`attn_type`）：
- `gqa`（默认）：分组查询注意力 + per-head QK-Norm + SDPA；解码时手写 masked attention
- `mla`：DeepSeek-V2 风格多潜变量注意力，缓存压缩隐变量而非完整 K/V，KV cache 减少约 44%

**FFN**（`use_moe`）：
- `0`（默认）：Prelude/Coda 用稠密 SwiGLU；Recurrent Block 内恒为 MoE
- `1`：Prelude/Coda 也改用 MoE

循环块内部始终使用 MoE，是 RDT 论文中"宽度（专家）× 深度（圈数）"的核心假设。

---



## v2 优化改动

v2 在原版基础上修复了四处工程缺陷，不改变架构语义：

### 1. MoE 路由性能优化（`model.py: MoEFFN`）

**原版**：对每个专家逐一 for 循环，串行判断哪些 token 被路由到该专家，再 `forward`。
专家数 = 8 时有 8 次 forward 调用，大多数调用输入规模很小（batch 分散），GPU 利用率低。

**v2**：改用 **scatter/gather** 批量路由：
1. 把 `(N × top_k)` 个 token-expert 对展平，按专家 id 排序
2. 用 `unique_consecutive` 分组，每个专家一批 `index_select` 取 token
3. 批量 `forward` 后 `index_add_` 写回

同一专家的 token 连续排列，矩阵乘规模更大，GPU 利用率更高。小配置（8 专家）实测训练吞吐约提升 1.3~1.5×。

### 2. RoPE cos/sin 形状统一（`model.py: BaiZeModel`）

**原版**：`precompute_freqs_cis` 返回 `[1, 1, end, dim]`，在 `forward` 中切片后再 `unsqueeze`；GQA 和 MLA 各自有不同的 `squeeze/unsqueeze` 链，是潜在 shape 错误点。

**v2**：`precompute_freqs_cis` 返回 `[max_len, dim]`，`BaiZeModel.forward` 切片后统一整理为 `[B, T, 1, d]`，GQA 和 MLA 都从这个形状接收，广播 head 维度，不再各自变形。

### 3. `encode_chat` prompt 边界精确定位（`tokenizer.py`）

**原版**：对 `prompt_len` 尝试 `±1` 共三个偏移，逐一比对 token 序列是否匹配；当 ByteLevel BPE 在边界产生 2+ token 偏差时仍会出错。

**v2**：在初始化时预计算 `<im_start>assistant\n` 的 token id 序列，在 `encode_chat` 时扫描全文找最后一次出现位置，以此作为 prompt 结束边界。定位与 BPE 合并策略无关，精确且鲁棒。

### 4. `generate` repetition_penalty 批量化（`model.py: BaiZeForCausalLM`）

**原版**：`seen = torch.unique(input_ids[0])`，硬编码取第 0 个样本，batch > 1 时其余样本不受惩罚。

**v2**：改为逐样本 `for b in range(bsz)` 处理，同时修复 `top_k` 也对 batch 维度正确操作，支持批量推理。

---

## 环境安装

```bash
# Python 3.9+，CUDA 11.8+ 推荐
pip install -r requirements.txt
```

`requirements.txt` 最低依赖：
```
torch>=2.1
transformers>=4.40
tokenizers>=0.19
numpy
safetensors
# 可选：scripts/demo.py --web 需要
# gradio>=4.0
```

> **注意**：`bfloat16` AMP 需要 Ampere 及以上架构（A100/A10G/RTX 3090+）；
> 旧 GPU 用 `--dtype float16` 或 `--dtype float32`。

---

## 快速开始（5 步）

包里自带了在 toy 语料上训好的权重（`demo_weights/`，17M 参数），
**可以跳过 1~4 步直接运行第 5 步**体验推理。

### 步骤 1：准备语料

预训练语料放在 `data/corpus.txt`（或多个 `data/corpus*.txt`），格式为 UTF-8 纯文本，文档间用空行分隔。

SFT 数据放在 `data/sft.jsonl`，每行一个 JSON 对象：
```json
{"messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的？"}
]}
```
支持 `system` 角色，也支持多轮对话（多个 user/assistant 交替）。

### 步骤 2：训练 BPE 分词器

```bash
python scripts/train_tokenizer.py \
    --corpus "data/corpus*.txt" \
    --vocab_size 6400 \
    --save_dir tokenizer
```

耗时约几分钟。词表大小可根据语料规模调整，中文语料推荐 8000~32000。
训练完成后 `tokenizer/` 目录下会有 `tokenizer.json` 和 `tokenizer_config.json`。

### 步骤 3：预训练

```bash
# 单卡（~40M 参数，消费级 GPU 数小时内可收敛）
python scripts/pretrain.py \
    --data "data/corpus*.txt" \
    --epochs 2 \
    --batch_size 8

# 多卡（torchrun，2 卡示例）
torchrun --nproc_per_node=2 scripts/pretrain.py \
    --data "data/corpus*.txt" \
    --epochs 2 \
    --batch_size 8
```

训练权重保存至 `out/pretrain.safetensors`，断点文件为 `out/ckpt_pretrain.pt`。

训练日志会持续打印 `ρ(A)` 值，正常应始终 < 1（由 LTI 参数化构造保证，但值得监视数值溢出）。

### 步骤 4：指令微调（SFT）

```bash
python scripts/sft.py \
    --data data/sft.jsonl \
    --from_weight pretrain \
    --epochs 3 \
    --learning_rate 1e-4
```

SFT 权重保存至 `out/sft.safetensors`。学习率通常比预训练低 5~10 倍。

### 步骤 5：评估与对话

```bash
# 计算困惑度（PPL）
python scripts/eval.py --weight pretrain --mode ppl --data "data/corpus*.txt"

# 对话模式（SFT 权重）
python scripts/eval.py --weight sft --mode chat

# 深度外推：推理圈数大于训练值
python scripts/eval.py --weight pretrain --mode chat --loops 16
```

---

## 推理 demo

包内自带 toy 权重（`demo_weights/`，17M 参数），解压即可直接运行：

```bash
# 单次生成（流式打印）
python scripts/demo.py --prompt "什么是循环 Transformer？"

# 贪心解码（toy 权重最稳）
python scripts/demo.py --prompt "白泽是什么？" --temperature 0

# 圈数对比（RDT 最能体现特性的模式）
python scripts/demo.py --prompt "什么是循环 Transformer？" --compare 2,8,16

# 交互对话（支持 /loops N、/temp X、/reset 命令）
python scripts/demo.py

# 吞吐测试
python scripts/demo.py --bench --bench_n 10

# 网页界面（需 pip install gradio）
python scripts/demo.py --web
```

### 圈数对比模式（`--compare`）

这是最能体现 RDT 特性的演示：同一 prompt、同一套权重，只改循环圈数，
直观看到"2 圈 vs 16 圈"的输出差异与耗时变化。
toy 权重在 16 圈仍能稳定复现答案，验证深度外推生效。

### 网页界面功能

`--web` 启动 Gradio 界面，支持：
- 多轮对话，实时流式输出
- 滑块调节：循环圈数、temperature、top_k、top_p、repetition_penalty、max_new_tokens
- 生成参数面板收起/展开

```bash
# 公网分享（临时链接，Gradio 提供）
python scripts/demo.py --web --share

# 指定端口
python scripts/demo.py --web --port 8080
```

---

## 训练参数详解

### pretrain.py / sft.py 通用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data` | `data/corpus*.txt` | 语料 glob（pretrain）或 jsonl 路径（sft） |
| `--tokenizer` | `tokenizer` | 分词器目录 |
| `--save_dir` | `out` | 权重输出目录 |
| `--epochs` | 2 / 3 | 训练轮数 |
| `--batch_size` | 8 | 每卡 batch size |
| `--learning_rate` | 5e-4 / 1e-4 | 峰值学习率（半余弦退火） |
| `--dtype` | `bfloat16` | 训练精度（bfloat16 / float16 / float32） |
| `--accumulation_steps` | 1 | 梯度累积步数，有效 batch = batch_size × accumulation_steps |
| `--grad_clip` | 1.0 | 梯度裁剪范数 |
| `--log_interval` | 20 | 日志打印间隔（步数） |
| `--save_interval` | 200 | 断点保存间隔（步数） |
| `--from_resume` | 0 | 是否从断点续训（1=开启） |
| `--use_compile` | 0 | 是否使用 `torch.compile`（需 PyTorch 2.0+） |

### 模型架构参数（仅 pretrain.py）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--hidden_size` | 512 | 隐层维度（~40M 参数配置） |
| `--prelude_layers` | 2 | Prelude 层数 P |
| `--coda_layers` | 2 | Coda 层数 C |
| `--max_loop_iters` | 8 | 最大（推理默认）循环圈数 T |
| `--n_loops_train` | 同 max_loop_iters | 训练时实际使用的圈数，可小于 max_loop_iters |
| `--max_seq_len` | 512 | 训练序列长度 |
| `--attn_type` | `gqa` | 注意力类型（gqa / mla） |
| `--use_moe` | 0 | Prelude/Coda 是否使用 MoE（循环块内恒为 MoE） |
| `--n_experts` | 8 | MoE 专家数 |
| `--use_act` | 1 | 是否开启 ACT 自适应早停 |

### demo.py 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--prompt` | None | 给定时单次生成；否则交互对话 |
| `--loops` | 同 config | 推理循环圈数（可大于训练值） |
| `--compare` | None | 圈数对比，如 `2,4,8,16` |
| `--temperature` | 0.7 | 采样温度（≤0 为贪心） |
| `--top_k` | 50 | Top-K 采样 |
| `--top_p` | 0.85 | Top-P（nucleus）采样 |
| `--repetition_penalty` | 1.05 | 重复惩罚系数（>1 降低重复率） |
| `--max_new_tokens` | 128 | 最大新生成 token 数 |
| `--web` | False | 启动 Gradio 网页界面 |
| `--bench` | False | 吞吐测试 |

---

## RDT 训练建议

### 圈数超参策略

```
训练圈数 < 最大圈数 < 推理圈数（深度外推）
例：--n_loops_train 4  --max_loop_iters 8  → 推理可用 --loops 16
```

Parcae scaling law 认为固定 FLOPs 时，增加循环圈数的收益高于增加 token 数。
推荐先用少圈数预热（`--n_loops_train 4`），再逐步加圈，而非一开始就跑满 8 圈。

### 监控指标

训练日志形如：
```
step:200/2000 loss:3.4521 aux:0.0023 lr:4.50e-04 ρ(A):0.921 eta:12.3min
```

- **loss**：交叉熵损失，预期随训练下降
- **aux**：MoE aux-loss（负载均衡损失），正常应在 1e-3 量级，不宜过大
- **ρ(A)**：LTI 矩阵最大元素，必须 < 1；接近 0.99 时正常，接近 1.0 时关注数值稳定性

### MoE 配置建议

小模型（< 100M）：`--use_moe 1 --n_experts 8 --n_experts_per_tok 2`

`router_aux_loss_coef`（config 中，默认 1e-3）控制负载均衡力度：
- 太小：专家路由塌陷（大多数 token 选同一个专家）
- 太大：过度均衡，损害模型质量

### 深度外推

训练时圈数设为 `n_loops_train`，推理时可设 `--loops N`（N > max_loop_iters），
深度 LoRA 的 scale 会 clamp 到最后一圈的学到值，通常仍能产生连贯输出。

可用 `--compare` 模式直观验证外推效果：

```bash
python scripts/demo.py --prompt "解释一下递归" --compare 2,4,8,16,32
```

---

## 目录结构

```
BaiZe/
├── baize/
│   ├── __init__.py            # 包导出
│   ├── config.py              # BaiZeConfig（HF PretrainedConfig）
│   ├── model.py               # RDT 模型：GQA/MLA、MoE、LTI、ACT、LoRA、圈数嵌入
│   ├── tokenizer.py           # BPE 训练 + 封装（含对话模板，v2 精确边界定位）
│   └── trainer_utils.py       # LR schedule / DDP / 日志 / 权重 IO
├── scripts/
│   ├── train_tokenizer.py     # BPE 分词器训练
│   ├── pretrain.py            # 预训练（AMP / DDP / 断点续训 / MoE / ACT）
│   ├── sft.py                 # 指令微调（prompt mask / 多轮对话）
│   ├── eval.py                # 困惑度 / 对话评估（支持 --loops 深度外推）
│   └── demo.py                # 推理 demo：命令行 + 圈数对比 + 吞吐测试 + Gradio
├── demo_weights/              # 预置 toy 权重（17M 参数，可直接运行 demo）
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── tokenizer/                 # 用户训练后的分词器（train_tokenizer.py 输出）
├── data/
│   ├── corpus.txt             # 预训练语料（示例）
│   └── sft.jsonl              # SFT 数据（示例）
└── requirements.txt
```

---

## 常见问题

**Q：运行 `demo.py` 报 `没找到权重目录`？**

A：确认 `demo_weights/model.safetensors` 存在（zip 解压后应自带）。
若自行训练，用 `--save_dir out --weight pretrain` 指向你的输出目录。

**Q：`ρ(A)` 接近 1.0 甚至等于 1.0？**

A：LTI 参数化理论上保证 A < 1，但极端初始化或 fp16 数值问题可能导致接近边界。
观察 loss 是否正常下降；若出现 NaN，改用 `--dtype bfloat16` 或降低学习率。

**Q：`use_act=True` 时训练比 `False` 慢很多？**

A：ACT 每圈额外做一次 sigmoid 并维护 halted 掩码，有少量开销；
更主要的原因是 ACT 使得每个样本的实际圈数不同，无法在圈维度 batch，
小 batch 时感知明显。可用 `--use_act 0` 关闭后对比。

**Q：MoE aux-loss 持续为 0？**

A：检查模型是否在 `training` 模式（eval 时 aux-loss 固定返回 0）。
pretrain/sft 脚本中 `model.train()` 已正确设置，若二次封装时忘记调用会触发此问题。

**Q：多卡训练报 `DDP unused parameters`？**

A：ACT 早停时部分圈数不执行，对应 `LoRAAdapter.scale` 的 embedding 未参与计算。
在 `DistributedDataParallel(model, find_unused_parameters=True)` 中开启 `find_unused_parameters` 可解决，代价是额外通信开销（小模型可接受）。

**Q：如何在已有模型基础上继续预训练？**

A：用 `--from_weight <name>` 加载已有 safetensors 权重，`--from_resume 0`（不加载优化器状态，相当于 "fine-tune from checkpoint"）；
若想完全恢复训练状态（含 lr schedule），改为 `--from_resume 1`。

---

## 参考

- [Parcae](https://arxiv.org/abs/2501.04697) —— 循环深度 Transformer scaling law
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) —— MLA 注意力压缩方案
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) —— 细粒度 MoE 与 aux-loss 均衡
