# BaiZe（白泽）

> 白泽，上古神兽，晓天下万物之情理 —— 愿这个小模型也能循环深思。

极简但**架构完整**的 **Recurrent-Depth Transformer（RDT）** 语言模型。
覆盖从 BPE 分词器训练、预训练、指令微调（SFT）、偏好对齐（DPO）到推理与标准评测的完整链路，
单张消费级 GPU 即可跑通全流程。

```
tokens → [Prelude × P] → [Recurrent Block × T 圈] → [Coda × C] → logits
                          ↑__________↓
```

$$h_{t+1} = A\,h_t + B\,e + \mathrm{Block}\big(\mathrm{RMSNorm}(h_t + e)\big) + \mathrm{LoRA}_t\big(\mathrm{RMSNorm}(h_t + e)\big)$$

工程链路完整：AMP、DDP / FSDP2 多卡、激活重计算、预分词 memmap 数据、可精确续训的断点，
可直接接入 AllenAI 预训练 / SFT / 偏好数据，权重格式与 HuggingFace 生态兼容。

---

## 目录

- [架构一览](#架构一览)
- [v3 改动](#v3-改动)
- [v2 优化改动](#v2-优化改动)
- [环境安装](#环境安装)
- [快速开始（5 步）](#快速开始5-步)
- [推理 demo](#推理-demo)
- [使用 AllenAI 数据](#使用-allenai-数据)
- [实测报告：0.13B](#实测报告013b)
- [参数使用指南](#参数使用指南)
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

$$h_{t+1} = A\,h_t + B\,e + \mathrm{Block}\big(\mathrm{RMSNorm}(h_t + e)\big) + \mathrm{LoRA}_t\big(\mathrm{RMSNorm}(h_t + e)\big)$$

| 机制 | 实现要点 | 开关 |
|---|---|---|
| **LTI 稳定注入** | $A = \exp\big(-\exp(\log d_t + \log A)\big)$， $\rho(A) < 1$ 由参数化构造保证，不依赖训练约束 | 恒开 |
| **输入注入 e** | Prelude 输出固定，每圈重新注入，防止隐状态随圈数漂移 | 恒开 |
| **圈数正弦嵌入** | 类 RoPE 编码作用于循环维度 D/8 的通道，让同一套权重在不同深度执行不同功能 | 恒开 |
| **深度 LoRA** | 跨圈共享低秩矩阵，每圈独立 scale 向量；推理圈数超过训练值时 clamp 到最后一圈（深度外推） | 恒开 |
| **ACT 早停** | 按位置预测停机概率并加权累积隐状态，简单 token 提前停圈；停机偏置初始化为 −3（初始跑满所有圈），ponder cost 鼓励学会早停 | `use_act`（默认开）、`act_init_bias`、`act_ponder_coef` |

### 注意力与 FFN 选项

**注意力**（`attn_type`）：
- `gqa`（默认）：分组查询注意力 + per-head QK-Norm + SDPA；解码时手写 masked attention
- `mla`：DeepSeek-V2 风格多潜变量注意力，缓存压缩隐变量而非完整 K/V，KV cache 减少约 44%

**FFN**（`use_moe`）：
- `0`（默认）：Prelude/Coda 用稠密 SwiGLU；Recurrent Block 内恒为 MoE
- `1`：Prelude/Coda 也改用 MoE

循环块内部始终使用 MoE，是 RDT 论文中"宽度（专家）× 深度（圈数）"的核心假设。
专家权重合并存储为 `[E, …]` 张量，路由后一次 `bmm` 算完所有专家；
`moe_capacity_factor > 0` 时按容量丢弃溢出 token（形状固定、无 GPU→CPU 同步）。

---

## v3 改动

### 严重 bug 修复

| 问题 | 后果 | 修复 |
|---|---|---|
| 预训练 labels 手动错位后模型内部又 shift 一次 | 预训练目标变成预测 t+2 | Dataset 返回对齐的 input/labels |
| 带 KV cache 的 prefill 走无因果 mask 的手写 attention | 推理与训练分布不一致 | GQA/MLA 统一走带 mask 的 SDPA，兼容 cache 偏移 |
| MoE 用 `enumerate` 下标当专家 id | 专家 0 未被选中时 token 被送错专家 | 批量 bmm 实现，按真实专家 id 路由 |
| 循环块 aux-loss 只计最后一圈，Prelude/Coda MoE 未计 | 负载均衡失效 | 各圈平均 + 所有 MoE 层求和 |
| ACT 跑满仍未停机时权重和 < 1 | 输出幅度偏小 | 最后一圈补齐剩余概率 |
| ACT 停机偏置被 HF 初始化为 0（p≈0.5）；且一开始就启用 ACT 时后几圈得不到训练 | **名义 8 圈实际只跑 2~3 圈** | 偏置默认 −3 + ponder cost + 默认前 10% 步关闭 ACT 跑满所有圈 |
| `save_weights` 先转 fp16 再去重 | tie 的 embedding 存两份 | 先去重再转换 |

### 训练工程

- **预分词 memmap 数据**：`scripts/tokenize_corpus.py` → `.bin`，训练时零拷贝读取，不占内存、启动零等待
- **统一训练循环** `baize/trainer.py`（pretrain / sft / dpo 共用）：
  - 线性 warmup + 余弦退火（`--warmup_steps`，默认总步数 1%）
  - 可续训采样器：单卡也打乱；续训从中断处下一个 batch 精确继续（测试验证：中断续训与不中断结果一致）
  - 续训在 `torch.compile` / DDP / FSDP 包装前加载，修复 `--use_compile` 下无法续训
  - **FSDP2**（`--fsdp 1`）、**激活重计算**（`--grad_checkpoint 1`）、ACT 延后启用（`--act_start_step`）
  - 梯度累积时跳过中间 micro-batch 的梯度同步；norm/bias 不做 weight decay；断点原子写入
- `load_weights` 默认严格校验：形状不匹配 / 缺参数直接报错并列出参数名；自动兼容旧版逐专家 MoE 权重

### 新功能

- **DPO 偏好对齐**：`scripts/dpo.py`；`prepare_allenai.py --task dpo` 直接拉取 Tulu 3 / OLMo 2 / UltraFeedback 偏好数据
- **标准评测**：`eval.py --mode bench`，支持 ARC、MMLU、C-Eval、HellaSwag、GSM8K 与本地 jsonl
- `generate` 全批量向量化（repetition penalty / top-k / top-p），已结束样本固定填充 eos
- 测试 `tests/`（39 项，含 DDP/FSDP 双进程、中断续训一致性）+ GitHub Actions CI；`demo_weights` 已用修复后的代码重训

---

## v2 优化改动

> 注：v2 的 MoE 路由与 `generate` 实现已在 v3 中被批量化版本替换，以下为历史说明。

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

`requirements.txt`：
```
torch>=2.6            # --fsdp 需要 FSDP2
transformers>=4.40
tokenizers>=0.19
safetensors
numpy
datasets>=2.19        # 可选：AllenAI 数据下载、标准评测
# gradio>=4.0         # 可选：demo.py --web
```

运行测试：`for t in tests/test_*.py; do python $t; done`（CPU 即可，含 DDP/FSDP 双进程测试）。

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

> 想用真实规模的数据？见 [使用 AllenAI 数据](#使用-allenai-数据)：一条命令即可从 C4/mC4、
> OLMo 预训练混合、Tulu 3 SFT 等数据集按比例、按数据量抽取语料。

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
# 小语料：直接读文本
python scripts/pretrain.py --data "data/corpus*.txt" --epochs 2 --batch_size 8

# 大语料：先预分词成 memmap（只需一次），再训练
python scripts/tokenize_corpus.py --data "data/corpus*.txt" --out data/pretrain.bin
python scripts/pretrain.py --data data/pretrain.bin --epochs 1 --batch_size 32 --accumulation_steps 8

# 多卡：DDP；模型大时加 FSDP 与激活重计算
torchrun --nproc_per_node=2 scripts/pretrain.py --data data/pretrain.bin
torchrun --nproc_per_node=8 scripts/pretrain.py --data data/pretrain.bin --fsdp 1 --grad_checkpoint 1

# 中断后续训（从中断处的下一个 batch 继续）
python scripts/pretrain.py --data data/pretrain.bin --from_resume 1
```

训练权重保存至 `out/pretrain.safetensors`，断点文件为 `out/ckpt_pretrain.pt`。

- `ckpt_pretrain.pt`：断点（fp32 权重 + 优化器状态 + 步数），每 `--save_interval` 步**覆盖**一次，只用于续训；
- `pretrain.safetensors` + `config.json`：训练结束时生成的 fp16 权重，用于推理 / SFT / 评测；
- 想保留各阶段的权重：训练时加 `--snapshot_interval 1000`，或对正在跑的训练另开一个进程
  `python scripts/export_ckpt.py --ckpt out/ckpt_pretrain.pt --watch`，断点每更新一次就导出到 `out/snapshots/step_XXXXXX/`。

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

从预训练权重继续训练时，SFT / DPO 默认沿用预训练的训练圈数（`config.json` 里的 `n_loops_train`），
并且从第 0 步就启用 ACT（停机头已在预训练中学好）。需要时可以用 `--n_loops_train` / `--act_start_step` 覆盖。

### 步骤 4.5（可选）：偏好对齐（DPO）

```bash
# 拉取 AllenAI Tulu 3 偏好数据（需 pip install datasets）
python scripts/prepare_allenai.py --task dpo --sources tulu3-pref --max_docs 20000 \
    --out data/allenai_dpo.jsonl
python scripts/dpo.py --data data/allenai_dpo.jsonl --from_weight sft --beta 0.1
```

数据为 `{"chosen": [对话], "rejected": [对话]}`，两条对话只有最后一条 assistant 回复不同。
日志中 `acc`（chosen 奖励高于 rejected 的比例）应逐渐上升，`margin` 逐渐增大。
权重保存至 `out/dpo.safetensors`。

### 步骤 5：评估与对话

```bash
# 计算困惑度（PPL），--loops 指定推理圈数
python scripts/eval.py --weight pretrain --mode ppl --data data/allenai_pretrain.val.jsonl --loops 4

# 预训练权重：纯续写（预训练模型没见过对话模板，不要用 chat 模式测）
python scripts/eval.py --weight pretrain --mode generate --loops 4 --prompt "中国的首都是||人工智能是一种"

# 对话模式（SFT 权重）
python scripts/eval.py --weight sft --mode chat

# 深度外推：推理圈数大于训练值
python scripts/eval.py --weight pretrain --mode chat --loops 16

# 标准评测（需联网 + datasets）：选择题按对数似然打分，GSM8K 贪心生成
python scripts/eval.py --weight pretrain --mode bench --bench arc-easy,ceval,mmlu,hellaswag --bench_limit 500
python scripts/eval.py --weight sft --mode bench --bench gsm8k --chat 1 --bench_out results.json
# 本地题库：每行 {"question": ..., "choices": [...], "answer": 0 或 "A"}
python scripts/eval.py --weight sft --mode bench --bench jsonl:data/my_bench.jsonl
```

评测结果同时给出 `acc`、`acc_norm`（按选项长度归一化）与随机基线 `random_baseline`。
小模型在 MMLU / C-Eval 上接近随机基线是正常的；0.1B 量级主要看 ARC-Easy 和 HellaSwag。
实测数据见 [实测报告：0.13B](#实测报告013b)。

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

> `demo_weights` 用 v3 代码重训：预训练前一半步数关闭 ACT、跑满 8 圈，之后启用 ACT。
> toy 语料极易记忆，启用 ACT 后模型学会在 1~2 圈就停（日志 `loops` 从 8 降到约 1），
> 这是 ACT 对简单输入的正常行为；真实语料上难 token 会保留更多圈数。

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

## 使用 AllenAI 数据

`scripts/prepare_allenai.py` 从 HuggingFace **流式**读取 AllenAI 的预训练 / 后训练数据集，
按权重混合、过滤，读够配额即停（不会整库下载），输出 jsonl 供各训练脚本直接使用。
需要额外安装 `pip install datasets`。

```bash
python scripts/prepare_allenai.py --list          # 查看内置数据源

# 预训练：mC4 中文 70% + C4 英文 30%，共 15 亿字符，另留 2000 篇验证集
python scripts/prepare_allenai.py --task pretrain --sources c4-zh:0.7,c4-en:0.3 \
    --max_chars 1_500_000_000 --val_docs 2000 --out data/allenai_pretrain.jsonl

# 后训练：Tulu 3 SFT mixture 抽 5 万条；偏好对齐：Tulu 3 偏好数据 2 万对
python scripts/prepare_allenai.py --task sft --sources tulu3 --max_docs 50000 \
    --out data/allenai_sft.jsonl
python scripts/prepare_allenai.py --task dpo --sources tulu3-pref --max_docs 20000 \
    --out data/allenai_dpo.jsonl

# 预分词成 memmap，训练时还可再截断：
python scripts/tokenize_corpus.py --data data/allenai_pretrain.jsonl --out data/pretrain.bin
python scripts/pretrain.py --data data/pretrain.bin --max_tokens 1_000_000_000 --max_steps 20000
python scripts/sft.py --data data/allenai_sft.jsonl --max_samples 30000
python scripts/dpo.py --data data/allenai_dpo.jsonl --max_samples 10000
```

| 环节 | 数据量控制参数 |
|---|---|
| 下载 `prepare_allenai.py` | `--max_docs` / `--max_tokens` / `--max_chars`（按 `--sources` 权重分配）、`--max_scan`、`--val_docs`、`--skip` |
| 分词器 `train_tokenizer.py` | `--max_docs` |
| 预分词 `tokenize_corpus.py` | `--max_docs` / `--max_tokens` |
| 预训练 `pretrain.py` | `--max_docs` / `--max_tokens` / `--max_steps` |
| SFT `sft.py`、DPO `dpo.py` | `--max_samples` / `--max_steps` |
| 评估 `eval.py` | `--max_docs`（ppl）/ `--bench_limit`（bench） |

下载哪些数据集、推荐配比、数据量估算、过滤参数、许可等完整说明见
**[docs/allenai_data.md](docs/allenai_data.md)**。

---

## 实测报告：0.13B

单张 A100 80GB 跑完整条链路（AllenAI 数据 → 分词器 → 预训练 2B token → SFT → 评测）。
完整的配置、训练曲线、评测和分析见 **[docs/training_report_0.1b.md](docs/training_report_0.1b.md)**。

| 项目 | 值 |
|---|---|
| 模型 | 128.70M 参数（激活 115.72M）：hidden 1024，Prelude 4 + 循环块（MoE）+ Coda 4，训练 4 圈 |
| 数据 | 预训练：mC4 中文 70% + C4 英文 30%，2.0B token；SFT：Tulu 3 + WildChat 中文，8.5 万条 |
| 耗时 | 预训练约 8 小时（7,629 步），SFT 约 65 分钟 |
| 预训练末期 loss / 验证集 PPL | 约 2.76 / 16.5 |
| ACT 平均圈数 | 训练末期约 2.6（从第 762 步启用后缓慢下降，没有塌缩） |

标准评测（SFT 权重，每项 500 题，误差约 ±2%）：

| 圈数 | ARC-Easy | HellaSwag | C-Eval | MMLU |
|---|---|---|---|---|
| 1 | 32.0 | 33.6 | 22.0 | 22.0 |
| 2 | 31.2 | 34.6 | 22.6 | 23.4 |
| 4 | 31.4 | **35.2** | 22.6 | 23.4 |
| 8 | 31.4 | 35.2 | 22.6 | 23.4 |
| 随机 | 25.0 | 25.0 | 25.0 | 25.0 |

主要结论：

- 链路通、训练稳定，水平符合 0.13B / 2B token 的预期。HellaSwag 略好于 GPT-2 small（约 31%）；ARC-Easy 偏低，是因为英文数据只有约 0.6B token。
- **循环深度基本没有被利用**：ACT 学会了 2–3 圈就停，4 圈和 8 圈结果完全一样。推理用 `--loops 2` 即可，更快且效果不变。
  想让深度发挥作用，可以调小 `--act_ponder_coef`（如 1e-4）、推迟 `--act_start_step`，或加入数学 / 推理数据。
- 预训练权重要用 `--mode generate` 测续写；chat 模式会套对话模板，输出乱码。
- 提升分数最有效的方法依次是：加英文和知识类数据 → 加 token → 加参数。

---

## 参数使用指南

### 按规模推荐的配置

| 规模 | 架构参数 | 训练参数 | 数据量 | 单卡 A100 耗时 |
|---|---|---|---|---|
| 冒烟测试（约 10M） | 默认值 | `--batch_size 8 --max_steps 200` | 任意 | 几分钟 |
| **0.13B（已实测）** | `--hidden_size 1024 --num_attention_heads 16 --num_key_value_heads 4 --head_dim 64 --intermediate_size 2816 --prelude_layers 4 --coda_layers 4 --moe_intermediate_size 704` | `--max_seq_len 512 --batch_size 32 --accumulation_steps 16 --learning_rate 6e-4` | 2–5B token | 2B token 约 8 小时 |
| 约 0.4B | `--hidden_size 1536 --num_attention_heads 12 --num_key_value_heads 4 --head_dim 128 --intermediate_size 4096 --prelude_layers 6 --coda_layers 6 --moe_intermediate_size 1024` | `--max_seq_len 1024 --batch_size 16 --accumulation_steps 16 --learning_rate 4e-4 --grad_checkpoint 1` | 8–20B token | 10B token 约 5 天 |
| 约 0.9B | `--hidden_size 2048 --num_attention_heads 16 --num_key_value_heads 4 --head_dim 128 --intermediate_size 5632 --prelude_layers 8 --coda_layers 8 --moe_intermediate_size 2048` | `--max_seq_len 1024 --batch_size 16 --accumulation_steps 8 --learning_rate 3e-4 --grad_checkpoint 1`，建议多卡 `--fsdp 1` | 20B+ token | 20B token 单卡约 3 周，8 卡约 3 天 |

所有规模都建议使用 `--max_loop_iters 8 --n_loops_train 4`。实际参数量以启动日志中的 `Model Params` 为准，词表越大参数越多。
长跑之前先加 `--max_steps 200` 冒烟，确认显存、速度和 loss 正常。

### 关键参数怎么选

| 想要 | 怎么设 |
|---|---|
| 每步 token 数 | `batch_size × accumulation_steps × 卡数 × max_seq_len`，预训练建议 25 万 ~ 100 万 |
| 控制数据量 | 准备时用 `prepare_allenai.py --max_chars / --max_tokens`；训练时用 `--max_tokens` / `--max_steps` 截断 |
| 学习率 | 预训练：0.1B 用 6e-4，0.4B 用 4e-4，1B 用 3e-4；SFT 用 1e-4（约为预训练的 1/5）；DPO 用 1e-6 |
| 显存不够 | 先开 `--grad_checkpoint 1`，再减小 `--batch_size` 并同比增大 `--accumulation_steps`（每步 token 数不变） |
| 中途断了 | 原命令加 `--from_resume 1`；`--data`、`--batch_size`、`--accumulation_steps`、`--seed` 和模型参数不能改 |
| 保留中间权重 | `--snapshot_interval 2000`，或另开进程运行 `scripts/export_ckpt.py --watch` |
| 在网页上看曲线 | `--use_wandb 1`（见下方「用 wandb 记录训练」） |
| ACT 早停过猛（loops 很快跌到 2 以下） | 调小 `--act_ponder_coef`（如 1e-4），或增大 `--act_start_step` |
| 想用满循环深度 | `--use_act 0`，固定跑 `n_loops_train` 圈 |
| 推理更快 | `--loops 2`（ACT 模型大多 2–3 圈就停，实测效果不变） |
| 减少复读 | 生成时加 `--temperature 0.7 --repetition_penalty 1.2`；不要用贪心解码长回答 |

### 用 wandb 记录训练

```bash
pip install wandb && wandb login          # 一次性；服务器无外网时跳过 login，用 --wandb_mode offline

python scripts/pretrain.py ... --use_wandb 1 --wandb_project baize --wandb_run_name pretrain-0.13b
python scripts/sft.py      ... --use_wandb 1 --wandb_project baize --wandb_run_name sft-0.13b

# 离线模式：先在本地记录，有网时再上传
python scripts/pretrain.py ... --use_wandb 1 --wandb_mode offline
wandb sync out/wandb/offline-run-*
```

- 只在主进程（rank 0）记录，多卡不会重复；每 `--log_interval` 步写一次。
- 记录内容：`train/loss`、`train/aux`（DPO 还有 `train/acc`、`train/margin`）、`train/lr`、`train/grad_norm`、
  `train/rho_A`、`train/samples`、`train/steps_per_sec`、`act/avg_loops`、`act/enabled`；
  config 里保存全部训练参数和模型结构。
- 断点里会保存 wandb run id，`--from_resume 1` 续训时**接着写同一个 run**，曲线不会断开。
- 没装 wandb 或登录失败时只打印警告，训练照常进行。

### 训练日志怎么看

| 指标 | 健康状态 | 需要处理 |
|---|---|---|
| loss | 平稳下降；单步 ±0.1 的波动正常 | 连续几百步不降反升 → 降低学习率后续训 |
| gnorm | 预训练 0.2–1，SFT 刚开始 1–2 | 持续飙升 → 学习率过大 |
| loops | ACT 启用后缓慢下降，稳定在 2–4 | 几百步内跌到 2 以下 → 调小 `--act_ponder_coef` |
| aux | 1e-3 ~ 1e-2 | 突然大幅变化，结合 loops 一起看 |
| ρ(A) | 小于 1，训练中缓慢下降 | 接近 1 → 关注数值稳定性 |

---

## 训练参数详解

### 数据参数

| 参数 | 脚本 | 默认 | 说明 |
|---|---|---|---|
| `--data` | 全部 | — | 语料 glob，逗号分隔多个。pretrain：`.bin`（推荐）或 `.txt`/`.jsonl`；sft/dpo：jsonl |
| `--max_tokens` | pretrain | 不限 | 最多使用多少 token |
| `--max_docs` | pretrain | 不限 | [文本语料] 最多读入多少篇 |
| `--max_samples` | sft / dpo | 不限 | 最多使用多少条对话 / 偏好对 |
| `--max_seq_len` | 全部 | 512 | 序列长度（超长对话截断） |
| `--tokenizer` | 全部 | `tokenizer` | 分词器目录 |
| `--from_weight` | 全部 | none / pretrain / sft | 初始化权重名（在 `save_dir` 下） |
| `--beta` | dpo | 0.1 | DPO 温度 |
| `--n_loops_train` | sft / dpo | 沿用预训练 | 训练圈数；默认读取 `config.json` 里的 `n_loops_train`，没有则用 `max_loop_iters` |

### 训练参数（pretrain / sft / dpo 共用，见 `baize/trainer.py`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--save_dir` | `out` | 权重 / 断点输出目录 |
| `--epochs` | 2 / 3 / 1 | 训练轮数 |
| `--max_steps` | 不限 | 最多训练多少个优化步（与 epochs 取较小者） |
| `--batch_size` | 8 | 每卡 micro-batch |
| `--accumulation_steps` | 1 | 梯度累积；每步样本数 = batch_size × accumulation_steps × 卡数 |
| `--learning_rate` | 5e-4 / 1e-4 / 1e-6 | 峰值学习率 |
| `--warmup_steps` | 总步数 1% | 线性 warmup 步数，之后余弦退火到 0.1 × 峰值 |
| `--weight_decay` | 0.1 | 只作用于 ≥2 维权重矩阵 |
| `--dtype` | `bfloat16` | bfloat16 / float16 / float32（FSDP 不支持 float16） |
| `--grad_clip` | 1.0 | 梯度裁剪范数 |
| `--log_interval` / `--save_interval` | 20 / 200 | 日志 / 断点间隔（步数） |
| `--snapshot_interval` | 0 | 每 N 步额外保存一份 fp16 权重快照到 `<save_dir>/snapshots/step_XXXXXX/`（不覆盖，可直接推理 / 评测） |
| `--from_resume` | 0 | 从 `ckpt_<save_weight>.pt` 续训（数据位置、优化器、步数全部恢复） |
| `--fsdp` | 0 | torchrun 多卡时用 FSDP2 切分参数 / 梯度 / 优化器状态 |
| `--grad_checkpoint` | 0 | 激活重计算：循环块每圈、Prelude/Coda 每层只存输入，反向重算 |
| `--act_start_step` | 预训练：总步数 10%；SFT/DPO（加载预训练权重时）：0 | 前 N 步关闭 ACT 跑满所有圈，之后启用早停；0 = 一开始就启用 |
| `--use_compile` | 0 | `torch.compile` |
| `--seed` | 42 | 随机种子（也决定数据打乱顺序） |
| `--use_wandb` | 0 | 把训练指标记录到 Weights & Biases（需 `pip install wandb` 并 `wandb login`） |
| `--wandb_project` / `--wandb_entity` | `baize` / 登录账号 | wandb 项目名 / 团队名 |
| `--wandb_run_name` | `<阶段>-<时间>` | run 名称，如 `pretrain-1004-0628` |
| `--wandb_mode` | `online` | `offline` 只写本地 `<save_dir>/wandb/`，之后 `wandb sync` 上传（服务器没外网时用） |

### 模型架构参数（仅 pretrain.py）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--hidden_size` | 512 | 隐层维度 |
| `--num_attention_heads` / `--num_key_value_heads` / `--head_dim` | 8 / 2 / 64 | 注意力头数 / KV 头数（GQA）/ 每头维度 |
| `--intermediate_size` | 1024 | 稠密 FFN 中间维度 |
| `--prelude_layers` | 2 | Prelude 层数 P |
| `--coda_layers` | 2 | Coda 层数 C |
| `--max_loop_iters` | 8 | 最大（推理默认）循环圈数 T |
| `--n_loops_train` | 同 max_loop_iters | 训练时实际使用的圈数，可小于 max_loop_iters；会写进 `config.json`，SFT/DPO 默认沿用 |
| `--max_seq_len` | 512 | 训练序列长度 |
| `--attn_type` | `gqa` | 注意力类型（gqa / mla） |
| `--use_moe` | 0 | Prelude/Coda 是否使用 MoE（循环块内恒为 MoE） |
| `--n_experts` / `--n_experts_per_tok` | 8 / 2 | MoE 专家数 / 每 token 激活专家数 |
| `--moe_intermediate_size` | 512 | 每个专家的中间维度 |
| `--moe_capacity_factor` | 0 | >0 时按容量丢弃溢出 token；0 = 不丢 |
| `--use_act` | 1 | 是否开启 ACT 自适应早停 |
| `--act_init_bias` | −3 | 停机预测器初始偏置（−3 → 初始停机概率≈0.05，先跑满所有圈） |
| `--act_ponder_coef` | 1e-3 | ponder cost 系数，越大越倾向早停；0 = 关闭 |

### eval.py 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | `chat` | `ppl` 困惑度 / `generate` 纯续写（预训练权重）/ `chat` 对话（SFT 之后）/ `bench` 标准评测 |
| `--weight` / `--save_dir` | `pretrain` / `out` | 权重名与所在目录（目录下需要有 `config.json`） |
| `--loops` | 同 config | 推理圈数；ppl / generate / chat / bench 都生效 |
| `--data` / `--max_docs` | — | [ppl] 评估语料（txt / jsonl）与最多篇数 |
| `--prompt` | None | [generate] 续写开头，多个用 `\|\|` 分隔；不给则进入交互输入 |
| `--temperature` / `--top_p` / `--top_k` | 0.7 / 0.85 / 50 | [generate] 采样参数，temperature ≤ 0 为贪心 |
| `--repetition_penalty` | 1.2 | [generate] 重复惩罚，小模型建议 1.1–1.3 |
| `--max_new_tokens` | 256 | 最大生成长度 |
| `--bench` | `arc-easy` | [bench] 逗号分隔：`arc-easy,arc-challenge,mmlu,ceval[:学科],hellaswag,gsm8k,jsonl:<路径>` |
| `--bench_limit` | 不限 | [bench] 每个评测集最多多少题；500 题误差约 ±2% |
| `--chat` | 0 | [bench] 用对话模板包裹题目，SFT 模型用 1 |
| `--bench_out` | None | [bench] 结果写入 json |
| `--hf_endpoint` | 环境变量 | HuggingFace 地址，如 `https://hf-mirror.com` |

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

ACT 早停：`--act_start_step`（默认总步数的 10%）之前关闭 ACT、跑满所有圈，让循环块先把后几圈训练出来，
之后再让 ACT 学习何时停。**不要一开始就启用 ACT**：未训练的后几圈只会引入噪声，停机头会在几十步内学会
1~2 圈就停，后几圈因权重≈0 拿不到梯度，循环深度坍缩（实测从 8 圈跌到 1.7 圈）。
循环圈数越多激活显存越大，显存紧张时加 `--grad_checkpoint 1`。

### 监控指标

训练日志形如：
```
step:200/2000 loss:3.4521 aux:0.0083 lr:4.50e-04 gnorm:0.92 loops:7.40 ρ(A):0.921 eta:12.3min
```

- **loss**：交叉熵损失，预期随训练下降
- **aux**：MoE 负载均衡损失 + ACT ponder cost（`act_ponder_coef × 平均圈数`），正常在 1e-3 ~ 1e-2 量级
- **loops**：本步各位置实际跑的平均圈数。ACT 启用前等于训练圈数，之后随 ACT 学会早停缓慢下降
  （0.13B 实测：约 3,000 步从 4 降到 2.8，最后稳定在约 2.6）；若几百步内就跌到 2 以下，说明 ponder cost 太大（调小 `--act_ponder_coef`）
- **gnorm**：裁剪前的梯度范数，持续飙升通常意味着学习率过大
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
│   ├── data.py                # 语料读取（.txt/.jsonl）、memmap .bin 数据集、可续训采样器
│   ├── trainer.py             # 通用训练循环：warmup / 续训 / DDP / FSDP2 / 激活重计算
│   ├── benchmarks.py          # 标准评测：ARC / MMLU / C-Eval / HellaSwag / GSM8K / 本地 jsonl
│   └── trainer_utils.py       # LR schedule / 分布式初始化 / 日志 / 权重 IO（严格校验）
├── scripts/
│   ├── prepare_allenai.py     # AllenAI 预训练 / SFT / 偏好数据流式下载、混合、过滤、配额控制
│   ├── train_tokenizer.py     # BPE 分词器训练
│   ├── tokenize_corpus.py     # 预分词 → memmap .bin（大语料训练用）
│   ├── pretrain.py            # 预训练
│   ├── sft.py                 # 指令微调（prompt mask / 多轮对话）
│   ├── dpo.py                 # 偏好对齐（DPO）
│   ├── eval.py                # 困惑度 / 对话 / 标准评测（支持 --loops 深度外推）
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
├── docs/
│   ├── allenai_data.md        # AllenAI 数据接入说明（下载哪些数据、数据量控制）
│   └── training_report_0.1b.md # 0.13B 实测报告（配置、训练曲线、评测、结论）
├── tests/                     # 测试（python tests/test_*.py；CI 见 .github/workflows/tests.yml）
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

A：v3 中 MoE 改为批量计算，所有专家参数每步都参与计算；只有在 ACT 未启用
（`--use_act 0` 或 `--act_start_step > 0` 的前期）时停机预测器不参与计算，
训练循环会自动为这两种情况打开 `find_unused_parameters`。

**Q：加载权重报"形状不匹配 / 缺失参数"？**

A：v3 的 `load_weights` 默认严格校验。最常见原因是 SFT/评估时用了与预训练不同的分词器（vocab_size 不同）
或 `config.json` 与权重不对应。确认 `--tokenizer` 与 `--save_dir/config.json` 与预训练一致；
确需部分加载时可在代码中调用 `load_weights(model, path, strict=False)`。

**Q：显存不够？**

A：依次尝试：`--grad_checkpoint 1`（激活重计算，循环圈数多时收益最大）→ 减小 `--batch_size` 并增大
`--accumulation_steps` → 多卡 `--fsdp 1`（参数 / 梯度 / 优化器状态按卡数切分）。

**Q：如何在已有模型基础上继续预训练？**

A：用 `--from_weight <name>` 加载已有 safetensors 权重（新的优化器与学习率调度）；
若是训练中断，用 `--from_resume 1`：模型、优化器、步数与数据位置全部恢复，从中断处的下一个 batch 继续。
注意续训时 `--batch_size`、`--accumulation_steps`、卡数、`--seed` 需与原来一致，否则数据位置无法对齐。

---

## 参考

- [Parcae](https://arxiv.org/abs/2501.04697) —— 循环深度 Transformer scaling law
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) —— MLA 注意力压缩方案
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) —— 细粒度 MoE 与 aux-loss 均衡
- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) —— ACT 早停与 ponder cost
- [DPO](https://arxiv.org/abs/2305.18290) —— 直接偏好优化
- [Tulu 3](https://arxiv.org/abs/2411.15124) —— AllenAI 开放后训练数据与配方
