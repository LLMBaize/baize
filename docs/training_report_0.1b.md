# BaiZe 0.13B 实测报告

单张 A100 80GB 跑完整条链路：AllenAI 数据 → 分词器 → 预训练 2B token → SFT → 评测。
本文记录所用的配置、训练过程、评测结果，以及从中得到的结论和下一轮的改进建议。

---

## 1. 总览

| 项目 | 值 |
|---|---|
| 模型 | RDT，128.70M 参数（每 token 激活 115.72M） |
| 词表 | 约 1.6 万（BPE，由参数量反推） |
| 结构 | hidden 1024，Prelude 4 层 + 循环块 1 层（MoE）+ Coda 4 层，GQA 16/4 头 |
| 循环 | `max_loop_iters 8`，训练 `n_loops_train 4`，ACT 开启 |
| 预训练数据 | mC4 中文 70% + C4 英文 30%，**2.0B token**，seq_len 512 |
| SFT 数据 | Tulu 3 + WildChat 中文，有效 84,880 条，2 epoch |
| 硬件 | 1 × A100 80GB（预训练显存占用约 25GB，GPU 利用率 100%） |
| 耗时 | 预训练约 8 小时（7,629 步），SFT 约 65 分钟（10,610 步） |
| 验证集 PPL | **16.5**（NLL 2.81，8.86M token） |
| HellaSwag / ARC-Easy | **35.2% / 31.4%**（acc_norm，随机 25%） |

结论：整条链路是通的，训练稳定，模型水平符合 0.13B / 2B token 的预期。
能力的瓶颈在数据量（尤其英文 / 知识类数据）和模型规模，不在代码或训练过程。

---

## 2. 复现命令

### 2.1 数据

```bash
# 预训练语料：中文 70% + 英文 30%，约 50 亿字符，另留 5000 篇验证集
python scripts/prepare_allenai.py --task pretrain --sources c4-zh:0.7,c4-en:0.3 \
    --max_chars 5_000_000_000 --val_docs 5000 --min_chars 200 \
    --out data/allenai_pretrain.jsonl --hf_endpoint https://hf-mirror.com

# 分词器 + 预分词成 memmap
python scripts/train_tokenizer.py --corpus data/allenai_pretrain.jsonl --vocab_size 16000 \
    --max_docs 500000 --save_dir tokenizer
python scripts/tokenize_corpus.py --data data/allenai_pretrain.jsonl --tokenizer tokenizer \
    --out data/pretrain.bin --max_tokens 2_000_000_000

# SFT 语料：Tulu 3 + WildChat 中文，各 50%
python scripts/prepare_allenai.py --task sft --sources tulu3:0.5,wildchat:0.5 --language Chinese \
    --max_docs 100000 --val_docs 500 --out data/allenai_sft.jsonl --hf_endpoint https://hf-mirror.com
```

### 2.2 预训练

```bash
python scripts/pretrain.py --data data/pretrain.bin --tokenizer tokenizer --save_dir out \
    --hidden_size 1024 --num_attention_heads 16 --num_key_value_heads 4 --head_dim 64 \
    --intermediate_size 2816 --prelude_layers 4 --coda_layers 4 \
    --max_loop_iters 8 --n_loops_train 4 --moe_intermediate_size 704 \
    --max_seq_len 512 --batch_size 32 --accumulation_steps 16 \
    --learning_rate 6e-4 --epochs 1 --dtype bfloat16 \
    --save_interval 500 --log_interval 20
```

每步 512 个样本 × 512 token ≈ 26 万 token，共 7,629 步，warmup 76 步，ACT 从第 762 步（10%）启用。
中途断过一次，用原命令加 `--from_resume 1` 从断点无损续上。

### 2.3 SFT

```bash
python scripts/sft.py --data data/allenai_sft.jsonl --tokenizer tokenizer --save_dir out \
    --from_weight pretrain --max_seq_len 1024 --batch_size 16 --learning_rate 1e-4 --epochs 2
```

> 本次 SFT 是在修复前跑的：圈数默认用了 `max_loop_iters`（8 圈），且前 10% 步关闭了 ACT。
> 现在 `sft.py` / `dpo.py` 默认沿用预训练圈数（`config.n_loops_train`）并从第 0 步启用 ACT，
> 不再需要手动加 `--n_loops_train 4 --act_start_step 0`。

### 2.4 评测

```bash
# 验证集困惑度
python scripts/eval.py --weight pretrain --mode ppl --data data/allenai_pretrain.val.jsonl --loops 4

# 预训练模型：纯续写（不能用 chat 模式，见 §5.1）
python scripts/eval.py --weight pretrain --mode generate --loops 4 --prompt "中国的首都是||人工智能是一种"

# SFT 模型：对话 / 圈数对比
python scripts/demo.py --save_dir out --weight sft --tokenizer tokenizer --loops 4 --temperature 0.7 --repetition_penalty 1.2
python scripts/demo.py --save_dir out --weight sft --tokenizer tokenizer --prompt "用三句话介绍人工智能" --compare 2,4,8 --temperature 0

# 标准评测
for L in 1 2 4 8; do
  python scripts/eval.py --weight sft --mode bench --bench arc-easy,hellaswag,ceval,mmlu \
      --bench_limit 500 --chat 1 --loops $L --hf_endpoint https://hf-mirror.com --bench_out out/bench_sft_L$L.json
done
```

---

## 3. 训练过程

### 3.1 预训练

| 步数 | loss | loops | ρ(A) | lr | 说明 |
|---|---|---|---|---|---|
| 20 | 8.34 | 4.00 | 0.368 | 1.6e-4 | warmup 中 |
| 762 | — | 4.00 | — | — | ACT 启用，aux 从 0.0011 跳到约 0.006（加入 ponder cost） |
| 1,000 | 3.47 | 4.00 | 0.293 | 5.8e-4 | |
| 1,100 | 3.36 | 3.99 | 0.286 | 5.8e-4 | ACT 开始让少数 token 提前停 |
| 4,020 | 2.96–3.02 | 2.82 | 0.223 | 3.1e-4 | 断点续训后 |
| 6,500–7,020 | 2.68–2.83（均值约 2.76） | 2.48–2.66 | 0.207 | 8e-5 → 7e-5 | loops 稳定在约 2.6 |
| 结束 | — | — | 0.205 | — | |

- loss 全程平稳下降，没有发散；gnorm 稳定在 0.2–0.33。
- ρ(A) 从 0.37 单调降到 0.21，远小于 1，循环注入很稳定。
- loops 在 ACT 启用后**缓慢**下降（约 3,000 步才从 4 降到 2.8），没有出现快速塌缩，最后稳定在约 2.6。

### 3.2 SFT

| 步数 | loss | gnorm | 说明 |
|---|---|---|---|
| 20 | 2.10 | 1.70 | 刚接触对话模板 |
| 120 | 1.26 | 1.11 | warmup 结束（106 步） |
| 200 以后 | 1.1–1.4（单批偶尔到 1.9） | 约 1.0 | 正常的批次波动 |

- SFT 只对回复部分计 loss，格式又很规整，所以 loss 远低于预训练的 2.75。
- 15,120 条样本被跳过：prompt 超过 1024 token，回复被整段截掉。

---

## 4. 评测结果

### 4.1 困惑度

验证集（5,000 篇，8.86M token）：**NLL 2.81，PPL 16.5**，与训练末期 loss 约 2.75 吻合，没有过拟合。

### 4.2 标准评测（SFT 权重，`--chat 1`，每项 500 题，误差约 ±2%）

| 圈数 | ARC-Easy | HellaSwag | C-Eval | MMLU |
|---|---|---|---|---|
| 1 | 32.0 | 33.6 | 22.0 | 22.0 |
| 2 | 31.2 | 34.6 | 22.6 | 23.4 |
| 4 | 31.4 | **35.2** | 22.6 | 23.4 |
| 8 | 31.4 | 35.2 | 22.6 | 23.4 |
| 随机水平 | 25.0 | 25.0 | 25.0 | 25.0 |
| GPT-2 small（124M，约 10B 英文 token）参考 | 约 40 | 约 31 | — | 约 25 |

ARC / HellaSwag 为 acc_norm，C-Eval / MMLU 为 acc。

- **HellaSwag 35.2%**：明显高于随机，比 GPT-2 small 还略好。C4 网页文本和这个任务很对口。
- **ARC-Easy 31.4%**：高于随机，但不如 GPT-2 small。英文只占约 0.6B token，科学知识不够。
- **MMLU / C-Eval 在随机水平**：这个规模的正常现象，一般要 1B 以上才会脱离随机。

### 4.3 生成效果

- 预训练模型做纯续写时语句基本通顺，事实经常不准。
- SFT 后学会了对话格式，回答结构正常，比如先下定义再展开。但知识错误明显，例如"人工智能是用于智能手机的核心技术"。
- 贪心解码（`--temperature 0`）下容易陷入短语循环；用 `--temperature 0.7 --repetition_penalty 1.2` 可以明显缓解。

---

## 5. 发现与结论

### 5.1 预训练模型不能用 chat 模式测

chat 模式会套 `<im_start>user ... <im_end>` 模板。预训练语料里没有这些标记，模型会直接输出乱码，并陷入"卸载卸载卸载"这样的复读。
预训练权重要用 `eval.py --mode generate` 做纯续写，SFT 之后再用 chat。

### 5.2 循环深度基本没有被利用

评测在 4 圈和 8 圈下**逐位相同**，2 圈几乎相同，1 圈也只差 1 个点左右（在误差范围内）。原因有两点：

1. ACT 的 ponder cost 让模型学会了尽早停机（训练末期平均约 2.6 圈）。到第 4 圈时，所有 token 都已经停机，后面的圈数不再影响输出。
2. 这些评测主要考知识，不考多步推理；0.13B 模型能做的推理任务也很有限。多跑几圈换不来知识。

实际用途：推理直接用 `--loops 2`，效果不变，速度更快。生成时因为要保证 KV cache 完整，每一圈都得跑满，所以 8 圈的速度只有 4 圈的约 64%。

### 5.3 本次实测中修复的问题

| 问题 | 影响 | 修复 |
|---|---|---|
| `eval.py --mode ppl` 忽略 `--loops` | 不管填多少，都按 `max_loop_iters` 算 | 已修复 |
| ppl 重叠窗口重复计分 | PPL 略偏高 | 每个 token 只计一次 |
| ppl 逐窗口 fp32 计算、没有进度输出 | 慢，看起来像卡住 | 改成批处理 + bf16 + 进度输出 |
| 没有纯续写模式 | 预训练模型只能套对话模板测 | 新增 `--mode generate` |
| SFT/DPO 默认用 8 圈，并重新做 10% ACT 预热 | 计算量翻倍，第 10% 步处输出分布跳变 | 默认沿用 `config.n_loops_train`，ACT 从第 0 步启用 |

---

## 6. 下一轮建议

### 6.1 想让分数更高（按收益从大到小）

1. **加英文和知识类数据**：在预训练里加入 `olmo-mix-wiki`、`olmo-mix-pes2o` 或 dolmino 的高质量数据，ARC 会直接受益。
2. **加 token**：从 2B 提到 5–10B（0.13B 模型的 Chinchilla 最优点约 2.6B，多训仍有收益）。
3. **加参数**：到 0.5–1B 量级时，MMLU / C-Eval 才会开始高于随机。

DPO 主要改善回答风格，对选择题分数帮助很小。

### 6.2 想让循环深度真正发挥作用

| 改法 | 参数 |
|---|---|
| 减弱 ponder 惩罚 | `--act_ponder_coef 1e-4`（默认 1e-3） |
| 推迟启用 ACT | `--act_start_step` 设为总步数的 30% |
| 加入推理类数据 | `dolmino-math`、`olmo-mix-open-web-math` |
| 对照组：关掉 ACT、固定圈数 | `--use_act 0 --n_loops_train 4` |

### 6.3 推荐的下一轮配置（约 0.4B，单卡 A100）

```bash
python scripts/prepare_allenai.py --task pretrain \
    --sources c4-zh:0.45,c4-en:0.25,olmo-mix-wiki:0.1,olmo-mix-pes2o:0.1,olmo-mix-open-web-math:0.1 \
    --max_chars 25_000_000_000 --val_docs 5000 --min_chars 200 --out data/allenai_pretrain.jsonl

python scripts/pretrain.py --data data/pretrain.bin --tokenizer tokenizer --save_dir out_0.4b \
    --hidden_size 1536 --num_attention_heads 12 --num_key_value_heads 4 --head_dim 128 \
    --intermediate_size 4096 --prelude_layers 6 --coda_layers 6 \
    --max_loop_iters 8 --n_loops_train 4 --moe_intermediate_size 1024 \
    --max_seq_len 1024 --batch_size 16 --accumulation_steps 16 \
    --learning_rate 4e-4 --epochs 1 --grad_checkpoint 1 \
    --act_ponder_coef 1e-4 --save_interval 500 --snapshot_interval 2000
```

该配置约 0.36B（词表 6400）到 0.40B（词表 32000）参数，以启动日志里的 `Model Params` 为准。建议先用 `--max_steps 200` 冒烟测试，确认显存和速度后再长跑。
