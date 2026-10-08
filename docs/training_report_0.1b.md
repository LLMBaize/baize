# BaiZe 0.13B Field Report

**English** | [简体中文](training_report_0.1b.zh-CN.md)

A single A100 80GB ran the full pipeline: AllenAI data → tokenizer → 2B-token pretraining → SFT → benchmarks.
This document records the configuration used, the training process, benchmark results, and the conclusions
and improvement suggestions drawn from them.

---

## 1. Overview

| Item | Value |
|---|---|
| Model | RDT, 128.70M parameters (115.72M active per token) |
| Vocabulary | ~16k (BPE, inferred from parameter count) |
| Architecture | hidden 1024, Prelude 4 layers + 1 recurrent block (MoE) + Coda 4 layers, GQA 16/4 heads |
| Loops | `max_loop_iters 8`, trained with `n_loops_train 4`, ACT on |
| Pretraining data | mC4 Chinese 70% + C4 English 30%, **2.0B tokens**, seq_len 512 |
| SFT data | Tulu 3 + WildChat Chinese, 84,880 valid samples, 2 epochs |
| Hardware | 1 × A100 80GB (pretraining used ~25GB VRAM, 100% GPU utilization) |
| Time | Pretraining ~8 hours (7,629 steps); SFT ~65 minutes (10,610 steps) |
| Validation PPL | **16.5** (NLL 2.81, 8.86M tokens) |
| HellaSwag / ARC-Easy | **35.2% / 31.4%** (acc_norm, random 25%) |

Conclusion: the full pipeline works, training is stable, and model quality matches expectations for
0.13B / 2B tokens. The capability bottleneck is data volume (especially English / knowledge-heavy data)
and model scale — not the code or the training process.

---

## 2. Reproduction Commands

### 2.1 Data

```bash
# Pretraining corpus: Chinese 70% + English 30%, ~5B chars, plus 5,000 validation docs
python scripts/prepare_allenai.py --task pretrain --sources c4-zh:0.7,c4-en:0.3 \
    --max_chars 5_000_000_000 --val_docs 5000 --min_chars 200 \
    --out data/allenai_pretrain.jsonl --hf_endpoint https://hf-mirror.com

# Tokenizer + pre-tokenize to memmap
python scripts/train_tokenizer.py --corpus data/allenai_pretrain.jsonl --vocab_size 16000 \
    --max_docs 500000 --save_dir tokenizer
python scripts/tokenize_corpus.py --data data/allenai_pretrain.jsonl --tokenizer tokenizer \
    --out data/pretrain.bin --max_tokens 2_000_000_000

# SFT corpus: Tulu 3 + WildChat Chinese, 50/50
python scripts/prepare_allenai.py --task sft --sources tulu3:0.5,wildchat:0.5 --language Chinese \
    --max_docs 100000 --val_docs 500 --out data/allenai_sft.jsonl --hf_endpoint https://hf-mirror.com
```

### 2.2 Pretraining

```bash
python scripts/pretrain.py --data data/pretrain.bin --tokenizer tokenizer --save_dir out \
    --hidden_size 1024 --num_attention_heads 16 --num_key_value_heads 4 --head_dim 64 \
    --intermediate_size 2816 --prelude_layers 4 --coda_layers 4 \
    --max_loop_iters 8 --n_loops_train 4 --moe_intermediate_size 704 \
    --max_seq_len 512 --batch_size 32 --accumulation_steps 16 \
    --learning_rate 6e-4 --epochs 1 --dtype bfloat16 \
    --save_interval 500 --log_interval 20
```

Each step: 512 samples × 512 tokens ≈ 260k tokens; 7,629 steps total, 76 warmup steps, ACT activated
at step 762 (10%). The run was interrupted once and resumed losslessly by adding `--from_resume 1`
to the original command.

### 2.3 SFT

```bash
python scripts/sft.py --data data/allenai_sft.jsonl --tokenizer tokenizer --save_dir out \
    --from_weight pretrain --max_seq_len 1024 --batch_size 16 --learning_rate 1e-4 --epochs 2
```

> This SFT run predates the fix: the loop count defaulted to `max_loop_iters` (8 loops), and ACT was
> off for the first 10% of steps. Now `sft.py` / `dpo.py` inherit the pretraining loop count
> (`config.n_loops_train`) by default and enable ACT from step 0 — no need to manually pass
> `--n_loops_train 4 --act_start_step 0` anymore.

### 2.4 Evaluation

```bash
# Validation perplexity
python scripts/eval.py --weight pretrain --mode ppl --data data/allenai_pretrain.val.jsonl --loops 4

# Pretrained model: plain continuation (chat mode is not usable, see §5.1)
python scripts/eval.py --weight pretrain --mode generate --loops 4 --prompt "The capital of China is||Artificial intelligence is a"

# SFT model: chat / loop-count comparison
python scripts/demo.py --save_dir out --weight sft --tokenizer tokenizer --loops 4 --temperature 0.7 --repetition_penalty 1.2
python scripts/demo.py --save_dir out --weight sft --tokenizer tokenizer --prompt "Introduce AI in three sentences" --compare 2,4,8 --temperature 0

# Standard benchmarks
for L in 1 2 4 8; do
  python scripts/eval.py --weight sft --mode bench --bench arc-easy,hellaswag,ceval,mmlu \
      --bench_limit 500 --chat 1 --loops $L --hf_endpoint https://hf-mirror.com --bench_out out/bench_sft_L$L.json
done
```

---

## 3. Training Process

### 3.1 Pretraining

| Steps | loss | loops | ρ(A) | lr | Notes |
|---|---|---|---|---|---|
| 20 | 8.34 | 4.00 | 0.368 | 1.6e-4 | Warming up |
| 762 | — | 4.00 | — | — | ACT activated; aux jumped from 0.0011 to ~0.006 (ponder cost added) |
| 1,000 | 3.47 | 4.00 | 0.293 | 5.8e-4 | |
| 1,100 | 3.36 | 3.99 | 0.286 | 5.8e-4 | ACT started stopping a few tokens early |
| 4,020 | 2.96–3.02 | 2.82 | 0.223 | 3.1e-4 | After checkpoint resume |
| 6,500–7,020 | 2.68–2.83 (mean ~2.76) | 2.48–2.66 | 0.207 | 8e-5 → 7e-5 | loops stabilized at ~2.6 |
| End | — | — | 0.205 | — | |

- Loss decreased smoothly throughout, never diverged; gnorm stayed at 0.2–0.33.
- ρ(A) decreased monotonically from 0.37 to 0.21, far below 1 — recurrent injection is very stable.
- loops decreased **slowly** after ACT activated (~3,000 steps from 4 to 2.8), with no rapid collapse, finally stabilizing at ~2.6.

### 3.2 SFT

| Steps | loss | gnorm | Notes |
|---|---|---|---|
| 20 | 2.10 | 1.70 | Just exposed to the chat template |
| 120 | 1.26 | 1.11 | Warmup finished (106 steps) |
| After 200 | 1.1–1.4 (occasionally 1.9 per batch) | ~1.0 | Normal batch-to-batch jitter |

- SFT computes loss only on reply tokens, and the format is very regular, so loss is far below pretraining's 2.75.
- 15,120 samples were skipped: prompts over 1,024 tokens had their replies truncated away entirely.

---

## 4. Benchmark Results

### 4.1 Perplexity

Validation set (5,000 docs, 8.86M tokens): **NLL 2.81, PPL 16.5**, consistent with the final training
loss of ~2.75 — no overfitting.

### 4.2 Standard benchmarks (SFT weights, `--chat 1`, 500 questions each, error ~±2%)

| Loops | ARC-Easy | HellaSwag | C-Eval | MMLU |
|---|---|---|---|---|
| 1 | 32.0 | 33.6 | 22.0 | 22.0 |
| 2 | 31.2 | 34.6 | 22.6 | 23.4 |
| 4 | 31.4 | **35.2** | 22.6 | 23.4 |
| 8 | 31.4 | 35.2 | 22.6 | 23.4 |
| Random | 25.0 | 25.0 | 25.0 | 25.0 |
| GPT-2 small (124M, ~10B English tokens) reference | ~40 | ~31 | — | ~25 |

ARC / HellaSwag use acc_norm; C-Eval / MMLU use acc.

- **HellaSwag 35.2%**: clearly above random, slightly better than GPT-2 small. C4 web text suits this task well.
- **ARC-Easy 31.4%**: above random but below GPT-2 small. English was only ~0.6B tokens — not enough scientific knowledge.
- **MMLU / C-Eval at random level**: normal at this scale; models generally need 1B+ parameters to beat random.

### 4.3 Generation quality

- The pretrained model produces mostly fluent continuations, but facts are often wrong.
- After SFT it learned the dialogue format — answer structure is normal (e.g. definition first, then elaboration).
  But knowledge errors are obvious, e.g. "artificial intelligence is a core technology used in smartphones."
- Greedy decoding (`--temperature 0`) easily falls into phrase loops; `--temperature 0.7 --repetition_penalty 1.2` helps noticeably.

---

## 5. Findings and Conclusions

### 5.1 Pretrained models can't be tested in chat mode

Chat mode wraps input in the `<im_start>user ... <im_end>` template. The pretraining corpus contains none of
these markers, so the model outputs garbage directly and falls into repetition. Test pretrained weights with
`eval.py --mode generate` for plain continuation; use chat only after SFT.

### 5.2 Recurrent depth was barely used

Results at 4 and 8 loops are **bit-identical**, 2 loops nearly identical, and even 1 loop is only ~1 point off
(within error). Two reasons:

1. ACT's ponder cost taught the model to stop as early as possible (~2.6 loops on average at end of training).
   By loop 4, all tokens have halted, so further loops no longer affect the output.
2. These benchmarks test knowledge, not multi-step reasoning; and the reasoning tasks a 0.13B model can do are
   limited anyway. Extra loops can't substitute for knowledge.

Practical implication: use `--loops 2` at inference — same quality, faster. During generation, every loop must
run fully to keep the KV cache complete, so 8 loops run at ~64% the speed of 4 loops.

### 5.3 Issues fixed during this run

| Issue | Impact | Fix |
|---|---|---|
| `eval.py --mode ppl` ignored `--loops` | Always computed at `max_loop_iters` regardless | Fixed |
| PPL overlapping windows double-scored | PPL slightly inflated | Each token scored exactly once |
| PPL computed per-window in fp32 with no progress output | Slow, looked hung | Batched + bf16 + progress output |
| No plain-continuation mode | Pretrained models could only be tested with the chat template | Added `--mode generate` |
| SFT/DPO defaulted to 8 loops and redid the 10% ACT warmup | Doubled compute; output distribution jumps at the 10% step | Defaults to `config.n_loops_train`, ACT from step 0 |

---

## 6. Suggestions for the Next Round

### 6.1 For higher scores (in order of payoff)

1. **Add English and knowledge-heavy data**: add `olmo-mix-wiki`, `olmo-mix-pes2o`, or high-quality Dolmino data
   to pretraining — ARC benefits directly.
2. **Add tokens**: from 2B to 5–10B (the Chinchilla optimum for 0.13B is ~2.6B; training longer still helps).
3. **Add parameters**: MMLU / C-Eval only start beating random around 0.5–1B.

DPO mainly improves answer style; it barely moves multiple-choice scores.

### 6.2 To make recurrent depth actually matter

| Change | Parameter |
|---|---|
| Weaken the ponder penalty | `--act_ponder_coef 1e-4` (default 1e-3) |
| Delay ACT activation | `--act_start_step` at 30% of total steps |
| Add reasoning data | `dolmino-math`, `olmo-mix-open-web-math` |
| Control group: ACT off, fixed loops | `--use_act 0 --n_loops_train 4` |

### 6.3 Recommended next config (~0.4B, single A100)

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

This config is ~0.36B (vocab 6400) to ~0.40B (vocab 32000) parameters — trust `Model Params` in the startup log.
Smoke-test with `--max_steps 200` first to confirm memory and speed before the long run.
