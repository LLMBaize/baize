# BaiZe

**English** | [简体中文](README.zh-CN.md)

> BaiZe, the mythical beast that knows the nature of all things — may this small model also learn to think in loops.

A minimal yet **architecturally complete** **Recurrent-Depth Transformer (RDT)** language model.
It covers the full pipeline — BPE tokenizer training, pretraining, instruction tuning (SFT),
preference alignment (DPO), inference, and standard benchmarks — all on a single consumer GPU.

```
tokens → [Prelude × P] → [Recurrent Block × T loops] → [Coda × C] → logits
                          ↑__________↓
```

$$h_{t+1} = A\,h_t + B\,e + \mathrm{Block}\big(\mathrm{RMSNorm}(h_t + e)\big) + \mathrm{LoRA}_t\big(\mathrm{RMSNorm}(h_t + e)\big)$$

The engineering chain is complete: AMP, DDP / FSDP2 multi-GPU, activation recomputation, pre-tokenized
memmap data, and checkpoints that resume exactly. It plugs directly into AllenAI pretraining / SFT /
preference data, and its weight format is compatible with the HuggingFace ecosystem.

---

## Contents

- [Architecture Overview](#architecture-overview)
- [What's New in v3](#whats-new-in-v3)
- [v2 Optimizations](#v2-optimizations)
- [Installation](#installation)
- [Quick Start (5 Steps)](#quick-start-5-steps)
- [Inference Demo](#inference-demo)
- [Using AllenAI Data](#using-allenai-data)
- [Field Report: 0.13B](#field-report-013b)
- [Parameter Guide](#parameter-guide)
- [Training Parameters Reference](#training-parameters-reference)
- [RDT Training Tips](#rdt-training-tips)
- [Repository Layout](#repository-layout)
- [FAQ](#faq)
- [License](#license)

---

## Architecture Overview

### Three-stage pipeline

| Stage | Layers (default) | Role |
|---|---|---|
| **Prelude** | P = 2 | Standard Transformer layers; output is frozen as the per-loop injection signal `e` |
| **Recurrent Block** | × T loops (default 8) | A single set of weights reused in a loop; the loop count is the "compute depth" |
| **Coda** | C = 2 | Standard Transformer layers; stabilizes the output distribution |

### Recurrent block core mechanisms

Per-loop update:

$$h_{t+1} = A\,h_t + B\,e + \mathrm{Block}\big(\mathrm{RMSNorm}(h_t + e)\big) + \mathrm{LoRA}_t\big(\mathrm{RMSNorm}(h_t + e)\big)$$

| Mechanism | Implementation | Switch |
|---|---|---|
| **LTI stable injection** | $A = \exp\big(-\exp(\log d_t + \log A)\big)$, $\rho(A) < 1$ guaranteed by parameterization, no training constraint needed | Always on |
| **Input injection e** | The Prelude output is fixed and re-injected every loop, preventing hidden-state drift over loops | Always on |
| **Loop-index sinusoidal embedding** | RoPE-style encoding applied to D/8 channels of the recurrent dimension, letting the same weights perform different functions at different depths | Always on |
| **Depth LoRA** | Low-rank matrices shared across loops with a per-loop scale vector; when inference loops exceed the training value, scales are clamped to the last loop (depth extrapolation) | Always on |
| **ACT early stopping** | Predicts a halting probability per position and accumulates hidden states weighted by it, so easy tokens stop looping early; the halting bias is initialized to −3 (all loops run at first), and the ponder cost encourages learning to stop early | `use_act` (default on), `act_init_bias`, `act_ponder_coef` |

### Attention and FFN options

**Attention** (`attn_type`):
- `gqa` (default): grouped-query attention + per-head QK-Norm + SDPA; hand-written masked attention during decoding
- `mla`: DeepSeek-V2-style multi-head latent attention, caching compressed latents instead of full K/V, reducing KV cache by ~44%

**FFN** (`use_moe`):
- `0` (default): dense SwiGLU in Prelude/Coda; the Recurrent Block always uses MoE
- `1`: Prelude/Coda also switch to MoE

The recurrent block always uses MoE internally — the core RDT assumption of "width (experts) × depth (loops)".
Expert weights are stored merged as `[E, …]` tensors; after routing, a single `bmm` computes all experts.
With `moe_capacity_factor > 0`, overflowing tokens are dropped by capacity (fixed shapes, no GPU→CPU sync).

---

## What's New in v3

### Critical bug fixes

| Issue | Consequence | Fix |
|---|---|---|
| Pretraining labels were shifted manually, then shifted again inside the model | Pretraining target became predicting t+2 | Dataset returns aligned input/labels |
| Prefill with KV cache used hand-written attention without a causal mask | Inference/training distribution mismatch | GQA/MLA both use masked SDPA, compatible with cache offsets |
| MoE used `enumerate` indices as expert ids | Tokens were sent to the wrong expert when expert 0 was not selected | Batched bmm implementation, routing by true expert id |
| Recurrent-block aux-loss counted only the last loop; Prelude/Coda MoE not counted | Load balancing broken | Averaged across loops + summed over all MoE layers |
| When ACT ran all loops without halting, weights summed to < 1 | Output magnitude too small | Last loop tops up the remaining probability |
| ACT halting bias initialized to 0 by HF (p≈0.5); with ACT on from the start, later loops never got trained | **Nominally 8 loops, actually only 2–3 ran** | Bias defaults to −3 + ponder cost + ACT disabled for the first 10% of steps so all loops run |
| `save_weights` converted to fp16 before deduplication | Tied embeddings saved twice | Deduplicate first, then convert |

### Training engineering

- **Pre-tokenized memmap data**: `scripts/tokenize_corpus.py` → `.bin`; zero-copy reads at training time, no memory footprint, zero startup wait
- **Unified training loop** `baize/trainer.py` (shared by pretrain / sft / dpo):
  - Linear warmup + cosine decay (`--warmup_steps`, default 1% of total steps)
  - Resumable sampler: shuffles even on a single GPU; resume continues exactly at the batch after the interruption (verified by tests: interrupted-resumed training matches uninterrupted results)
  - Resume loads before `torch.compile` / DDP / FSDP wrapping, fixing resume under `--use_compile`
  - **FSDP2** (`--fsdp 1`), **activation recomputation** (`--grad_checkpoint 1`), delayed ACT activation (`--act_start_step`)
  - Skips gradient sync for intermediate micro-batches during accumulation; no weight decay on norms/biases; atomic checkpoint writes
- `load_weights` validates strictly by default: shape mismatch / missing parameters raise an error listing parameter names; automatically compatible with legacy per-expert MoE weights

### New features

- **DPO preference alignment**: `scripts/dpo.py`; `prepare_allenai.py --task dpo` pulls Tulu 3 / OLMo 2 / UltraFeedback preference data directly
- **Standard benchmarks**: `eval.py --mode bench`, supporting ARC, MMLU, C-Eval, HellaSwag, GSM8K, and local jsonl
- `generate` fully batched and vectorized (repetition penalty / top-k / top-p); finished samples are padded with eos
- Tests in `tests/` (39 items, including DDP/FSDP two-process and interrupt-resume consistency) + GitHub Actions CI; `demo_weights` retrained with the fixed code

---

## v2 Optimizations

> Note: v2's MoE routing and `generate` implementations were replaced by batched versions in v3; the following is historical documentation.

v2 fixed four engineering defects on top of the original, without changing architectural semantics:

### 1. MoE routing performance (`model.py: MoEFFN`)

**Original**: looped over experts one by one, serially checking which tokens routed to each expert, then `forward`.
With 8 experts that's 8 forward calls, most with tiny inputs (scattered batch), leaving the GPU underutilized.

**v2**: switched to **scatter/gather** batched routing:
1. Flatten the `(N × top_k)` token-expert pairs and sort by expert id
2. Group with `unique_consecutive`; each expert gets one batched `index_select` of its tokens
3. After batched `forward`, write back with `index_add_`

Tokens for the same expert are contiguous, so matrix multiplications are larger and GPU utilization is higher. With a small config (8 experts), measured training throughput improved ~1.3–1.5×.

### 2. Unified RoPE cos/sin shapes (`model.py: BaiZeModel`)

**Original**: `precompute_freqs_cis` returned `[1, 1, end, dim]`, sliced then `unsqueeze`d in `forward`; GQA and MLA each had their own `squeeze/unsqueeze` chains — a potential shape-error source.

**v2**: `precompute_freqs_cis` returns `[max_len, dim]`; `BaiZeModel.forward` slices and normalizes to `[B, T, 1, d]`, which both GQA and MLA receive, broadcasting over heads with no per-branch reshaping.

### 3. Precise prompt-boundary detection in `encode_chat` (`tokenizer.py`)

**Original**: tried three offsets (`±1`) for `prompt_len`, comparing token sequences one by one; still failed when ByteLevel BPE produced a 2+ token deviation at the boundary.

**v2**: precomputes the token id sequence of `<im_start>assistant\n` at init, then scans the full text in `encode_chat` for its last occurrence as the prompt-end boundary. Detection is independent of BPE merge policy — exact and robust.

### 4. Batched `generate` repetition_penalty (`model.py: BaiZeForCausalLM`)

**Original**: `seen = torch.unique(input_ids[0])` — hardcoded sample 0, so with batch > 1 the other samples got no penalty.

**v2**: processes per-sample with `for b in range(bsz)`, and also fixes `top_k` to operate correctly over the batch dimension, enabling batched inference.

---

## Installation

```bash
# Python 3.9+, CUDA 11.8+ recommended
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.6            # --fsdp requires FSDP2
transformers>=4.40
tokenizers>=0.19
safetensors
numpy
datasets>=2.19        # optional: AllenAI data download, standard benchmarks
# gradio>=4.0         # optional: demo.py --web
```

Run tests: `for t in tests/test_*.py; do python $t; done` (CPU is fine; includes DDP/FSDP two-process tests).

> **Note**: `bfloat16` AMP requires Ampere or newer (A100/A10G/RTX 3090+);
> on older GPUs use `--dtype float16` or `--dtype float32`.

---

## Quick Start (5 Steps)

The package ships with weights trained on a toy corpus (`demo_weights/`, 17M parameters),
so **you can skip steps 1–4 and run step 5 directly** to try inference.

### Step 1: Prepare the corpus

Pretraining corpus goes in `data/corpus.txt` (or multiple `data/corpus*.txt`), as UTF-8 plain text with documents separated by blank lines.

SFT data goes in `data/sft.jsonl`, one JSON object per line:
```json
{"messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help you?"}
]}
```
The `system` role is supported, as is multi-turn dialogue (alternating user/assistant).

> Want real-scale data? See [Using AllenAI Data](#using-allenai-data): one command samples corpora
> by ratio and volume from C4/mC4, the OLMo pretraining mix, Tulu 3 SFT, and more.

### Step 2: Train the BPE tokenizer

```bash
python scripts/train_tokenizer.py \
    --corpus "data/corpus*.txt" \
    --vocab_size 6400 \
    --save_dir tokenizer
```

Takes a few minutes. Adjust vocab size to your corpus; for Chinese corpora, 8000–32000 is recommended.
After training, `tokenizer/` will contain `tokenizer.json` and `tokenizer_config.json`.

### Step 3: Pretrain

```bash
# Small corpus: read text directly
python scripts/pretrain.py --data "data/corpus*.txt" --epochs 2 --batch_size 8

# Large corpus: pre-tokenize to memmap first (once), then train
python scripts/tokenize_corpus.py --data "data/corpus*.txt" --out data/pretrain.bin
python scripts/pretrain.py --data data/pretrain.bin --epochs 1 --batch_size 32 --accumulation_steps 8

# Multi-GPU: DDP; add FSDP and activation recomputation for large models
torchrun --nproc_per_node=2 scripts/pretrain.py --data data/pretrain.bin
torchrun --nproc_per_node=8 scripts/pretrain.py --data data/pretrain.bin --fsdp 1 --grad_checkpoint 1

# Resume after interruption (continues at the batch after the interruption)
python scripts/pretrain.py --data data/pretrain.bin --from_resume 1
```

Trained weights are saved to `out/pretrain.safetensors`; the checkpoint file is `out/ckpt_pretrain.pt`.

- `ckpt_pretrain.pt`: checkpoint (fp32 weights + optimizer state + step count), **overwritten** every `--save_interval` steps, used only for resuming;
- `pretrain.safetensors` + `config.json`: fp16 weights generated at the end of training, used for inference / SFT / evaluation;
- To keep weights from intermediate stages: add `--snapshot_interval 1000` during training, or run
  `python scripts/export_ckpt.py --ckpt out/ckpt_pretrain.pt --watch` in a separate process to export each checkpoint update to `out/snapshots/step_XXXXXX/`.

Training logs continuously print the `ρ(A)` value, which should always be < 1 (guaranteed by the LTI parameterization, but worth watching for numerical overflow).

### Step 4: Instruction tuning (SFT)

```bash
python scripts/sft.py \
    --data data/sft.jsonl \
    --from_weight pretrain \
    --epochs 3 \
    --learning_rate 1e-4
```

SFT weights are saved to `out/sft.safetensors`. The learning rate is typically 5–10× lower than pretraining.

When continuing from pretrained weights, SFT / DPO inherit the pretraining loop count (`n_loops_train` in `config.json`)
by default, and enable ACT from step 0 (the halting head was already learned during pretraining). Override with `--n_loops_train` / `--act_start_step` when needed.

### Step 4.5 (optional): Preference alignment (DPO)

```bash
# Pull AllenAI Tulu 3 preference data (requires pip install datasets)
python scripts/prepare_allenai.py --task dpo --sources tulu3-pref --max_docs 20000 \
    --out data/allenai_dpo.jsonl
python scripts/dpo.py --data data/allenai_dpo.jsonl --from_weight sft --beta 0.1
```

Data is `{"chosen": [dialogue], "rejected": [dialogue]}` — the two dialogues differ only in the last assistant reply.
In the logs, `acc` (fraction where the chosen reward exceeds the rejected) should rise gradually, and `margin` should grow.
Weights are saved to `out/dpo.safetensors`.

### Step 5: Evaluate and chat

```bash
# Perplexity (PPL); --loops sets the inference loop count
python scripts/eval.py --weight pretrain --mode ppl --data data/allenai_pretrain.val.jsonl --loops 4

# Pretrained weights: plain continuation (the pretrained model has never seen the chat template — don't test it in chat mode)
python scripts/eval.py --weight pretrain --mode generate --loops 4 --prompt "The capital of China is||Artificial intelligence is a"

# Chat mode (SFT weights)
python scripts/eval.py --weight sft --mode chat

# Depth extrapolation: inference loops greater than the training value
python scripts/eval.py --weight pretrain --mode chat --loops 16

# Standard benchmarks (requires network + datasets): multiple-choice scored by log-likelihood, GSM8K greedy generation
python scripts/eval.py --weight pretrain --mode bench --bench arc-easy,ceval,mmlu,hellaswag --bench_limit 500
python scripts/eval.py --weight sft --mode bench --bench gsm8k --chat 1 --bench_out results.json
# Local question bank: one {"question": ..., "choices": [...], "answer": 0 or "A"} per line
python scripts/eval.py --weight sft --mode bench --bench jsonl:data/my_bench.jsonl
```

Results include `acc`, `acc_norm` (normalized by choice length), and the `random_baseline`.
It's normal for small models to be near the random baseline on MMLU / C-Eval; at the 0.1B scale, focus on ARC-Easy and HellaSwag.
Measured numbers: see [Field Report: 0.13B](#field-report-013b).

---

## Inference Demo

The package ships with toy weights (`demo_weights/`, 17M parameters) — ready to run out of the box:

```bash
# Single generation (streaming output)
python scripts/demo.py --prompt "What is a recurrent Transformer?"

# Greedy decoding (most stable with toy weights)
python scripts/demo.py --prompt "What is BaiZe?" --temperature 0

# Loop-count comparison (the mode that best shows off RDT)
python scripts/demo.py --prompt "What is a recurrent Transformer?" --compare 2,8,16

# Interactive chat (supports /loops N, /temp X, /reset commands)
python scripts/demo.py

# Throughput test
python scripts/demo.py --bench --bench_n 10

# Web UI (requires pip install gradio)
python scripts/demo.py --web
```

### Loop-comparison mode (`--compare`)

This is the demo that best shows off RDT: same prompt, same weights, only the loop count changes —
you directly see the output differences and latency changes of "2 loops vs 16 loops".
The toy weights still reproduce answers stably at 16 loops, confirming depth extrapolation works.

> `demo_weights` was retrained with v3 code: ACT off for the first half of pretraining, running all 8 loops,
> then ACT on. Toy corpora are trivially memorized, so with ACT on the model learns to stop after 1–2 loops
> (the `loops` log drops from 8 to ~1) — normal ACT behavior for easy inputs; on real corpora,
> hard tokens keep more loops.

### Web UI features

`--web` launches a Gradio interface with:
- Multi-turn chat with real-time streaming
- Sliders for: loop count, temperature, top_k, top_p, repetition_penalty, max_new_tokens
- Collapsible generation-parameter panel

```bash
# Public share (temporary link, provided by Gradio)
python scripts/demo.py --web --share

# Custom port
python scripts/demo.py --web --port 8080
```

---

## Using AllenAI Data

`scripts/prepare_allenai.py` **streams** AllenAI pretraining / post-training datasets from HuggingFace,
mixes and filters by weight, and stops once the quota is filled (no full-dataset download), writing jsonl
that the training scripts consume directly. Requires `pip install datasets`.

```bash
python scripts/prepare_allenai.py --list          # list built-in data sources

# Pretraining: mC4 Chinese 70% + C4 English 30%, 1.5B chars total, plus 2,000 validation docs
python scripts/prepare_allenai.py --task pretrain --sources c4-zh:0.7,c4-en:0.3 \
    --max_chars 1_500_000_000 --val_docs 2000 --out data/allenai_pretrain.jsonl

# Post-training: 50k samples from the Tulu 3 SFT mixture; preference alignment: 20k pairs of Tulu 3 preference data
python scripts/prepare_allenai.py --task sft --sources tulu3 --max_docs 50000 \
    --out data/allenai_sft.jsonl
python scripts/prepare_allenai.py --task dpo --sources tulu3-pref --max_docs 20000 \
    --out data/allenai_dpo.jsonl

# Pre-tokenize to memmap; you can truncate further at training time:
python scripts/tokenize_corpus.py --data data/allenai_pretrain.jsonl --out data/pretrain.bin
python scripts/pretrain.py --data data/pretrain.bin --max_tokens 1_000_000_000 --max_steps 20000
python scripts/sft.py --data data/allenai_sft.jsonl --max_samples 30000
python scripts/dpo.py --data data/allenai_dpo.jsonl --max_samples 10000
```

| Stage | Volume-control parameters |
|---|---|
| Download `prepare_allenai.py` | `--max_docs` / `--max_tokens` / `--max_chars` (allocated across sources by `--sources` weights), `--max_scan`, `--val_docs`, `--skip` |
| Tokenizer `train_tokenizer.py` | `--max_docs` |
| Pre-tokenize `tokenize_corpus.py` | `--max_docs` / `--max_tokens` |
| Pretrain `pretrain.py` | `--max_docs` / `--max_tokens` / `--max_steps` |
| SFT `sft.py`, DPO `dpo.py` | `--max_samples` / `--max_steps` |
| Eval `eval.py` | `--max_docs` (ppl) / `--bench_limit` (bench) |

For the full documentation — which datasets to download, recommended ratios, volume estimates,
filter parameters, licensing — see **[docs/allenai_data.md](docs/allenai_data.md)**.

---

## Field Report: 0.13B

A single A100 80GB ran the full pipeline (AllenAI data → tokenizer → 2B-token pretraining → SFT → benchmarks).
Full configuration, training curves, benchmarks, and analysis: **[docs/training_report_0.1b.md](docs/training_report_0.1b.md)**.

| Item | Value |
|---|---|
| Model | 128.70M parameters (115.72M active): hidden 1024, Prelude 4 + recurrent block (MoE) + Coda 4, trained with 4 loops |
| Data | Pretraining: mC4 Chinese 70% + C4 English 30%, 2.0B tokens; SFT: Tulu 3 + WildChat Chinese, 85k samples |
| Time | Pretraining ~8 hours (7,629 steps); SFT ~65 minutes |
| Final pretraining loss / validation PPL | ~2.76 / 16.5 |
| ACT average loops | ~2.6 at end of training (slowly decreased after activation at step 762, no collapse) |

Standard benchmarks (SFT weights, 500 questions each, error ~±2%):

| Loops | ARC-Easy | HellaSwag | C-Eval | MMLU |
|---|---|---|---|---|
| 1 | 32.0 | 33.6 | 22.0 | 22.0 |
| 2 | 31.2 | 34.6 | 22.6 | 23.4 |
| 4 | 31.4 | **35.2** | 22.6 | 23.4 |
| 8 | 31.4 | 35.2 | 22.6 | 23.4 |
| Random | 25.0 | 25.0 | 25.0 | 25.0 |

Key takeaways:

- The pipeline works, training is stable, and quality matches expectations for 0.13B / 2B tokens. HellaSwag is slightly better than GPT-2 small (~31%); ARC-Easy is lower because English data was only ~0.6B tokens.
- **Recurrent depth was barely used**: ACT learned to stop after 2–3 loops; results at 4 and 8 loops are identical. Use `--loops 2` at inference — faster with no quality change.
  To make depth matter, lower `--act_ponder_coef` (e.g. 1e-4), delay `--act_start_step`, or add math / reasoning data.
- Pretrained weights must be tested with `--mode generate` for continuation; chat mode applies the dialogue template and produces garbage.
- The most effective ways to raise scores, in order: add English and knowledge-heavy data → add tokens → add parameters.

---

## Parameter Guide

### Recommended configs by scale

| Scale | Architecture parameters | Training parameters | Data volume | Time on one A100 |
|---|---|---|---|---|
| Smoke test (~10M) | Defaults | `--batch_size 8 --max_steps 200` | Any | A few minutes |
| **0.13B (measured)** | `--hidden_size 1024 --num_attention_heads 16 --num_key_value_heads 4 --head_dim 64 --intermediate_size 2816 --prelude_layers 4 --coda_layers 4 --moe_intermediate_size 704` | `--max_seq_len 512 --batch_size 32 --accumulation_steps 16 --learning_rate 6e-4` | 2–5B tokens | ~8 hours for 2B tokens |
| ~0.4B | `--hidden_size 1536 --num_attention_heads 12 --num_key_value_heads 4 --head_dim 128 --intermediate_size 4096 --prelude_layers 6 --coda_layers 6 --moe_intermediate_size 1024` | `--max_seq_len 1024 --batch_size 16 --accumulation_steps 16 --learning_rate 4e-4 --grad_checkpoint 1` | 8–20B tokens | ~5 days for 10B tokens |
| ~0.9B | `--hidden_size 2048 --num_attention_heads 16 --num_key_value_heads 4 --head_dim 128 --intermediate_size 5632 --prelude_layers 8 --coda_layers 8 --moe_intermediate_size 2048` | `--max_seq_len 1024 --batch_size 16 --accumulation_steps 8 --learning_rate 3e-4 --grad_checkpoint 1`, multi-GPU `--fsdp 1` recommended | 20B+ tokens | ~3 weeks single-GPU for 20B tokens, ~3 days on 8 GPUs |

For all scales, use `--max_loop_iters 8 --n_loops_train 4`. The actual parameter count is shown in `Model Params` in the startup log — larger vocab means more parameters.
Before a long run, smoke-test with `--max_steps 200` to confirm memory, speed, and loss are normal.

### How to choose key parameters

| Goal | Setting |
|---|---|
| Tokens per step | `batch_size × accumulation_steps × GPUs × max_seq_len`; 250k–1M recommended for pretraining |
| Control data volume | At preparation: `prepare_allenai.py --max_chars / --max_tokens`; at training: truncate with `--max_tokens` / `--max_steps` |
| Learning rate | Pretraining: 6e-4 for 0.1B, 4e-4 for 0.4B, 3e-4 for 1B; SFT: 1e-4 (~1/5 of pretraining); DPO: 1e-6 |
| Out of memory | First enable `--grad_checkpoint 1`, then reduce `--batch_size` and increase `--accumulation_steps` proportionally (tokens per step unchanged) |
| Interrupted run | Add `--from_resume 1` to the original command; `--data`, `--batch_size`, `--accumulation_steps`, `--seed`, and model parameters must not change |
| Keep intermediate weights | `--snapshot_interval 2000`, or run `scripts/export_ckpt.py --watch` in a separate process |
| View curves in a web UI | `--use_wandb 1` (see "Logging with wandb" below) |
| ACT stops too aggressively (loops quickly drops below 2) | Lower `--act_ponder_coef` (e.g. 1e-4), or increase `--act_start_step` |
| Use full recurrent depth | `--use_act 0`, fixed `n_loops_train` loops |
| Faster inference | `--loops 2` (ACT models mostly stop at 2–3 loops; measured results unchanged) |
| Less repetition | Generate with `--temperature 0.7 --repetition_penalty 1.2`; avoid greedy decoding for long answers |

### Logging with wandb

```bash
pip install wandb && wandb login          # one-time; on servers without internet, skip login and use --wandb_mode offline

python scripts/pretrain.py ... --use_wandb 1 --wandb_project baize --wandb_run_name pretrain-0.13b
python scripts/sft.py      ... --use_wandb 1 --wandb_project baize --wandb_run_name sft-0.13b

# Offline mode: log locally first, upload when you have network
python scripts/pretrain.py ... --use_wandb 1 --wandb_mode offline
wandb sync out/wandb/offline-run-*
```

- Only the main process (rank 0) logs — no duplicates on multi-GPU; written every `--log_interval` steps.
- Logged: `train/loss`, `train/aux` (DPO also `train/acc`, `train/margin`), `train/lr`, `train/grad_norm`,
  `train/rho_A`, `train/samples`, `train/steps_per_sec`, `act/avg_loops`, `act/enabled`;
  the config stores all training parameters and the model architecture.
- The checkpoint saves the wandb run id, so `--from_resume 1` **continues writing to the same run** — curves never break.
- If wandb is not installed or login fails, only a warning is printed and training proceeds normally.

### Reading the training log

| Metric | Healthy | Needs attention |
|---|---|---|
| loss | Steadily decreasing; ±0.1 step-to-step jitter is normal | Rising for hundreds of steps → lower the learning rate and resume |
| gnorm | 0.2–1 pretraining, 1–2 early SFT | Sustained spikes → learning rate too high |
| loops | After ACT activates, slowly decreases and stabilizes at 2–4 | Drops below 2 within a few hundred steps → lower `--act_ponder_coef` |
| aux | 1e-3 ~ 1e-2 | Sudden large changes — check together with loops |
| ρ(A) | Below 1, slowly decreasing during training | Near 1 → watch numerical stability |

---

## Training Parameters Reference

### Data parameters

| Parameter | Script | Default | Description |
|---|---|---|---|
| `--data` | All | — | Corpus glob, comma-separated for multiple. pretrain: `.bin` (recommended) or `.txt`/`.jsonl`; sft/dpo: jsonl |
| `--max_tokens` | pretrain | Unlimited | Max tokens to use |
| `--max_docs` | pretrain | Unlimited | [Text corpus] Max documents to load |
| `--max_samples` | sft / dpo | Unlimited | Max dialogues / preference pairs to use |
| `--max_seq_len` | All | 512 | Sequence length (longer dialogues are truncated) |
| `--tokenizer` | All | `tokenizer` | Tokenizer directory |
| `--from_weight` | All | none / pretrain / sft | Name of initial weights (under `save_dir`) |
| `--beta` | dpo | 0.1 | DPO temperature |
| `--n_loops_train` | sft / dpo | Inherit from pretraining | Training loop count; by default reads `n_loops_train` from `config.json`, falling back to `max_loop_iters` |

### Training parameters (shared by pretrain / sft / dpo, see `baize/trainer.py`)

| Parameter | Default | Description |
|---|---|---|
| `--save_dir` | `out` | Output directory for weights / checkpoints |
| `--epochs` | 2 / 3 / 1 | Training epochs |
| `--max_steps` | Unlimited | Max optimizer steps (min of this and epochs) |
| `--batch_size` | 8 | Per-GPU micro-batch |
| `--accumulation_steps` | 1 | Gradient accumulation; samples per step = batch_size × accumulation_steps × GPUs |
| `--learning_rate` | 5e-4 / 1e-4 / 1e-6 | Peak learning rate |
| `--warmup_steps` | 1% of total steps | Linear warmup steps, then cosine decay to 0.1 × peak |
| `--weight_decay` | 0.1 | Applies only to ≥2-D weight matrices |
| `--dtype` | `bfloat16` | bfloat16 / float16 / float32 (FSDP does not support float16) |
| `--grad_clip` | 1.0 | Gradient clipping norm |
| `--log_interval` / `--save_interval` | 20 / 200 | Log / checkpoint interval (steps) |
| `--snapshot_interval` | 0 | Every N steps, additionally save an fp16 weight snapshot to `<save_dir>/snapshots/step_XXXXXX/` (non-overwriting, directly usable for inference / evaluation) |
| `--from_resume` | 0 | Resume from `ckpt_<save_weight>.pt` (data position, optimizer, and step count all restored) |
| `--fsdp` | 0 | Shard parameters / gradients / optimizer state with FSDP2 under torchrun multi-GPU |
| `--grad_checkpoint` | 0 | Activation recomputation: store only inputs per recurrent loop and per Prelude/Coda layer, recompute in backward |
| `--act_start_step` | Pretraining: 10% of total steps; SFT/DPO (loading pretrained weights): 0 | Run all loops with ACT off for the first N steps, then enable early stopping; 0 = enable from the start |
| `--use_compile` | 0 | `torch.compile` |
| `--seed` | 42 | Random seed (also determines data shuffle order) |
| `--use_wandb` | 0 | Log training metrics to Weights & Biases (requires `pip install wandb` and `wandb login`) |
| `--wandb_project` / `--wandb_entity` | `baize` / logged-in account | wandb project / team name |
| `--wandb_run_name` | `<stage>-<time>` | Run name, e.g. `pretrain-1004-0628` |
| `--wandb_mode` | `online` | `offline` writes only to local `<save_dir>/wandb/`; upload later with `wandb sync` (for servers without internet) |

### Model architecture parameters (pretrain.py only)

| Parameter | Default | Description |
|---|---|---|
| `--hidden_size` | 512 | Hidden dimension |
| `--num_attention_heads` / `--num_key_value_heads` / `--head_dim` | 8 / 2 / 64 | Attention heads / KV heads (GQA) / per-head dimension |
| `--intermediate_size` | 1024 | Dense FFN intermediate dimension |
| `--prelude_layers` | 2 | Prelude layers P |
| `--coda_layers` | 2 | Coda layers C |
| `--max_loop_iters` | 8 | Maximum (inference default) loop count T |
| `--n_loops_train` | Same as max_loop_iters | Loops actually used during training; can be less than max_loop_iters; written to `config.json`, inherited by SFT/DPO by default |
| `--max_seq_len` | 512 | Training sequence length |
| `--attn_type` | `gqa` | Attention type (gqa / mla) |
| `--use_moe` | 0 | Whether Prelude/Coda use MoE (the recurrent block always uses MoE) |
| `--n_experts` / `--n_experts_per_tok` | 8 / 2 | MoE experts / experts activated per token |
| `--moe_intermediate_size` | 512 | Intermediate dimension per expert |
| `--moe_capacity_factor` | 0 | >0 drops overflowing tokens by capacity; 0 = no drop |
| `--use_act` | 1 | Enable ACT adaptive early stopping |
| `--act_init_bias` | −3 | Halting predictor initial bias (−3 → initial halting probability ≈0.05, all loops run at first) |
| `--act_ponder_coef` | 1e-3 | Ponder cost coefficient — larger favors earlier stopping; 0 = off |

### eval.py parameters

| Parameter | Default | Description |
|---|---|---|
| `--mode` | `chat` | `ppl` perplexity / `generate` plain continuation (pretrained weights) / `chat` dialogue (after SFT) / `bench` standard benchmarks |
| `--weight` / `--save_dir` | `pretrain` / `out` | Weight name and directory (must contain `config.json`) |
| `--loops` | Same as config | Inference loop count; applies to ppl / generate / chat / bench |
| `--data` / `--max_docs` | — | [ppl] Evaluation corpus (txt / jsonl) and max documents |
| `--prompt` | None | [generate] Continuation prefix; separate multiple with `\|\|`; interactive input if omitted |
| `--temperature` / `--top_p` / `--top_k` | 0.7 / 0.85 / 50 | [generate] Sampling parameters; temperature ≤ 0 is greedy |
| `--repetition_penalty` | 1.2 | [generate] Repetition penalty; 1.1–1.3 recommended for small models |
| `--max_new_tokens` | 256 | Max generation length |
| `--bench` | `arc-easy` | [bench] Comma-separated: `arc-easy,arc-challenge,mmlu,ceval[:subject],hellaswag,gsm8k,jsonl:<path>` |
| `--bench_limit` | Unlimited | [bench] Max questions per benchmark; 500 questions gives ~±2% error |
| `--chat` | 0 | [bench] Wrap questions in the chat template; use 1 for SFT models |
| `--bench_out` | None | [bench] Write results to json |
| `--hf_endpoint` | Env var | HuggingFace endpoint, e.g. `https://hf-mirror.com` |

### demo.py parameters

| Parameter | Default | Description |
|---|---|---|
| `--prompt` | None | Single generation when given; otherwise interactive chat |
| `--loops` | Same as config | Inference loop count (may exceed the training value) |
| `--compare` | None | Loop-count comparison, e.g. `2,4,8,16` |
| `--temperature` | 0.7 | Sampling temperature (≤0 is greedy) |
| `--top_k` | 50 | Top-K sampling |
| `--top_p` | 0.85 | Top-P (nucleus) sampling |
| `--repetition_penalty` | 1.05 | Repetition penalty (>1 reduces repetition) |
| `--max_new_tokens` | 128 | Max new tokens |
| `--web` | False | Launch the Gradio web UI |
| `--bench` | False | Throughput test |

---

## RDT Training Tips

### Loop-count hyperparameter strategy

```
training loops < max loops < inference loops (depth extrapolation)
e.g. --n_loops_train 4  --max_loop_iters 8  → inference can use --loops 16
```

The Parcae scaling law argues that at fixed FLOPs, increasing loop count pays off more than increasing token count.
We recommend warming up with fewer loops (`--n_loops_train 4`) and gradually adding loops, rather than running all 8 from the start.

ACT early stopping: before `--act_start_step` (default 10% of total steps), ACT is off and all loops run,
so the recurrent block trains the later loops first; then ACT learns when to stop. **Do not enable ACT from the start**:
untrained later loops only inject noise — the halting head learns to stop after 1–2 loops within tens of steps,
the later loops get ≈0 gradient because their weights are ≈0, and recurrent depth collapses
(measured: from 8 loops down to 1.7).
More loops means more activation memory; when memory is tight, add `--grad_checkpoint 1`.

### Monitoring metrics

Training logs look like:
```
step:200/2000 loss:3.4521 aux:0.0083 lr:4.50e-04 gnorm:0.92 loops:7.40 ρ(A):0.921 eta:12.3min
```

- **loss**: cross-entropy loss, expected to decrease over training
- **aux**: MoE load-balancing loss + ACT ponder cost (`act_ponder_coef × average loops`), normally on the order of 1e-3 ~ 1e-2
- **loops**: average loops actually run per position this step. Equals the training loop count before ACT activates, then slowly decreases as ACT learns to stop early
  (0.13B measurement: dropped from 4 to 2.8 over ~3,000 steps, finally stabilizing at ~2.6); dropping below 2 within a few hundred steps means the ponder cost is too large (lower `--act_ponder_coef`)
- **gnorm**: pre-clip gradient norm; sustained spikes usually mean the learning rate is too high
- **ρ(A)**: largest element of the LTI matrix, must be < 1; near 0.99 is normal, near 1.0 warrants attention to numerical stability

### MoE configuration advice

Small models (< 100M): `--use_moe 1 --n_experts 8 --n_experts_per_tok 2`

`router_aux_loss_coef` (in config, default 1e-3) controls load-balancing strength:
- Too small: expert routing collapses (most tokens pick the same expert)
- Too large: over-balancing hurts model quality

### Depth extrapolation

Train with `n_loops_train` loops, then infer with `--loops N` (N > max_loop_iters);
the depth LoRA scales clamp to the learned value of the last loop, and output usually remains coherent.

Verify extrapolation visually with `--compare` mode:

```bash
python scripts/demo.py --prompt "Explain recursion" --compare 2,4,8,16,32
```

---

## Repository Layout

```
BaiZe/
├── baize/
│   ├── __init__.py            # Package exports
│   ├── config.py              # BaiZeConfig (HF PretrainedConfig)
│   ├── model.py               # RDT model: GQA/MLA, MoE, LTI, ACT, LoRA, loop-index embedding
│   ├── tokenizer.py           # BPE training + wrapper (incl. chat template, v2 exact boundary detection)
│   ├── data.py                # Corpus reading (.txt/.jsonl), memmap .bin dataset, resumable sampler
│   ├── trainer.py             # Generic training loop: warmup / resume / DDP / FSDP2 / activation recomputation
│   ├── benchmarks.py          # Standard benchmarks: ARC / MMLU / C-Eval / HellaSwag / GSM8K / local jsonl
│   └── trainer_utils.py       # LR schedule / distributed init / logging / weight IO (strict validation)
├── scripts/
│   ├── prepare_allenai.py     # AllenAI pretrain / SFT / preference data: streaming download, mixing, filtering, quota control
│   ├── train_tokenizer.py     # BPE tokenizer training
│   ├── tokenize_corpus.py     # Pre-tokenize → memmap .bin (for large-corpus training)
│   ├── pretrain.py            # Pretraining
│   ├── sft.py                 # Instruction tuning (prompt mask / multi-turn dialogue)
│   ├── dpo.py                 # Preference alignment (DPO)
│   ├── eval.py                # Perplexity / chat / standard benchmarks (supports --loops depth extrapolation)
│   └── demo.py                # Inference demo: CLI + loop comparison + throughput test + Gradio
├── demo_weights/              # Bundled toy weights (17M params, demo runs directly)
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── tokenizer/                 # Your trained tokenizer (train_tokenizer.py output)
├── data/
│   ├── corpus.txt             # Pretraining corpus (example)
│   └── sft.jsonl              # SFT data (example)
├── docs/
│   ├── allenai_data.md        # AllenAI data guide (English)
│   ├── allenai_data.zh-CN.md  # AllenAI data guide (which datasets, volume control; Chinese)
│   ├── training_report_0.1b.md   # 0.13B field report (English)
│   └── training_report_0.1b.zh-CN.md # 0.13B field report (config, curves, benchmarks, conclusions; Chinese)
├── tests/                     # Tests (python tests/test_*.py; CI in .github/workflows/tests.yml)
├── requirements.txt
└── LICENSE                    # MIT License
```

---

## FAQ

**Q: `demo.py` reports "weight directory not found"?**

A: Make sure `demo_weights/model.safetensors` exists (it should ship with the zip).
If you trained your own, point to your output directory with `--save_dir out --weight pretrain`.

**Q: `ρ(A)` close to or even equal to 1.0?**

A: The LTI parameterization theoretically guarantees A < 1, but extreme initialization or fp16 numerical issues
can approach the boundary. Watch whether the loss decreases normally; if NaN appears, switch to `--dtype bfloat16`
or lower the learning rate.

**Q: Training is much slower with `use_act=True` than `False`?**

A: ACT adds a sigmoid per loop and maintains halted masks, which has some overhead;
the bigger reason is that ACT gives each sample a different effective loop count, so the loop dimension can't be batched —
noticeable at small batch sizes. Compare with `--use_act 0`.

**Q: MoE aux-loss stays at 0?**

A: Check that the model is in `training` mode (aux-loss returns 0 in eval).
The pretrain/sft scripts set `model.train()` correctly; forgetting this in a custom wrapper triggers this issue.

**Q: Multi-GPU training reports `DDP unused parameters`?**

A: In v3, MoE computes in batch, so all expert parameters participate every step; only when ACT is not active
(`--use_act 0` or the early phase of `--act_start_step > 0`) does the halting predictor not participate,
and the training loop automatically enables `find_unused_parameters` for these two cases.

**Q: Weight loading reports "shape mismatch / missing parameters"?**

A: v3's `load_weights` validates strictly by default. The most common cause is using a different tokenizer at SFT/eval time
than in pretraining (different vocab_size), or a `config.json` that doesn't match the weights. Make sure `--tokenizer` and
`--save_dir/config.json` match pretraining; if you genuinely need partial loading, call `load_weights(model, path, strict=False)` in code.

**Q: Out of memory?**

A: Try in order: `--grad_checkpoint 1` (activation recomputation, biggest win with many loops) → reduce `--batch_size`
and increase `--accumulation_steps` → multi-GPU `--fsdp 1` (parameters / gradients / optimizer state sharded across GPUs).

**Q: How to continue pretraining on top of an existing model?**

A: Use `--from_weight <name>` to load existing safetensors weights (fresh optimizer and LR schedule);
if training was interrupted, use `--from_resume 1`: model, optimizer, step count, and data position are all restored,
continuing at the batch after the interruption.
Note that on resume, `--batch_size`, `--accumulation_steps`, GPU count, and `--seed` must match the original run, or the data position cannot be aligned.

---

## References

- [Parcae](https://arxiv.org/abs/2501.04697) — scaling law for recurrent-depth Transformers
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MLA attention compression
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — fine-grained MoE and aux-loss balancing
- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) — ACT early stopping and ponder cost
- [DPO](https://arxiv.org/abs/2305.18290) — Direct Preference Optimization
- [Tulu 3](https://arxiv.org/abs/2411.15124) — AllenAI open post-training data and recipes

---

## License

The code in this project is open-sourced under the [MIT License](LICENSE).

Training data is not covered by this license: the AllenAI datasets downloaded via `prepare_allenai.py`
(C4 / mC4, OLMo mix, Tulu 3, etc.) each follow their original licenses (mostly ODC-BY). Check
[docs/allenai_data.md](docs/allenai_data.md) and each dataset's page before use.
