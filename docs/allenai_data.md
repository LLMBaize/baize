# Training BaiZe with AllenAI Data

**English** | [简体中文](allenai_data.zh-CN.md)

This document explains how to plug AllenAI (Ai2)'s open **pretraining data** (C4 / mC4, the OLMo 2
pretraining mix, Dolmino), **post-training data** (Tulu 3 SFT, WildChat), and **preference data**
(Tulu 3 / OLMo 2 preference mixtures, UltraFeedback) into BaiZe, and how to **control the data volume**
at every stage.

The pipeline has two steps:

1. `scripts/prepare_allenai.py`: **streams** datasets from HuggingFace, mixes and filters them by ratio,
   stops once the quota is filled, and writes a local `jsonl`.
2. `train_tokenizer.py` / `sft.py` / `dpo.py` / `eval.py` read these `jsonl` files directly; pretraining
   corpora are first pre-tokenized into `.bin` (memmap, no memory footprint) with `tokenize_corpus.py`,
   which `pretrain.py` reads directly.
   At training time you can further truncate with `--max_docs` / `--max_tokens` / `--max_samples` / `--max_steps`.

> **Streaming**: datasets are never downloaded in full. The script reads and processes one sample at a
> time and stops when the quota is used up — so even for TB-scale C4 or OLMo-mix, only the portion
> actually used is downloaded.

---

## 1. Setup

```bash
pip install datasets            # new dependency (everything else as in the README)
```

**Network**: you need access to `huggingface.co`. If access is difficult from mainland China, use the mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Network problems**: if you get `ConnectionError`, `timed out`, or `[Errno 101] Network is unreachable`,
troubleshoot in this order:

1. **Mirror** (first choice in mainland China): pass `--hf_endpoint https://hf-mirror.com`. At startup the script prints `[hf] endpoint = ...` —
   confirm it shows the mirror address. You can also `export HF_ENDPOINT=https://hf-mirror.com`, but you **must export it before starting Python**:
   huggingface_hub / datasets read this variable only once at import time; writing `HF_ENDPOINT=...` (without export),
   setting it in another terminal, or changing `os.environ` after import in Python all have no effect.
   Another pitfall: for paginated APIs like file listing, the "next page" URL comes from the server in response
   headers, and hf-mirror still returns huggingface.co — so with only `HF_ENDPOINT` set, page 1 goes through the
   mirror but page 2 onward hits the official site again (`allenai/c4`'s `multilingual/` has tens of thousands of
   files, so pagination is inevitable). `--hf_endpoint` rewrites pagination URLs back to the mirror.
2. **IPv6 issues**: if `Network is unreachable` comes and goes, the domain probably resolved to an IPv6 address
   while the server has no IPv6 route. Add a line `precedence ::ffff:0:0/96 100` to `/etc/gai.conf` to prefer IPv4.
   Verify with `curl -4 -I https://huggingface.co` vs `curl -6 -I https://huggingface.co`.
3. **Proxy**: `export HTTPS_PROXY=http://<proxy-host>:<port>`.
4. **Offline**: download files locally first, then read local files (`json@<local-glob>` makes the script use the `json` reader on local files):

   ```bash
   # Download only the first 20 training shards of mC4 Chinese (a few hundred MB per shard, adjust as needed)
   huggingface-cli download allenai/c4 --repo-type dataset \
       --include "multilingual/c4-zh.tfrecord-000[01]*.json.gz" --local-dir data/hf/c4
   python scripts/prepare_allenai.py --task pretrain \
       --sources "json@data/hf/c4/multilingual/c4-zh.*.json.gz" --max_chars 1_000_000_000 \
       --out data/allenai_pretrain.jsonl
   ```
   Same for SFT and DPO datasets: download with `huggingface-cli download allenai/tulu-3-sft-mixture --repo-type dataset --local-dir ...`,
   then read with `--sources "parquet@<local-dir>/data/*.parquet"`.

**Login**: all datasets in the table are public and generally don't require login. If a dataset requires
accepting terms on its webpage first, accept them there, then run `huggingface-cli login`
(or set the `HF_TOKEN` environment variable).

---

## 2. Which AllenAI Data to Download

Run `python scripts/prepare_allenai.py --list` to see all built-in data sources.

### 2.1 Pretraining data (`--task pretrain`)

| Preset | HF dataset | Language | Size (approx.) | Suggested use |
|---|---|---|---|---|
| `c4-zh` | `allenai/c4` (`multilingual/c4-zh.*`, i.e. mC4 Chinese) | Chinese | Hundreds of GB | **Primary Chinese corpus, first choice** |
| `c4-en` | `allenai/c4` (`en/`) | English | 365M docs / ~156B tokens / ~300GB | Primary English corpus |
| `c4-realnewslike` | `allenai/c4` (`realnewslike/`) | English | 13M docs | News-style English |
| `c4-zh-val` / `c4-en-val` | `allenai/c4` validation shards | ZH / EN | — | **PPL evaluation sets** |
| `olmo-mix-dclm` | `allenai/olmo-mix-1124#dclm` | English | OLMo 2 pretraining backbone | High-quality English web |
| `olmo-mix-wiki` | `allenai/olmo-mix-1124#wiki` | Mostly English | — | Encyclopedic knowledge |
| `olmo-mix-pes2o` / `olmo-mix-arxiv` | `allenai/olmo-mix-1124` | English | — | Academic text |
| `olmo-mix-starcoder` | `allenai/olmo-mix-1124#starcoder` | Code | — | Coding ability |
| `olmo-mix-open-web-math` | `allenai/olmo-mix-1124#open-web-math` | English | — | Math |
| `dolmino-wiki` / `dolmino-flan` / `dolmino-math` / `dolmino-stackexchange` | `allenai/dolmino-mix-1124` | English | — | High-quality "annealing" data, for late training |

**Recommended mixes** (BaiZe's default tokenizer targets Chinese and English):

- Chinese only: `c4-zh`
- Chinese-English mix: `c4-zh:0.7,c4-en:0.3`
- Some math and code ability: `c4-zh:0.6,c4-en:0.2,olmo-mix-open-web-math:0.1,olmo-mix-starcoder:0.1`

> ⚠️ The **config names for OLMo-mix and Dolmino are subject to the HuggingFace dataset pages**
> (these repos get updated, e.g. newer Dolma 3 / OLMo 3 data).
> If a preset errors with something like `BuilderConfig ... not found`, the script skips that source
> and continues with the others. In that case, look up the correct config name or file path on the
> dataset page and use a custom spec instead — see 2.4.
>
> `allenai/dolma` itself uses a legacy loading script that modern `datasets` no longer supports, so it is not built in.
> If you need Dolma, use OLMo-mix (the successor of the Dolma series) or the official Dolma tooling.

### 2.2 Post-training data (`--task sft`)

| Preset | HF dataset | Size (approx.) | Notes |
|---|---|---|---|
| `tulu3` | `allenai/tulu-3-sft-mixture` | 940k | **Main Tulu 3 SFT mixture, first choice**; has a `source` field for filtering by origin |
| `tulu3-olmo2` | `allenai/tulu-3-sft-olmo-2-mixture` | — | The version used by OLMo 2 Instruct |
| `tulu3-personas-math` | `allenai/tulu-3-sft-personas-math` | — | Synthetic math problems |
| `tulu3-personas-code` | `allenai/tulu-3-sft-personas-code` | — | Synthetic coding problems |
| `tulu3-personas-if` | `allenai/tulu-3-sft-personas-instruction-following` | — | Precise instruction following |
| `wildchat` | `allenai/WildChat-1M` (`conversation` field) | 1M | Real user conversations, **lots of Chinese** — pair with `--language Chinese` |

Tulu 3 is mostly English. To get Chinese dialogue ability, mix like this:

```bash
--sources wildchat:0.5,tulu3:0.5 --language Chinese   # language filter only applies to datasets with a language field
# Or filter by CJK character ratio — applies to all datasets:
--sources tulu3,wildchat --min_cjk_ratio 0.3
```

### 2.3 Preference data (`--task dpo`, for `scripts/dpo.py`)

| Preset | HF dataset | Size (approx.) | Notes |
|---|---|---|---|
| `tulu3-pref` | `allenai/llama-3.1-tulu-3-8b-preference-mixture` | 270k pairs | **Tulu 3 preference mixture, first choice** |
| `tulu3-pref-olmo2` | `allenai/olmo-2-1124-7b-preference-mix` | — | The version used for OLMo 2 7B DPO |
| `ultrafeedback` | `allenai/ultrafeedback_binarized_cleaned` (split `train_prefs`) | 60k pairs | Cleaned UltraFeedback |

Output is one `{"chosen": [dialogue], "rejected": [dialogue]}` per line. Cleaning rules:
- The prompts of the two dialogues (all turns except the last assistant reply) must be identical, otherwise the pair is dropped;
- Pairs where chosen and rejected replies are identical are dropped;
- The string format `{"prompt": "...", "chosen": "...", "rejected": "..."}` is also accepted and converted to dialogues automatically.

`--language` / `--source_filter` / `--min_cjk_ratio` / `--max_turns` also apply (computed on chosen).
For small models, 10k–50k preference pairs are usually enough; the DPO learning rate should be very small
(default 1e-6), trained for 1 epoch only.

### 2.4 Custom data sources (datasets not in the built-in list)

Besides preset names, `--sources` also accepts HF repos directly:

```bash
--sources "allenai/olmo-mix-1124#wiki:0.5"                  # repo#config:weight
--sources "allenai/c4@multilingual/c4-ja.*.json.gz"          # repo@file-glob (e.g. mC4 Japanese)
--sources "allenai/some-new-dataset#some-config" --text_field text     # use --text_field when the text field isn't "text"
--sources "allenai/some-dialogue-dataset" --task sft --messages_field conversation
```

---

## 3. Controlling Data Volume

Data volume can be controlled at **two stages**: how much is written at download time, and how much is
read / how many steps are run at training time.

### 3.1 At download time (`prepare_allenai.py`)

| Parameter | Description |
|---|---|
| `--max_docs N` | Cap on documents / dialogues written |
| `--max_tokens N` | Token cap (requires `--tokenizer` for exact counting) |
| `--max_chars N` | Character cap (no tokenizer needed — useful before the tokenizer is trained) |
| `--sources a:w1,b:w2` | Mixing weights. **The total quota is split across sources by weight** — e.g. `--max_docs 1000` with `a:3,b:1` means 750 docs from a and 250 from b |
| `--max_scan N` | Max raw samples scanned per source, so over-strict filters don't read forever |
| `--val_docs N` | First write N extra samples to `<out>.val.jsonl` as a validation set, not counted against the training quota |
| `--skip N` | Skip the first N samples of each source, to fetch a slice that doesn't overlap a previous run (keep `--seed` unchanged) |
| `--shuffle_buffer N` / `--seed` | Streaming shuffle (also shuffles shard order); 0 reads in original order |

Rules:

- All three caps (docs / tokens / chars) can be given together — **whichever is hit first stops the run**.
- Sources are written **interleaved** by weight, not one after another.
- When a source is exhausted or errors out, its quota is **not** reassigned to other sources. Check `docs` and `error` in the summary to spot this.

Filter parameters:

| Parameter | Applies to | Description |
|---|---|---|
| `--min_chars` | pretrain | Minimum document length in characters (default 50) |
| `--max_doc_chars` | pretrain | Per-document truncation length (0 = no truncation) |
| `--min_cjk_ratio` | All | Minimum CJK character ratio, e.g. 0.3 |
| `--language` | sft | Filter by the sample's `language` field (WildChat: `Chinese`, `English`, …) |
| `--source_filter` | sft | Keep only samples whose `source` field contains the given substring (Tulu 3); multiple allowed |
| `--max_turns` | sft | Cap on assistant reply turns |
| `--dedup` | All | Exact deduplication (on by default) |

SFT data is also normalized automatically:

- Only `system` / `user` / `assistant` roles are kept; dialogues containing other roles (e.g. tool calls) are **dropped entirely**.
- Trailing turns that don't end with an assistant reply are removed.
- WildChat dialogues flagged as `toxic` are dropped.

Every run additionally writes `<out>.manifest.json` recording:

- How many docs / tokens / chars each source actually wrote;
- How many samples were scanned, filtered out, and deduplicated;
- Error messages and the full parameter set of this run.

Use it to verify the data mix.

### 3.2 At training time

| Script | Parameter | Description |
|---|---|---|
| `train_tokenizer.py` | `--max_docs` | Train the tokenizer on only the first N docs (a few hundred thousand is enough for a stable vocab) |
| `tokenize_corpus.py` | `--max_docs` / `--max_tokens` | Truncate during pre-tokenization |
| `pretrain.py` | `--max_docs` / `--max_tokens` | Truncate when loading the corpus (`.bin` supports only `--max_tokens`) |
| `pretrain.py` / `sft.py` / `dpo.py` | `--max_steps` | Max optimizer steps; the smaller of this and `--epochs` wins |
| `sft.py` / `dpo.py` | `--max_samples` | Max valid dialogues / preference pairs to use |
| `eval.py` | `--max_docs` | PPL evaluation uses only the first N docs |

`--data` / `--corpus` accept `.txt` (one document per line) and `.jsonl` (`text` field);
separate multiple globs with commas, e.g. `--data "data/allenai_pretrain.jsonl,data/corpus*.txt"`.

### 3.3 How much data to prepare

Rule of thumb (Chinchilla): pretraining tokens ≈ **20×** the parameter count.
BaiZe's default config is 17M–40M parameters; the recurrent block shares weights, but every loop consumes compute.

| Goal | Parameters | Suggested pretraining tokens | Reference command |
|---|---|---|---|
| Get the pipeline running | Any | 5M–10M | `--max_docs 20000` |
| Small-scale experiment | ~17M | 300M–500M | `--max_tokens 400_000_000` |
| Fully train the default config | ~40M | 800M–1B | `--max_tokens 1_000_000_000` |

Conversion references:

- With a 6400-token BPE vocab, Chinese is roughly **1–1.5 characters per token**, English roughly **3–4 characters per token**.
- The written `jsonl` size is roughly chars × bytes per char: ~3 bytes per Chinese character, ~1 byte per English character.
- These are rough estimates — **trust the actual numbers in the manifest**.
- Feeding text corpora directly to `pretrain.py` keeps tokens in memory (~400MB per 100M tokens);
  for large corpora, convert to `.bin` with `tokenize_corpus.py` first (2 bytes per token when vocab < 65536; memmap reads use no memory).

For SFT, **20k–100k samples** is usually enough. Note that dialogues longer than `--max_seq_len` (default 512)
get truncated; if the reply part is under 2 tokens after truncation, the dialogue is skipped.

---

## 4. End-to-End Example

Because `--max_tokens` needs a tokenizer, and the tokenizer is trained on the corpus, the recommended order is:
**download by character count first, train the tokenizer, then truncate by tokens at training time**.

```bash
# ① Pretraining corpus: Chinese 70% + English 30%, ~1.5B chars, plus 2,000 validation docs
python scripts/prepare_allenai.py --task pretrain \
    --sources c4-zh:0.7,c4-en:0.3 \
    --max_chars 1_500_000_000 --val_docs 2000 \
    --out data/allenai_pretrain.jsonl

# ② Train the tokenizer on 300k of those docs (bump the vocab for Chinese corpora)
python scripts/train_tokenizer.py --corpus data/allenai_pretrain.jsonl \
    --max_docs 300000 --vocab_size 16000 --save_dir tokenizer

# ③ Pre-tokenize to memmap (once), then pretrain: use at most 1B tokens, at most 20k steps
python scripts/tokenize_corpus.py --data data/allenai_pretrain.jsonl --tokenizer tokenizer \
    --out data/pretrain.bin --max_tokens 1_000_000_000
python scripts/pretrain.py --data data/pretrain.bin --max_steps 20000 --epochs 1

# ④ Post-training data: Tulu 3 + WildChat Chinese, 50k samples, plus 500 validation
python scripts/prepare_allenai.py --task sft \
    --sources tulu3:0.5,wildchat:0.5 --language Chinese \
    --max_docs 50000 --val_docs 500 --out data/allenai_sft.jsonl

# ⑤ SFT (can also be combined with the bundled sft.jsonl)
python scripts/sft.py --data "data/allenai_sft.jsonl,data/sft.jsonl" \
    --from_weight pretrain --max_samples 50000

# ⑥ Preference alignment (optional): 20k pairs of Tulu 3 preference data
python scripts/prepare_allenai.py --task dpo --sources tulu3-pref --max_docs 20000 \
    --out data/allenai_dpo.jsonl
python scripts/dpo.py --data data/allenai_dpo.jsonl --from_weight sft

# ⑦ Perplexity on the validation set, plus standard benchmarks (ARC is also an AllenAI dataset)
python scripts/eval.py --weight pretrain --mode ppl \
    --data data/allenai_pretrain.val.jsonl --max_docs 2000
python scripts/eval.py --weight dpo --mode bench --bench arc-easy,arc-challenge,ceval --chat 1
```

If you already have a tokenizer, you can control by tokens precisely at download time too:

```bash
python scripts/prepare_allenai.py --task pretrain --sources c4-zh \
    --max_tokens 200_000_000 --tokenizer tokenizer
```

To append a non-overlapping batch of data on top of a previous run (keep `--seed` the same):

```bash
python scripts/prepare_allenai.py --task pretrain --sources c4-zh \
    --max_chars 500_000_000 --skip 3000000 --out data/allenai_pretrain_part2.jsonl
python scripts/tokenize_corpus.py --data data/allenai_pretrain_part2.jsonl --out data/pretrain_part2.bin
python scripts/pretrain.py --data "data/pretrain*.bin"
```

### Output format

```jsonc
// Pretraining: data/allenai_pretrain.jsonl
{"text": "Full document text (may contain newlines)", "source": "c4-zh"}
// SFT: data/allenai_sft.jsonl (compatible with the bundled data/sft.jsonl format)
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "source": "tulu3"}
// DPO: data/allenai_dpo.jsonl
{"chosen": [{"role": "user", ...}, {"role": "assistant", "content": "the better reply"}],
 "rejected": [{"role": "user", ...}, {"role": "assistant", "content": "the worse reply"}], "source": "tulu3-pref"}
```

---

## 5. Notes

- **Licensing**:
  - C4 / mC4, OLMo-mix, Dolmino, and Tulu 3 are all under **ODC-BY**. Some Tulu 3 subsets carry their own licenses, plus usage terms for model-generated upstream data.
  - WildChat-1M has its own terms of use.
  - Review each dataset page individually before commercial use.
- **Chinese capability**: Ai2 data is mostly English; Chinese comes mainly from `c4-zh` (mC4) and WildChat.
  mC4 Chinese quality is uneven — pair with `--min_chars` / `--min_cjk_ratio`, or apply your own quality filtering.
- **Vocabulary**: the default 6400 vocab is small for large-scale Chinese corpora. Retrain the tokenizer on AllenAI data with a vocab of 8000–32000.
  **Changing the vocab requires pretraining from scratch** — old weights cannot be reused.
- **Preset rot**: dataset structures on HF may change. On errors, check `ERROR` in the summary and the dataset page,
  and switch to the `repo#config` or `repo@file-glob` syntax — no code changes needed.
