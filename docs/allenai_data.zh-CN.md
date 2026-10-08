# 使用 AllenAI 数据训练 BaiZe

[English](allenai_data.md) | **简体中文**

本文说明如何把 AllenAI（Ai2）开源的**预训练数据**（C4 / mC4、OLMo 2 预训练混合、Dolmino）、
**后训练数据**（Tulu 3 SFT、WildChat）和**偏好数据**（Tulu 3 / OLMo 2 偏好混合、UltraFeedback）接入 BaiZe，
以及如何在每个环节**控制数据量**。

整个流程分两步：

1. `scripts/prepare_allenai.py`：从 HuggingFace **流式**读取数据集，按比例混合、过滤，
   读够配额即停止，写出本地 `jsonl`。
2. `train_tokenizer.py` / `sft.py` / `dpo.py` / `eval.py` 直接读取这些 `jsonl`；预训练语料先用
   `tokenize_corpus.py` 预分词成 `.bin`（memmap，不占内存），`pretrain.py` 直接读 `.bin`。
   训练时还可以再用 `--max_docs` / `--max_tokens` / `--max_samples` / `--max_steps` 截断数据量。

> **流式读取**：数据不会整库下载。脚本读一条、处理一条，配额用完就停，
> 所以就算是 TB 级的 C4 或 OLMo-mix，也只会下载实际用到的那部分。

---

## 1. 环境准备

```bash
pip install datasets            # 新增依赖（其余依赖同 README）
```

**网络**：需要能访问 `huggingface.co`。如果在国内访问困难，可以用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**网络问题**：如果报 `ConnectionError`、`timed out`，或者 `[Errno 101] Network is unreachable`，按顺序排查：

1. **镜像**（国内首选）：加参数 `--hf_endpoint https://hf-mirror.com`。脚本启动时会打印 `[hf] endpoint = ...`，
   确认是镜像地址即可。也可以用 `export HF_ENDPOINT=https://hf-mirror.com`，但**必须在启动 Python 之前 export**：
   huggingface_hub / datasets 只在 import 时读取一次该变量，写成 `HF_ENDPOINT=...`（没有 export）、
   在别的终端设置、或在 Python 里 import 之后再改 `os.environ` 都不会生效。
   另一个坑：列目录等分页接口的"下一页"地址由服务器在响应头里返回，hf-mirror 返回的仍是 huggingface.co，
   所以只设置 `HF_ENDPOINT` 时第 1 页走镜像、第 2 页起又会去连官方站点（`allenai/c4` 的 `multilingual/`
   有上万个文件，必然翻页）。`--hf_endpoint` 会把翻页地址一并改写回镜像。
2. **IPv6 问题**：`Network is unreachable` 时通时断，多半是域名解析到了 IPv6 地址，但服务器没有 IPv6 路由。
   在 `/etc/gai.conf` 里加一行 `precedence ::ffff:0:0/96 100`，让系统优先走 IPv4。
   可以用 `curl -4 -I https://huggingface.co` 和 `curl -6 -I https://huggingface.co` 对比验证。
3. **代理**：`export HTTPS_PROXY=http://<代理地址>:<端口>`。
4. **离线**：先把文件下载到本地，再读本地文件（`json@<本地glob>` 会让脚本用 `json` 读取器读本地文件）：

   ```bash
   # 只下载 mC4 中文的前 20 个训练分片（约每片几百 MB，可按需增减）
   huggingface-cli download allenai/c4 --repo-type dataset \
       --include "multilingual/c4-zh.tfrecord-000[01]*.json.gz" --local-dir data/hf/c4
   python scripts/prepare_allenai.py --task pretrain \
       --sources "json@data/hf/c4/multilingual/c4-zh.*.json.gz" --max_chars 1_000_000_000 \
       --out data/allenai_pretrain.jsonl
   ```
   SFT、DPO 数据集同理：用 `huggingface-cli download allenai/tulu-3-sft-mixture --repo-type dataset --local-dir ...`
   下载后，用 `--sources "parquet@<本地目录>/data/*.parquet"` 读取。

**登录**：表中的数据集都是公开数据集，一般不需要登录。如果某个数据集要求先在网页上同意使用条款，
先在对应页面点同意，再运行 `huggingface-cli login`（或设置 `HF_TOKEN` 环境变量）。

---

## 2. 下载哪些 AllenAI 数据

运行 `python scripts/prepare_allenai.py --list` 可以查看全部内置数据源。

### 2.1 预训练数据（`--task pretrain`）

| 预设名 | HF 数据集 | 语言 | 规模（约） | 用途建议 |
|---|---|---|---|---|
| `c4-zh` | `allenai/c4`（`multilingual/c4-zh.*`，即 mC4 中文） | 中文 | 数百 GB | **中文主语料，首选** |
| `c4-en` | `allenai/c4`（`en/`） | 英文 | 3.65 亿篇 / 约 156B token / 约 300GB | 英文主语料 |
| `c4-realnewslike` | `allenai/c4`（`realnewslike/`） | 英文 | 1300 万篇 | 新闻风格英文 |
| `c4-zh-val` / `c4-en-val` | `allenai/c4` 验证集分片 | 中 / 英 | — | **PPL 评估集** |
| `olmo-mix-dclm` | `allenai/olmo-mix-1124#dclm` | 英文 | OLMo 2 预训练主体 | 高质量英文网页 |
| `olmo-mix-wiki` | `allenai/olmo-mix-1124#wiki` | 英文为主 | — | 百科知识 |
| `olmo-mix-pes2o` / `olmo-mix-arxiv` | `allenai/olmo-mix-1124` | 英文 | — | 学术文本 |
| `olmo-mix-starcoder` | `allenai/olmo-mix-1124#starcoder` | 代码 | — | 代码能力 |
| `olmo-mix-open-web-math` | `allenai/olmo-mix-1124#open-web-math` | 英文 | — | 数学 |
| `dolmino-wiki` / `dolmino-flan` / `dolmino-math` / `dolmino-stackexchange` | `allenai/dolmino-mix-1124` | 英文 | — | 高质量“退火”数据，放在训练后期 |

**推荐组合**（BaiZe 默认分词器面向中英文）：

- 只练中文：`c4-zh`
- 中英混合：`c4-zh:0.7,c4-en:0.3`
- 想让模型懂一点数学和代码：`c4-zh:0.6,c4-en:0.2,olmo-mix-open-web-math:0.1,olmo-mix-starcoder:0.1`

> ⚠️ OLMo-mix 和 Dolmino 的 **config 名称以 HuggingFace 数据集页面为准**
> （这些仓库会更新，例如更新的 Dolma 3 / OLMo 3 数据）。
> 如果某个预设报 `BuilderConfig ... not found` 之类的错误，脚本会跳过这个数据源，其它数据源照常继续。
> 这时去数据集页面查到正确的 config 名或文件路径，用自定义写法替换，见 2.4。
>
> `allenai/dolma` 本体使用的是旧版加载脚本，新版 `datasets` 不再支持，所以没有内置。
> 需要 Dolma 时，请用 OLMo-mix（Dolma 系列的后续版本），或参考 Dolma 官方工具链。

### 2.2 后训练数据（`--task sft`）

| 预设名 | HF 数据集 | 规模（约） | 说明 |
|---|---|---|---|
| `tulu3` | `allenai/tulu-3-sft-mixture` | 94 万条 | **Tulu 3 主 SFT 混合，首选**；带 `source` 字段，可按来源筛选 |
| `tulu3-olmo2` | `allenai/tulu-3-sft-olmo-2-mixture` | — | OLMo 2 Instruct 使用的版本 |
| `tulu3-personas-math` | `allenai/tulu-3-sft-personas-math` | — | 合成数学题 |
| `tulu3-personas-code` | `allenai/tulu-3-sft-personas-code` | — | 合成代码题 |
| `tulu3-personas-if` | `allenai/tulu-3-sft-personas-instruction-following` | — | 精确指令遵循 |
| `wildchat` | `allenai/WildChat-1M`（`conversation` 字段） | 100 万条 | 真实用户对话，**中文较多**，配合 `--language Chinese` 使用 |

Tulu 3 以英文为主。想要中文对话能力时，可以这样组合：

```bash
--sources wildchat:0.5,tulu3:0.5 --language Chinese   # language 过滤只对带 language 字段的数据集生效
# 或者按中文字符占比筛选，对所有数据集都生效：
--sources tulu3,wildchat --min_cjk_ratio 0.3
```

### 2.3 偏好数据（`--task dpo`，供 `scripts/dpo.py` 使用）

| 预设名 | HF 数据集 | 规模（约） | 说明 |
|---|---|---|---|
| `tulu3-pref` | `allenai/llama-3.1-tulu-3-8b-preference-mixture` | 27 万对 | **Tulu 3 偏好混合，首选** |
| `tulu3-pref-olmo2` | `allenai/olmo-2-1124-7b-preference-mix` | — | OLMo 2 7B DPO 使用的版本 |
| `ultrafeedback` | `allenai/ultrafeedback_binarized_cleaned`（split `train_prefs`） | 6 万对 | UltraFeedback 清洗版 |

输出每行 `{"chosen": [对话], "rejected": [对话]}`。清洗规则：
- 两条对话的 prompt（除最后一条 assistant 回复外的所有轮次）必须完全一致，否则丢弃；
- chosen 与 rejected 回复相同的样本丢弃；
- 也接受 `{"prompt": "...", "chosen": "...", "rejected": "..."}` 的字符串格式，会自动转成对话。

`--language` / `--source_filter` / `--min_cjk_ratio` / `--max_turns` 同样适用（按 chosen 计算）。
偏好数据一般 1 万到 5 万对就够小模型用；DPO 学习率要很小（默认 1e-6），只训 1 个 epoch。

### 2.4 自定义数据源（不在内置列表里的数据集）

`--sources` 除了预设名，也可以直接写 HF 仓库：

```bash
--sources "allenai/olmo-mix-1124#wiki:0.5"                  # 仓库#config:权重
--sources "allenai/c4@multilingual/c4-ja.*.json.gz"          # 仓库@文件glob（例如 mC4 日文）
--sources "allenai/某新数据集#某config" --text_field text     # 文本字段名不是 text 时用 --text_field 指定
--sources "allenai/某对话数据集" --task sft --messages_field conversation
```

---

## 3. 数据量控制

数据量可以在**两个环节**控制：下载时控制写多少，训练时控制读多少、训多少步。

### 3.1 下载时（`prepare_allenai.py`）

| 参数 | 说明 |
|---|---|
| `--max_docs N` | 写出的文档数 / 对话数上限 |
| `--max_tokens N` | token 数上限（需要 `--tokenizer`，用来精确计数） |
| `--max_chars N` | 字符数上限（不需要分词器，适合还没训练分词器的时候） |
| `--sources a:w1,b:w2` | 混合权重。**总配额按权重分给各数据源**，例如 `--max_docs 1000` 配 `a:3,b:1`，就是 a 750 篇、b 250 篇 |
| `--max_scan N` | 每个数据源最多扫描多少条原始样本，防止过滤条件太严时一直读下去 |
| `--val_docs N` | 先额外写出 N 条到 `<out>.val.jsonl` 作验证集，不占训练配额 |
| `--skip N` | 每个数据源先跳过前 N 条，用来取和上次不重叠的数据切片（保持 `--seed` 不变） |
| `--shuffle_buffer N` / `--seed` | 流式随机打乱（同时打乱分片顺序），0 表示按原始顺序读 |

规则：

- 三种上限（docs / tokens / chars）可以同时给，**任一个先达到就停**。
- 各数据源按权重**交错**写出，不会先写完一个数据源再写下一个。
- 某个数据源读完或出错时，它的配额**不会**转给其它数据源。看汇总里的 `docs` 和 `error` 就能发现这种情况。

过滤参数：

| 参数 | 适用 | 说明 |
|---|---|---|
| `--min_chars` | pretrain | 文档最少字符数（默认 50） |
| `--max_doc_chars` | pretrain | 单篇文档截断长度（0 表示不截断） |
| `--min_cjk_ratio` | 都适用 | 中文字符占比下限，例如 0.3 |
| `--language` | sft | 按样本的 `language` 字段过滤（WildChat：`Chinese`、`English` …） |
| `--source_filter` | sft | 只保留 `source` 字段包含指定子串的样本（Tulu 3），可以给多个 |
| `--max_turns` | sft | assistant 回复轮数上限 |
| `--dedup` | 都适用 | 精确去重（默认开启） |

SFT 数据还会自动规范化：

- 只保留 `system` / `user` / `assistant` 三种角色；含工具调用等其它角色的对话**整条丢弃**。
- 去掉末尾不是 assistant 的轮次。
- WildChat 中标注为 `toxic` 的对话会被丢弃。

每次运行都会额外生成 `<out>.manifest.json`，记录：

- 每个数据源实际写出的篇数、token 数、字符数；
- 扫描、过滤、去重的条数；
- 错误信息和本次的完整参数。

用它来核对数据配比。

### 3.2 训练时

| 脚本 | 参数 | 说明 |
|---|---|---|
| `train_tokenizer.py` | `--max_docs` | 只用前 N 篇训练分词器（几十万篇就足以得到稳定的词表） |
| `tokenize_corpus.py` | `--max_docs` / `--max_tokens` | 预分词时截断 |
| `pretrain.py` | `--max_docs` / `--max_tokens` | 读入语料时截断（`.bin` 只支持 `--max_tokens`） |
| `pretrain.py` / `sft.py` / `dpo.py` | `--max_steps` | 最多训练多少个优化步，与 `--epochs` 取两者中较小的 |
| `sft.py` / `dpo.py` | `--max_samples` | 最多使用多少条有效对话 / 偏好对 |
| `eval.py` | `--max_docs` | PPL 评估只用前 N 篇 |

`--data` / `--corpus` 都支持 `.txt`（每行一篇）和 `.jsonl`（`text` 字段），
多个 glob 用逗号分隔，例如 `--data "data/allenai_pretrain.jsonl,data/corpus*.txt"`。

### 3.3 该准备多少数据

经验值（Chinchilla 法则）：预训练 token 数约为参数量的 **20 倍**。
BaiZe 默认配置在 17M~40M 参数之间，循环块共享权重，但每一圈都会消耗计算量。

| 目标 | 参数量 | 建议预训练 token | 参考命令 |
|---|---|---|---|
| 跑通流程 | 任意 | 500 万~1000 万 | `--max_docs 20000` |
| 小规模实验 | ~17M | 3 亿~5 亿 | `--max_tokens 400_000_000` |
| 默认配置充分训练 | ~40M | 8 亿~10 亿 | `--max_tokens 1_000_000_000` |

换算参考：

- BPE 词表 6400 时，中文大约 **1~1.5 个汉字一个 token**，英文大约 **3~4 个字符一个 token**。
- 写出的 `jsonl` 体积约等于字符数乘每字符字节数：中文每字约 3 字节，英文每字符约 1 字节。
- 以上只是粗略估计，**以 manifest 中实际统计的数字为准**。
- 文本语料直接喂 `pretrain.py` 时 token 存在内存里（每 1 亿 token 约 400MB）；
  大语料请先 `tokenize_corpus.py` 转成 `.bin`（词表 < 65536 时每 token 2 字节，memmap 读取不占内存）。

SFT 一般 **2 万到 10 万条**就够了。注意，超过 `--max_seq_len`（默认 512）的对话会被截断；
如果截断后回复部分少于 2 个 token，这条对话会被跳过。

---

## 4. 完整流程示例

由于 `--max_tokens` 需要分词器，而分词器又要从语料训练，推荐的顺序是：
**先按字符数下载，再训练分词器，最后在训练时用 token 数截断**。

```bash
# ① 预训练语料：中文 70% + 英文 30%，约 15 亿字符，外加 2000 篇验证集
python scripts/prepare_allenai.py --task pretrain \
    --sources c4-zh:0.7,c4-en:0.3 \
    --max_chars 1_500_000_000 --val_docs 2000 \
    --out data/allenai_pretrain.jsonl

# ② 用其中 30 万篇训练分词器（中文语料建议把词表调大）
python scripts/train_tokenizer.py --corpus data/allenai_pretrain.jsonl \
    --max_docs 300000 --vocab_size 16000 --save_dir tokenizer

# ③ 预分词成 memmap（只需一次），再预训练：最多用 10 亿 token，最多训练 2 万步
python scripts/tokenize_corpus.py --data data/allenai_pretrain.jsonl --tokenizer tokenizer \
    --out data/pretrain.bin --max_tokens 1_000_000_000
python scripts/pretrain.py --data data/pretrain.bin --max_steps 20000 --epochs 1

# ④ 后训练数据：Tulu 3 + WildChat 中文，共 5 万条，外加 500 条验证集
python scripts/prepare_allenai.py --task sft \
    --sources tulu3:0.5,wildchat:0.5 --language Chinese \
    --max_docs 50000 --val_docs 500 --out data/allenai_sft.jsonl

# ⑤ SFT（也可以和自带的 sft.jsonl 一起用）
python scripts/sft.py --data "data/allenai_sft.jsonl,data/sft.jsonl" \
    --from_weight pretrain --max_samples 50000

# ⑥ 偏好对齐（可选）：Tulu 3 偏好数据 2 万对
python scripts/prepare_allenai.py --task dpo --sources tulu3-pref --max_docs 20000 \
    --out data/allenai_dpo.jsonl
python scripts/dpo.py --data data/allenai_dpo.jsonl --from_weight sft

# ⑦ 在验证集上算困惑度，并跑标准评测（ARC 也是 AllenAI 的数据集）
python scripts/eval.py --weight pretrain --mode ppl \
    --data data/allenai_pretrain.val.jsonl --max_docs 2000
python scripts/eval.py --weight dpo --mode bench --bench arc-easy,arc-challenge,ceval --chat 1
```

已经有分词器时，下载阶段也可以直接按 token 精确控制：

```bash
python scripts/prepare_allenai.py --task pretrain --sources c4-zh \
    --max_tokens 200_000_000 --tokenizer tokenizer
```

想在上次的数据基础上再追加一批不重叠的数据（`--seed` 保持一致）：

```bash
python scripts/prepare_allenai.py --task pretrain --sources c4-zh \
    --max_chars 500_000_000 --skip 3000000 --out data/allenai_pretrain_part2.jsonl
python scripts/tokenize_corpus.py --data data/allenai_pretrain_part2.jsonl --out data/pretrain_part2.bin
python scripts/pretrain.py --data "data/pretrain*.bin"
```

### 输出格式

```jsonc
// 预训练：data/allenai_pretrain.jsonl
{"text": "文档全文（可含换行）", "source": "c4-zh"}
// SFT：data/allenai_sft.jsonl（与原 data/sft.jsonl 格式兼容）
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "source": "tulu3"}
// DPO：data/allenai_dpo.jsonl
{"chosen": [{"role": "user", ...}, {"role": "assistant", "content": "更好的回答"}],
 "rejected": [{"role": "user", ...}, {"role": "assistant", "content": "较差的回答"}], "source": "tulu3-pref"}
```

---

## 5. 注意事项

- **许可**：
  - C4 / mC4、OLMo-mix、Dolmino、Tulu 3 均为 **ODC-BY** 许可。其中 Tulu 3 的部分子集带有各自的许可，以及上游模型生成数据的使用条款。
  - WildChat-1M 有单独的使用条款。
  - 商用前请逐一查看各数据集页面。
- **中文能力**：Ai2 的数据以英文为主，中文主要来自 `c4-zh`（mC4）和 WildChat。
  mC4 中文质量参差不齐，建议配合 `--min_chars`、`--min_cjk_ratio` 使用，或自行做质量过滤。
- **词表**：默认 6400 的词表对大规模中文语料偏小，建议用 AllenAI 数据重新训练分词器，词表设为 8000~32000。
  **词表变化后要从头预训练**，旧权重无法复用。
- **预设失效**：HF 上的数据集结构可能变化。出错时，看汇总里的 `ERROR` 和数据集页面，
  改用 `仓库#config` 或 `仓库@文件glob` 的写法即可，不需要改代码。
