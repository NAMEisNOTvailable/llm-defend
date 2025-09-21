Chinese Prompt‑Injection Dataset Composer (v2)

## One‑stop data generator for Chinese prompt‑injection examples (positives) and strong hard negatives, with:

- Semantic, multi‑signal de‑duplication (SimHash + MinHash‑LSH + hashed trigram cosine; optional FAISS/Annoy acceleration).

- A domain‑specific DSL to sample strategies/carriers/deliveries/evidence and enforce coverage & invariants, plus optional YAML compilation.

- Artifact control, CJK ratio control, coverage quotas, parity checks, and pair‑aware train/dev/test splitting to reduce leakage.

## Goal. Emphasize intent‑level injection semantics (not just lexical markers), expand benign target domains, and produce verifiable dataset diagnostics and audits.

## Repository layout
.
├── make_malicious_prompts_cn_compose_v2.py   # Main CLI for composing datasets
├── dsl_core.py                               # DSL for positive sampling + coverage/dedup logic
├── dedupe_core.py                            # Shared de-dup utilities (SimHash + MinHash + trigram cosine)
└── requirements.txt                          # Python dependencies


- make_malicious_prompts_cn_compose_v2.py orchestrates the entire pipeline: build targets, compose positives/negatives, global & group‑aware de‑dup, split, parity checks, and reporting.

- dsl_core.py implements the DSL: renderers, invariants, coverage budgeting, local/global de‑dup per mechanism/style, and optional YAML compilation.

- dedupe_core.py is a reusable de‑dup engine combining several signals with optional FAISS/Annoy acceleration.

- requirements.txt lists runtime deps (e.g., datasketch, annoy, faiss-cpu [non‑Windows], scikit-learn, regex, pyyaml, opencc).

## Installation

Use Python 3.9+.

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt


## Notes:

faiss-cpu is pinned to non‑Windows platforms; on Windows the code falls back to pure‑Python/Annoy paths automatically.

Some audits use scikit-learn (TF‑IDF & cosine) and run after the dataset is written.

## Quick start
1) Classic pipeline (no DSL)

Generates positives and hard negatives into one JSONL, writing an audit file with full metadata and a stats file for quality metrics.

python make_malicious_prompts_cn_compose_v2.py \
  --out data.jsonl --n 6000 --seed 42 \
  --neg_ratio 0.15 --family_max_frac 0.08 \
  --artifact_rate 0.25 --min_cjk_share_auto


The example above is included in the script’s header.

2) DSL‑driven positives

Use the DSL to control strategy × channel × carrier × delivery × evidence and style axes (speech_family/region/register/industry/persona), with coverage budgets and local+global de‑dup built in.

python make_malicious_prompts_cn_compose_v2.py \
  --out data.jsonl --n 4000 --neg_ratio 0.20 \
  --use_dsl --dsl_file dsl.yaml --dsl_n 3000 \
  --coverage_min_per_combo 2 --coverage_min_per_cfam 1


- The CLI loads and compiles YAML via compile_from_yaml if provided.

- DSL batch generation calls generate_batch(n, seed, pin, coverage_axes, ...) and returns samples + coverage diagnostics; the driver enforces a renderer‑missing error threshold for reliability.

## What gets written

For an output path like data.jsonl, the pipeline writes:

1. data.jsonl — training whitelist: each line only has {"text": ..., "label": ...} to minimize leakage. This is enforced and asserted in code.

2. data_audit.jsonl — full rows with diagnostics/metadata for reproducibility (e.g., intent, carrier, delivery, family, length buckets, CJK ratio bins, effect/diagnostics when available).

3. data_stats.json — post‑hoc semantic neighbor audit (char 3‑5 TF‑IDF cosine p95/p99) appended/created after the dataset is written; later steps also add anchor‑free/no‑keyword rates, parity summaries, etc.

## Data pipeline (high level)

1. Build target pool (benign task prompts) from HF datasets or local JSON, optionally streaming and revision‑pinned; configurable workers/cache.

- CLI knobs: --targets_json, --hf_revision, --target_streaming, --target_workers, --hf_cache_dir.

2. Compose positives

- Classic path: sampled euphemistic/indirect families with controlled artifact rates and CJK share; optional code‑switching and stacked carriers. Selection is managed by AttackSelector, which applies semantic de‑dup and coverage quotas while auditing.

- DSL path: generate_batch iteratively samples specs (strategy/channel/carrier/delivery/evidence), renders, validates invariants, enforces min coverage per combo and per evidence‑family pair (carrier×ev_family), and filters near‑duplicates with local (combo/mech/style) and global dedupers.

3. Compose negatives

- Plain negatives: normative texts with “hot words” but not injections.

- Topic‑shift negatives: task‑shifted benigns to reduce superficial cues.

- Real‑world negatives: adapters for WildChat, WildJailbreak, JBB behaviors, PKU SafeRLHF, optional PINT (or local JSON).

4. Merge & normalize; Global de‑dup (on anchor‑stripped text); optional greedy semantic de‑dup; then pair‑aware de‑dup and pair‑aware split so paired examples never leak across splits.

5. Post‑split code‑mix seeding for dev/test (mitigate distribution shift from CJK fixes).

6. Effect gate & diagnostics (optional) — if --use_model_eval and --effect_gate_for set, positives in dev/test (or all splits) must pass an effect check.

7. Coverage & parity checks — hard thresholds for minimal coverage per (intent, carrier, delivery), split‑wise coverage preservation, anchor‑free share and stealth/artifact quotas for positives, and tight symmetry of artifacts/masks/disclaimers between classes. Failing raises SystemExit.

8. Write train & audit files → run audits (semantic neighbors, leakage stress), print coverage matrices, layer counts, by‑intent effect pass rates, and other stats.

## The de‑duplication engine (how it works)

All de‑dup routes call a single library: dedupe_core.Deduper, which evaluates a record with multiple independent signals and rejects it if any signal says “near duplicate”.

# Signals

1. SimHash: Hamming distance threshold on a 64‑bit fingerprint (FAISS binary index if available; else BK‑tree fallback).

2. MinHash‑LSH over character shingles (exact Jaccard check on the candidate set; datasketch if present, otherwise a built‑in LSH).

3. Hashed trigram vector cosine (1024‑dim sparse count vector normalized to L2; FAISS inner‑product / Annoy angular or a fast matrix multiply fallback).

# Defaults (can be adjusted where the deduper is constructed):

- sim_bits=64, sim_thresh=1 (Hamming distance), k=5 shingle, n_hash=64, bands=16, jaccard_thresh=0.90, vec_dim=1024, cosine_thresh=0.92.

- The class supports external embeddings via Deduper.set_external_embedder(fn, cos_thresh); the driver will also reject on an external cosine ≥ threshold.

# Prepare → Check → Add flow

- prepare(text) builds a DedupeRecord with normalized text, simhash (and bytes for FAISS), shingles, minhash sig, and vector.

- check_record(record) returns (ok_bool, reason) with reasons like "simhash", "jaccard", "cosine", or "external".

- add_record(record) updates binary/dense indices or BK‑tree/Annoy/matrix cache.

The DSL wraps three local dedupers (by combo / by mechanism / by style) plus one global deduper with stricter thresholds to suppress both micro‑duplicates and structure‑preserving rephrases across styles.

## Minimal example

from dedupe_core import Deduper
d = Deduper(sim_thresh=1, jaccard_thresh=0.90, vec_dim=1024, cosine_thresh=0.92)
ok, reason, rec = d.probe("一段文本…")
if ok:
    d.add_record(rec)

## The DSL at a glance

The DSL lets you pin or sample along these axes and enforce coverage:

- Strategy, Channel, Carrier, Delivery, Evidence (with an evidence‑family notion for quotas);

- Style axes: speech_family, region, register, industry, persona (used for fair coverage).

# Generation loop (inside generate_batch)

Sample a spec → render carrier → wrap into delivery → obfuscate anchors in anchor‑free mode → validate invariants → dedupe (combo/mechanism/style/global) → accept. Returns (samples, coverage).

# Coverage & fairness

- min_per_combo on the chosen coverage_axes and min_per_cfam (carrier×evidence‑family) are enforced; combos can be temporarily blocked if they repeatedly fail dedupe/gates.

- CJK share, renderer‑missing rate, soft‑evidence usage, and anchor‑free hits are tracked in the coverage object; the driver hard‑fails if renderer‑missing rate is too high (default threshold 5%).

# De‑dup thresholds inside DSL

Local near‑dup similarity (~0.88–0.93 depending on signal) converted to SimHash Hamming, plus local vec‑cosine (~0.93), plus n‑gram Jaccard (~0.88); global sim thresholds are stricter.

# From DSL sample to training row

_dsl_to_row fills standard fields (task/len/len_bucket/latin_over_cjk(_bin)/family/delivery/mechanism/…) and normalizes channel→delivery, adding style buckets and soft‑evidence buckets as available.

# YAML compilation

The driver accepts --dsl_file and compiles YAML via compile_from_yaml (requires pyyaml).

- Example YAML (minimal)

strategy: openpi_hint
channel: in_context
carrier: html_comment
delivery: direct
evidence: [append, upstream]
constraints:
  anchor_free: true
style:
  speech_family: euphemism
  region: cn_mainland

## CLI overview (selected flags)

Run -h for the full list. Important groups:

# Core I/O & scale

--out, --n, --seed — output path, total rows, seed.

# CJK / language

--min_cjk_share (default 0.60, strict per row), --min_cjk_share_auto (randomized bins ≥ 0.60).

# Negatives

--neg_ratio, --plain_neg_ratio, --family_max_frac (cap per family).

# Targets

--targets_json, --hf_revision, --target_streaming, --target_workers, --hf_cache_dir.

# De‑dup (global)

--simhash_bits (64), --simhash_thresh (Hamming ≤1), --shingle_k (5), --minhash_n (64), --minhash_bands (16), --jaccard_thresh (0.90).

# Style & artifacts

--artifact_rate, --code_switch_prob, --carrier_mix_prob, --stack_prob, --sem_weight, --base_weight, --artifact_weight, --disc_rate.

# DSL

--use_dsl, --dsl_file, --dsl_n, --coverage_min_per_combo, --coverage_min_per_cfam.

# Evaluation gates

--use_model_eval, --effect_gate_for {none,devtest,all}: gate positives by effect success on selected splits.

# Splitting / leakage

Pair‑aware de‑dup + split, optional leave‑one‑out protocol by carrier|channel|style.

## Quality, audits & parity

After writing files, the script prints and stores:

- Semantic neighbor audit: char 3‑5 TF‑IDF cosine p95/p99 across all samples.

- Coverage matrices and carrier‑layer histograms for pos/neg.

- Parity checks: artifacts/masks/disclaimers must be near‑symmetric across classes (tight tolerance). Failure aborts.

- Anchor‑free share and stealth/artifact quotas enforced for positives.

## Reproducibility tips

- Set --seed, pin HF datasets with --hf_revision, and keep requirements.txt as recorded.

- The audit JSONL carries full per‑row metadata for downstream verification; the train JSONL carries only text,label.

## Extending the system
# Plug your own embeddings into de‑dup

Provide a function that maps normalized text → dense vector, then set it once:

from dedupe_core import Deduper
Deduper.set_external_embedder(my_embed_fn, cos_thresh=0.90)


The engine will additionally reject when external cosine ≥ threshold.

# Add new carriers/deliveries/evidence in DSL

Define a new renderer/delivery, register it in DSL, and include it in your YAML/spec. Coverage logic and local/global dedupe automatically apply when you pass through generate_batch.

# Troubleshooting

- FAISS not found / Windows — The engine falls back to a pure‑Python BK‑tree for SimHash and to Annoy/matrix ops for trigram cosine; this is expected.

- Renderer‑missing rate too high — DSL driver will hard‑fail if too many specs degrade to natural text; inspect coverage printouts and adjust your DSL pin/axes.

- Parity/coverage failures — Tight constraints abort with descriptive messages; reduce --coverage_min_per_combo, adjust artifact/mask/disclaimer weights, or revisit family caps.

- Effect gating — Requires model‑eval plumbing; when --use_model_eval and --effect_gate_for are set, positives must pass the effect check on the selected splits.

- Only text,label in train JSONL — This is intentional to avoid leakage; use the accompanying audit JSONL for analysis.

## Minimal code examples

# Use the de‑dup engine standalone

from dedupe_core import Deduper
texts = ["……一段注入样本……", "……另一个样本……"]
d = Deduper(sim_thresh=1, jaccard_thresh=0.90, vec_dim=1024, cosine_thresh=0.92)
kept = []
for t in texts:
    ok, reason, rec = d.probe(t)
    if ok:
        d.add_record(rec)
        kept.append(t)
    else:
        print("dropped due to:", reason)


# Generate with DSL (programmatic)

from dsl_core import generate_batch, compile_from_yaml
pin = compile_from_yaml(open("dsl.yaml", "r", encoding="utf-8").read())
samples, coverage = generate_batch(
    n=1000, seed=42, pin=pin,
    coverage_axes=("strategy","channel","carrier","delivery","ev_family","speech_family")
)


This returns fully rendered positives; you can integrate them into your own pipelines.

## Safety & intended use

The generator is designed for research on detection/robustness of prompt injection and includes strong negatives, parity controls, deduplication, and audits to avoid shortcut learning. Train files include only text,label; sensitive diagnostics and provenance remain in the audit artifacts.

## Appendix: Important defaults & thresholds (summary)

- Global de‑dup CLI defaults: simhash_bits=64, simhash_thresh=1, shingle_k=5, minhash_n=64, minhash_bands=16, jaccard_thresh=0.90.

- DSL local/global dedup: local sim ≈0.88/vec≈0.93/ngram≈0.88; global sim/ngram thresholds slightly tighter; vec disabled for global.

- CJK share: hard per‑row min (--min_cjk_share ≥ 0.60) with optional auto‑binning.

- Quotas: anchor‑free positives ≥ 50%; stealth‑high positives ≥ 40%; positive artifact share ≤ 30%; parity checks on artifacts/masks/disclaimers (tight tolerance).

## License

Not specified in the code. Please add a LICENSE file if you plan to release.

## Acknowledgements (dependencies)

This project relies on:

- datasketch for MinHash/LSH (optional), annoy for angular NN, faiss-cpu for dense/binary indices (optional), scikit-learn for audits, regex, pyyaml, datasets, opencc, numpy, numba, simhash, xxhash.



中文提示注入数据集生成器（v2）

一站式中文提示注入样本（正样本）与强硬负例数据生成器，具备：

语义化、多信号去重（SimHash + MinHash-LSH + 哈希三元组余弦；可选 FAISS/Annoy 加速）。

领域专用 DSL：可对策略/载体/交付/证据进行采样并强制覆盖与不变量约束，另支持可选的 YAML 编译。

伪迹控制、CJK（中文字符）占比控制、覆盖配额、对称性（parity）检查，以及成对感知的 train/dev/test 划分以降低泄漏。

目标：强调意图层面的注入语义（不只依赖词面锚点），扩展良性目标域，并产出可验证的数据集诊断与审计结果。

仓库结构
.
├── make_malicious_prompts_cn_compose_v2.py   # 组成数据集的主命令行入口
├── dsl_core.py                               # 用于正样本采样 + 覆盖/去重逻辑的 DSL
├── dedupe_core.py                            # 共享去重工具（SimHash + MinHash + 三元组余弦）
└── requirements.txt                          # Python 依赖


make_malicious_prompts_cn_compose_v2.py：编排整条流水线——构建目标池、生成正/负样本、全局与分组感知去重、数据集划分、对称性检查与报告。

dsl_core.py：实现 DSL——包含渲染器、不变量、覆盖预算、按机制/风格的本地与全局去重、可选 YAML 编译。

dedupe_core.py：可复用的去重引擎，将多种信号组合，并支持可选的 FAISS/Annoy 加速。

requirements.txt：运行依赖（如 datasketch、annoy、faiss-cpu［非 Windows］、scikit-learn、regex、pyyaml、opencc）。

安装

使用 Python 3.9+。

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt


说明：

faiss-cpu 仅在非 Windows 平台固定安装；在 Windows 上代码会自动回退到纯 Python/Annoy 路径。

部分审计使用 scikit-learn（TF-IDF & 余弦相似度），在数据集写出后执行。

快速开始
1）经典流水线（不使用 DSL）

在同一个 JSONL 中生成正样本与硬负例，同时写出带完整元数据的审计文件与用于质量指标的统计文件。

python make_malicious_prompts_cn_compose_v2.py \
  --out data.jsonl --n 6000 --seed 42 \
  --neg_ratio 0.15 --family_max_frac 0.08 \
  --artifact_rate 0.25 --min_cjk_share_auto


上面这个示例已包含在脚本头注释中。

2）使用 DSL 驱动的正样本

使用 DSL 控制策略 × 通道 × 载体 × 交付 × 证据与风格轴（speech_family/region/register/industry/persona），内置覆盖预算与本地+全局去重。

python make_malicious_prompts_cn_compose_v2.py \
  --out data.jsonl --n 4000 --neg_ratio 0.20 \
  --use_dsl --dsl_file dsl.yaml --dsl_n 3000 \
  --coverage_min_per_combo 2 --coverage_min_per_cfam 1


若提供 --dsl_file，CLI 会通过 compile_from_yaml 加载并编译 YAML。

DSL 批量生成通过 generate_batch(n, seed, pin, coverage_axes, ...) 执行，返回样本与覆盖诊断；驱动器会设置“渲染器缺失”错误阈值，提升稳健性。

产出文件

以 data.jsonl 为例，流水线将写出：

data.jsonl —— 训练白名单：每行仅包含 {"text": ..., "label": ...}，以最小化泄漏。代码中显式强制/断言这一点。

data_audit.jsonl —— 完整行，包含诊断/元数据，便于复现（如 intent、carrier、delivery、family、长度分桶、CJK 占比分桶、效果/诊断等）。

data_stats.json —— 事后“语义近邻审计”（字符 3-5 元 TF-IDF 余弦 p95/p99），在数据集写出后追加/创建；后续步骤还会补充“无锚/无关键词占比”、对称性汇总等。

数据流水线（高级概览）

构建目标池（良性任务提示）：来自 HF 数据集或本地 JSON；可选流式与版本固定；可配置并发/缓存。
CLI 选项：--targets_json、--hf_revision、--target_streaming、--target_workers、--hf_cache_dir。

生成正样本

经典路径：按设定的伪迹率与 CJK 占比采样委婉/间接家族；可选中英混写（code-switch）与载体叠加。选择由 AttackSelector 管理，执行语义去重与覆盖配额并记录审计。

DSL 路径：generate_batch 迭代采样规格（strategy/channel/carrier/delivery/evidence），渲染、校验不变量，对组合与“载体×证据族”设定最小覆盖，并通过本地（组合/机制/风格）与全局去重过滤近重复。

生成负样本

普通负例：包含“热词”但非注入的规范文本。

主题迁移负例：进行任务/主题切换以降低表面线索。

真实世界负例：适配 WildChat、WildJailbreak、JBB behaviors、PKU SafeRLHF，可选 PINT（或本地 JSON）。

合并与归一化；全局去重（对剥离锚点后的文本进行）；可选贪心语义去重；随后执行成对感知去重与成对感知划分，保证成对样本不跨分割泄漏。

划分后代码混写种子（dev/test）：缓解因 CJK 修复导致的分布漂移。

效果门控与诊断（可选）：若设置 --use_model_eval 与 --effect_gate_for，则 dev/test（或所有）分割中的正样本需通过效果检查。

覆盖与对称性检查：对（intent, carrier, delivery）设定最小覆盖阈值；保持按分割的覆盖；对正样本约束“无锚占比/隐蔽性/伪迹配额”；并要求正负类在伪迹/遮罩/免责声明等方面紧密对称。未通过将直接退出（SystemExit）。

写训练与审计文件 → 运行审计（语义近邻、泄漏压力），打印覆盖矩阵、层级计数、按意图的效果通过率及其他统计。

去重引擎（工作原理）

所有去重路径都调用同一个库：dedupe_core.Deduper。它对单条记录进行多种独立信号评估，只要任一信号判定为“近重复”，即予以拒绝。

信号：

SimHash：对 64 位指纹的汉明距离阈值（若可用则使用 FAISS 二进制索引；否则回退 BK-tree）。

字符 shingle 的 MinHash-LSH（候选集上再做精确 Jaccard；若装有 datasketch 则用之，否则使用内置 LSH）。

哈希三元组向量余弦（1024 维稀疏计数向量 L2 归一化；FAISS 内积/Annoy 夹角或快速矩阵乘回退）。

默认值（可在构造去重器时调整）：
sim_bits=64, sim_thresh=1（汉明距离）, k=5（shingle）, n_hash=64, bands=16, jaccard_thresh=0.90, vec_dim=1024, cosine_thresh=0.92。
类还支持外部嵌入：Deduper.set_external_embedder(fn, cos_thresh)；若外部余弦 ≥ 阈值，同样拒绝。

流程：准备 → 检查 → 添加

prepare(text)：构建 DedupeRecord（含规范化文本、simhash/FAISS 字节、shingles、minhash 签名、向量）。

check_record(record)：返回 (ok_bool, reason)，reason 可能为 "simhash"、"jaccard"、"cosine"、"external"。

add_record(record)：更新二进制/稠密索引或 BK-tree/Annoy/矩阵缓存。

DSL 的去重包装：在组合/机制/风格三个维度各自设一本地去重器，外加一个更严格的全局去重器，用以压制微重复与保持结构不变的重述（跨风格）。

最简示例：

from dedupe_core import Deduper
d = Deduper(sim_thresh=1, jaccard_thresh=0.90, vec_dim=1024, cosine_thresh=0.92)
ok, reason, rec = d.probe("一段文本…")
if ok:
    d.add_record(rec)

DSL 概览

DSL 允许你在以下轴上固定或采样并施加覆盖约束：

Strategy、Channel、Carrier、Delivery、Evidence（证据族的概念用于配额）

风格轴：speech_family、region、register、industry、persona（用于公平覆盖）

生成循环（generate_batch 内）
采样规格 → 渲染载体 → 交付包装 → 在 anchor-free 模式下做锚点隐匿 → 校验不变量 → 去重（组合/机制/风格/全局）→ 接受。返回（samples, coverage）。

覆盖与公平
对所选 coverage_axes 应用 min_per_combo，并对 min_per_cfam（载体 × 证据族）做配额；若某些组合多次未通过去重/门控，可暂时阻塞。
在覆盖对象中跟踪 CJK 占比、渲染器缺失率、软证据使用、无锚命中；若渲染器缺失率过高（默认 5% 阈值），驱动器将硬失败。

DSL 内部去重阈值
本地近重复相似度约 0.88–0.93（视信号而定），对应 SimHash 汉明阈值；另外本地向量余弦约 0.93，n-gram Jaccard 约 0.88；全局阈值更严格。

从 DSL 样本到训练行
_dsl_to_row 填入标准字段（task/len/len_bucket/latin_over_cjk(_bin)/family/delivery/mechanism/…），并规范 channel→delivery，按需增加风格分桶与软证据分桶。

YAML 编译
CLI 接受 --dsl_file 并通过 compile_from_yaml 编译（需要 pyyaml）。

最小 YAML 示例：

strategy: openpi_hint
channel: in_context
carrier: html_comment
delivery: direct
evidence: [append, upstream]
constraints:
  anchor_free: true
style:
  speech_family: euphemism
  region: cn_mainland

CLI 概览（部分常用参数）

（运行 -h 查看完整列表）

核心 I/O 与规模

--out, --n, --seed：输出路径、样本总数、随机种子。

CJK / 语言

--min_cjk_share（默认 0.60，逐行严格），--min_cjk_share_auto（随机分桶 ≥ 0.60）。

负样本

--neg_ratio, --plain_neg_ratio, --family_max_frac（限制单家族上限）。

目标池

--targets_json, --hf_revision, --target_streaming, --target_workers, --hf_cache_dir。

全局去重

--simhash_bits(64), --simhash_thresh(≤1 汉明), --shingle_k(5), --minhash_n(64), --minhash_bands(16), --jaccard_thresh(0.90)。

风格 & 伪迹

--artifact_rate, --code_switch_prob, --carrier_mix_prob, --stack_prob, --sem_weight, --base_weight, --artifact_weight, --disc_rate。

DSL

--use_dsl, --dsl_file, --dsl_n, --coverage_min_per_combo, --coverage_min_per_cfam。

效果门控

--use_model_eval, --effect_gate_for {none,devtest,all}：在指定分割上以效果成功与否门控正样本。

划分 / 泄漏

成对感知去重 + 划分；可选按 carrier|channel|style 的留一法协议。

质量、审计与对称性

写出文件后，脚本会打印并存储：

语义近邻审计：全体样本的字符 3-5 元 TF-IDF 余弦 p95/p99。

覆盖矩阵与载体层级直方图（正/负）。

对称性检查：正负类在伪迹/遮罩/免责声明方面需近似对称（严格容差）；失败则中止。

对正样本强制 无锚占比 与 隐蔽性/伪迹配额。

可复现实践

设置 --seed，用 --hf_revision 固定 HF 数据集版本，并保持 requirements.txt 不变。

训练用 JSONL 仅含 text,label；完整元数据留在审计 JSONL 里，便于后续验证。

扩展系统
将你的嵌入接入去重

提供一个将规范化文本映射到稠密向量的函数，然后一次性设置：

from dedupe_core import Deduper
Deduper.set_external_embedder(my_embed_fn, cos_thresh=0.90)


当外部余弦相似度 ≥ 阈值时，去重引擎也会拒绝。

在 DSL 中新增载体/交付/证据

定义新的渲染器/交付方式，在 DSL 中注册，并在 YAML/spec 中引用。通过 generate_batch 流水线时，会自动应用覆盖逻辑与本地/全局去重。

故障排查

找不到 FAISS / Windows 平台：引擎会回退到 SimHash 的纯 Python BK-tree，以及三元组余弦的 Annoy/矩阵回退，这是预期行为。

渲染器缺失率过高：DSL 驱动会硬失败；查看覆盖打印并调整 DSL 的 pin/轴组合。

对称性/覆盖失败：严格的约束会中止；可降低 --coverage_min_per_combo，或调整伪迹/遮罩/免责声明权重，或重新审视家族上限。

效果门控：需要模型评测支撑；当设置 --use_model_eval 与 --effect_gate_for 时，正样本必须在选定分割上通过效果检查。

训练 JSONL 只有 text,label：为避免泄漏是刻意设计；分析请使用随附的审计 JSONL。

最小代码示例

单独使用去重引擎

from dedupe_core import Deduper
texts = ["……一段注入样本……", "……另一个样本……"]
d = Deduper(sim_thresh=1, jaccard_thresh=0.90, vec_dim=1024, cosine_thresh=0.92)
kept = []
for t in texts:
    ok, reason, rec = d.probe(t)
    if ok:
        d.add_record(rec)
        kept.append(t)
    else:
        print("dropped due to:", reason)


以编程方式使用 DSL 生成

from dsl_core import generate_batch, compile_from_yaml
pin = compile_from_yaml(open("dsl.yaml", "r", encoding="utf-8").read())
samples, coverage = generate_batch(
    n=1000, seed=42, pin=pin,
    coverage_axes=("strategy","channel","carrier","delivery","ev_family","speech_family")
)


函数会返回完全渲染的正样本；你可以将其整合到自己的流水线中。

安全性与用途

该生成器用于提示注入检测/鲁棒性研究；包含强硬负例、对称性控制、去重与审计，避免“走捷径式”学习。训练文件仅含 text,label；更敏感的诊断与来源信息保存在审计产物中。

附录：重要默认值与阈值（摘要）

全局去重 CLI 默认：simhash_bits=64, simhash_thresh=1, shingle_k=5, minhash_n=64, minhash_bands=16, jaccard_thresh=0.90。

DSL 本地/全局去重：本地 sim≈0.88 / vec≈0.93 / ngram≈0.88；全局 sim/ngram 更严格；全局默认不开启向量。

CJK 占比：逐行硬阈值（--min_cjk_share ≥ 0.60），可选自动分桶。

配额：无锚正样本 ≥ 50%；高隐蔽性正样本 ≥ 40%；正样本伪迹占比 ≤ 30%；对伪迹/遮罩/免责声明进行严格对称性检查。

许可证

代码未标注具体许可证。如计划发布，请添加 LICENSE 文件。

致谢（依赖）

本项目使用：
datasketch（MinHash/LSH，可选）、annoy（角度近邻）、faiss-cpu（稠密/二进制索引，可选）、scikit-learn（审计）、regex、pyyaml、datasets、opencc、numpy、numba、simhash、xxhash。