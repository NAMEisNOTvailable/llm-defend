# Chinese Prompt-Injection Dataset Composer (v2)

This repository assembles Chinese prompt-injection datasets with hard negatives, DSL-driven renderers, layered de-duplication, and now an automated pipeline for harvesting soft-evidence phrases from raw corpora. It consists of five cooperating modules:

| File | Purpose |
| --- | --- |
| `make_malicious_prompts_cn_compose_v2.py` | End-to-end CLI: loads corpora (HF datasets or local JSON), synthesises positives/negatives, dedupes, audits, and writes stats. Capability probes cover optional packages (datasets, simhash, faiss, annoy, datasketch, etc.) and print a startup banner. |
| `extract_and_bank_zh_phrases.py` | KeyBERT + BGE-based candidate miner for Chinese corpora; filters, dedupes, clusters, and feeds phrases back into the DSL soft-evidence banks (with optional manual overrides). |
| `dsl_core.py` | DSL that renders injection intents, enforces invariants, and exposes coverage, anchor-free/anchor-required heuristics, soft-evidence checks, and JSON/markdown inspectors. |
| `dedupe_core.py` | Reusable SimHash + MinHash-LSH + hashed trigram cosine deduper with optional FAISS/Annoy/Numba/xxhash acceleration. Defaults are tuned for Chinese paraphrases and style wrapping. |
| `micro_grammar.py` | Micro-grammar bank for soft-evidence phrases with deterministic seeding, adaptive slot permutations, DSL style wrapping, and helpers to refill & rebuild DSL banks safely. |

## Installation
Use **Python 3.9+ (3.10 recommended)** before creating the virtual environment.

1. Create a virtual environment.
2. Install the grouped dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` groups dependencies by feature set:
   - Base runtime: `numpy`, `regex`, `pyyaml`.
   - Optional data sources / DSL helpers: `datasets`, `opencc`, `orjson`.
   - Soft-evidence extraction + banking: `jieba`, `scikit-learn`, `keybert`, `sentence-transformers`, `torch`. Install this block when you run `extract_and_bank_zh_phrases.py`; comment the lines out if you only need the composer.
3. (Recommended) Install accelerators for dedupe/audit hot paths:
   ```bash
   pip install simhash datasketch annoy xxhash numba
   # On Linux/macOS you can also: pip install faiss-cpu
   ```
   These unlock faster SimHash/MinHash pipelines and ANN probes inside `dedupe_core`.

GPU users should match the `torch` wheel to their CUDA runtime (or stay on CPU by omitting `--cuda` when running the extractor).

## Capability Detection
`make_malicious_prompts_cn_compose_v2.py` prints a capability report at startup. Each optional dependency either enables a faster path (e.g., native SimHash, FAISS, Annoy) or activates DSL helpers (`dsl_core.core`, reward filters, sandbox suite). When a module is missing the script records the disabled capability and continues with conservative fallbacks.

## Soft Evidence Extraction
`extract_and_bank_zh_phrases.py` mines high-signal soft-evidence phrases from newline-delimited corpora (`--input`) and merges them into the DSL banks.

Pipeline highlights:
- Pre-warms and optionally hydrates micro-grammars so the DSL's soft-evidence buckets contain fresh prototypes before scoring.
- Uses KeyBERT (with `BAAI/bge-small-zh-v1.5` embeddings) to surface multi-token Chinese phrases while enforcing a CJK share floor.
- Keeps only action-like phrases that clear the `_softish_score` gate, then applies the shared `Deduper` for SimHash + MinHash + cosine filtering.
- Clusters surviving phrases with SentenceTransformer community detection and auto-routes each cluster to the highest priority soft-evidence kind (or a `--kind` override).
- Optionally re-banks every cluster member (`--bank_all`) and emits coverage probes via `dsl_core.probe_soft_coverage`.

Example run:
```bash
python extract_and_bank_zh_phrases.py \
  --input zh_lines.txt \
  --cluster_thr 0.85 \
  --bank_all \
  --cuda
```
The script prints a JSON summary (candidate counts, routing stats, coverage deltas) so you can capture the run in logs. It updates the in-memory DSL, making the newly banked phrases immediately available to the composer and auditors.

## Running the Composer
Typical CLI usage:
```bash
python make_malicious_prompts_cn_compose_v2.py \
  --out data.jsonl \
  --n 6000 --seed 42 \
  --neg_ratio 0.15 \
  --family_max_frac 0.08 \
  --artifact_rate 0.25 \
  --min_cjk_share_auto
```
Small-run guardrails:
- If `n` is smaller than the coverage budget, the CLI now auto-lowers `coverage_min_per_combo` to 1 and prints an `[auto-config]` notice instead of hard failing.
- After rendering, candidates pass a coherence gate (`is_coherent_cn`) and a `mechanism_truth_record` check; anchor-free specs get a light soft-hint backfill when they miss the soft-evidence floor.

Outputs:
- `data.jsonl` - minimal training rows (`text`, `label`).
- `data_audit.jsonl` - diagnostic metadata (intent, carrier, coverage, side-effects, anchor-free hits).
- `data_stats.json` - aggregated audits (symmetry, anchor-free ratios, semantic neighbour percentiles, leakage scores).

Key toggles:
- `--use_dsl`, `--coverage_min_per_combo`, `--structural_p`, `--anchor_free_p` control DSL rendering and coverage.
- `--soft_hint_rate`, `--alias_p_cn`, `--tail_mix_p` drive soft-evidence injection and style wrapping.
- `--simhash_bits`, `--simhash_thresh`, `--minhash_n`, `--minhash_bands`, `--jaccard_thresh` tune dedup thresholds.
- `--targets_json` lets you supply local corpora when HF datasets are unavailable; `_HAS_HF_DATASETS` controls which adapters run.

`micro_grammar.py` now:
- Uses permutation-based slot orders when none are provided (up to 6 slots) for more surface diversity.
- Seeds each grammar with a `blake2b`-derived integer so sampling is repeatable across runs and hosts.
- Dedupes per evidence bucket, keeps closures pinned to the live DSL lists, and neutralises tone joiners when strong phrases fire.
- Wraps candidates through `dsl_core.apply_style` when the DSL is available and falls back gracefully otherwise.
- Provides `attach_to_dsl_core`, `refill_bank`, and `rebuild_soft_check` so regenerated phrases update the DSL banks in-place while respecting category-specific semantic thresholds. The extractor reuses these hooks during prewarm and hydration.

## Deduplication
`dedupe_core.Deduper` combines SimHash, MinHash-LSH, hashed trigram cosine, and optional ANN search. Defaults (`sim_thresh=1`, `jaccard_thresh=0.90`, `cosine_thresh=0.92`) balance Chinese paraphrase coverage with aggressive duplicate removal; adjust if style wrapping is either too strict or too permissive.

Example:
```python
from dedupe_core import Deduper

texts = ["prompt A", "prompt B", "prompt A!!!"]
keeper = Deduper(jaccard_thresh=0.90, cosine_thresh=0.92)
rows = []
for txt in texts:
    ok, reason, record = keeper.probe(txt)
    if ok:
        keeper.add_record(record)
        rows.append(txt)
    else:
        print('dropped', reason)
```

## DSL Batch API
Use `dsl_core.generate_batch` to render intent-level injections programmatically:
```python
from dsl_core import generate_batch
samples, coverage = generate_batch(
    n=1000,
    seed=42,
    pin={"min_cjk_share": 0.65, "structural_p": 0.55},
    coverage_axes=("strategy","channel","carrier","delivery"),
    min_per_combo=2,
)
```
Coverage stats help you enforce anchor-free ratios, mechanism variety, and structural wrap requirements.

## Notes
- Capability logging and auditing utilities surface any fallback paths (regex failures, DSL gaps, dedupe skips).
- HF datasets are probed lazily; when unavailable, adapters fall back to local JSON sources.
- Tests are limited to compile-time checks; integrate additional CI or data validation as needed.
