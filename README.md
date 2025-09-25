# Chinese Prompt-Injection Dataset Composer (v2)

This repository assembles Chinese prompt-injection datasets with hard negatives, DSL-driven renderers, and layered de-duplication. It consists of four cooperating modules:

| File | Purpose |
| --- | --- |
| `make_malicious_prompts_cn_compose_v2.py` | End-to-end CLI: loads corpora (HF datasets or local JSON), synthesises positives/negatives, dedupes, audits, and writes stats. Capability probes cover optional packages (datasets, simhash, faiss, annoy, datasketch, etc.) and print a startup banner. |
| `dsl_core.py` | DSL that renders injection intents, enforces invariants, and exposes coverage, anchor-free/anchor-required heuristics, soft-evidence checks, and JSON/markdown inspectors. |
| `dedupe_core.py` | Reusable SimHash + MinHash-LSH + hashed trigram cosine deduper with optional FAISS/Annoy/Numba/xxhash acceleration. Defaults are tuned for Chinese paraphrases and style wrapping. |
| `micro_grammar.py` | Micro-grammar bank for soft-evidence phrases, with adaptive sampling, permutation-based slot ordering, style wrapping, and helpers to refill & rebuild DSL banks safely. |

## Installation
1. Use Python 3.9 or newer and create a virtualenv.
2. Install core dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This installs the minimal runtime (`numpy`, `regex`, `pyyaml`) plus optional helpers (`datasets`, `opencc`, `orjson`) if available. Missing optional packages degrade gracefully.
3. (Recommended) For faster dedupe/audits install the accelerators:
   ```bash
   pip install simhash datasketch annoy scikit-learn xxhash numba
   # On Linux/macOS you can also: pip install faiss-cpu
   ```

## Capability Detection
`make_malicious_prompts_cn_compose_v2.py` prints a capability report at startup. Each optional dependency either enables a faster path (e.g., native SimHash, FAISS, Annoy) or activates DSL helpers (`dsl_core.core`, reward filters, sandbox suite). When a module is missing the script records the disabled capability and continues with conservative fallbacks.

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
Key toggles:
- `--use_dsl`, `--coverage_min_per_combo`, `--structural_p`, `--anchor_free_p` control DSL rendering and coverage.
- `--soft_hint_rate`, `--alias_p_cn`, `--tail_mix_p` drive soft-evidence injection and style wrapping.
- `--simhash_bits`, `--simhash_thresh`, `--minhash_n`, `--minhash_bands`, `--jaccard_thresh` tune dedup thresholds.
- `--targets_json` lets you supply local corpora when HF datasets are unavailable; `_HAS_HF_DATASETS` controls which adapters run.

Outputs:
- `data.jsonl` ¨C minimal training rows (`text`, `label`).
- `data_audit.jsonl` ¨C diagnostic metadata (intent, carrier, coverage, side-effects, anchor-free hits).
- `data_stats.json` ¨C aggregated audits (symmetry, anchor-free ratios, semantic neighbour percentiles, leakage scores).

## Micro-Grammar Helpers
`micro_grammar.py` now:
- Uses permutation-based slot orders when none are provided (up to 6 slots) for more surface diversity.
- Adapts sampling caps/trials based on slot diversity, optional counts, and strong-value usage (see `expand_grammar` docstring for complexity bounds).
- Normalises duplicates with strong-tone safeguards and region-specific overlays.
- Provides `attach_to_dsl_core`, `refill_bank`, and `rebuild_soft_check` so regenerated phrases update the DSL banks and closure cache safely.

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
