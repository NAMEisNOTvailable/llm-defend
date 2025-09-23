# Chinese Prompt-Injection Dataset Composer (v2)

This repository bundles the scripts used to assemble Chinese prompt-injection datasets with strong hard negatives, extensive auditing, and layered de-duplication.

## Overview
- `make_malicious_prompts_cn_compose_v2.py` - end-to-end CLI for dataset generation (positives, negatives, auditing, splitting, reporting).
- `dsl_core.py` - domain-specific language for intent-level attack rendering, invariants, coverage accounting, and local/global de-dup helpers.
- `dedupe_core.py` - reusable SimHash + MinHash-LSH + hashed n-gram cosine deduper with optional ANN acceleration and extensibility hooks.

### Repository Layout
```
.
|-- make_malicious_prompts_cn_compose_v2.py
|-- dsl_core.py
|-- dedupe_core.py
|-- requirements.txt
|-- ... (datasets, JSON artefacts)
```

## Capability Detection & Optional Modules
The CLI prints a capability banner at startup so you immediately know which optional accelerators or DSL helpers are active on the current machine. Example:

```
[capability] optional module probe:
  - simhash: disabled (ModuleNotFoundError: No module named 'simhash')
  - datasketch: disabled (ModuleNotFoundError: No module named 'datasketch')
  - faiss: disabled (gpu ModuleNotFoundError: No module named 'faiss'; cpu ModuleNotFoundError: No module named 'faiss_cpu')
  - annoy: disabled (ModuleNotFoundError: No module named 'annoy')
  - dsl_core.core: enabled (native)
  - dsl_core._h64: enabled (shared)
  - dedupe_core: enabled (native)
  - dsl_core.simhash_fast: enabled (native)
  - dsl_core.shingle_fast: enabled (native)
  - dsl_core.reward_filters: enabled (native)
  - dsl_core.sandbox_suite: enabled (native)
```

| Capability key | Module(s) | Purpose when enabled | Behaviour when missing |
| --- | --- | --- | --- |
| `simhash` | `simhash` | Native 64-bit SimHash for faster dedupe. | Falls back to NumPy implementation (slower but identical semantics).
| `datasketch` | `datasketch` | MinHash + LSH buckets for Jaccard filtering. | Uses tuple-based fallback; MinHash-LSH is skipped.
| `faiss` | `faiss` or `faiss_cpu` | Approximate nearest neighbour for dense vectors. | Vector search disabled; cosine gating uses pure Python.
| `annoy` | `annoy` | Alternative ANN for dedupe and audits. | Skipped; relies on deterministic hashing paths.
| `dsl_core.*` | bundled | DSL renderers, invariants, sandbox heuristics. | Each sub-capability is replaced with conservative no-op stubs.
| `dedupe_core` | bundled | Core SimHash/MinHash/trigram deduper. | Required; CLI aborts if import fails.

`dedupe_core.py` also uses `numba` and `xxhash` when available for faster sketches, and the auditing pass in `make_malicious_prompts_cn_compose_v2.py` optionally imports `scikit-learn` (TF-IDF cosine). Missing modules simply slow things down; functionality remains intact.

## Installation
1. Use Python 3.9 or newer.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # POSIX:  source .venv/bin/activate
   pip install -U pip wheel
   pip install -r requirements.txt
   ```
3. Optional accelerators:
   ```bash
   pip install simhash datasketch annoy scikit-learn xxhash numba
   # On Linux/macOS you can also: pip install faiss-cpu
   ```
   Windows does not provide official `faiss-cpu`; the CLI falls back automatically.

## Running the Generator
### Baseline run (legacy templates)
```bash
python make_malicious_prompts_cn_compose_v2.py \
  --out data.jsonl \
  --n 6000 --seed 42 \
  --neg_ratio 0.15 \
  --family_max_frac 0.08 \
  --artifact_rate 0.25 \
  --min_cjk_share_auto
```
This produces positives and hard negatives in `data.jsonl`, plus audit and stats sidecars.

### DSL-driven positives
```bash
python make_malicious_prompts_cn_compose_v2.py \
  --out data.jsonl \
  --n 4000 --neg_ratio 0.20 \
  --use_dsl \
  --coverage_min_per_combo 2 \
  --structural_p 0.55 \
  --anchor_free_p 0.65
```
The DSL path samples strategy/channel/carrier/delivery combinations, enforces anchor-free coverage, and adapts CJK share or structural frequencies using the provided knobs. `dsl_core.generate_batch` accepts an optional `pin` dictionary, so advanced users can set prior weights or freeze specific mechanisms.

Key CLI toggles to be aware of:
- `--plain_neg_ratio`, `--topic_neg_ratio`, `--wild_neg_ratio` control the negative mix.
- `--structural_p`, `--alias_p_cn`, `--anchor_free_p` adjust carrier/style biases across positives.
- `--coverage_min_per_combo`, `--coverage_min_per_cfam` add hard minimums for DSL coverage audits.
- `--simhash_bits`, `--simhash_thresh`, `--minhash_n`, `--minhash_bands`, `--jaccard_thresh` tune dedupe strength.

## Generated Artefacts
Given `--out data.jsonl`, the run creates:
- `data.jsonl` - minimal training view containing `{ "text", "label" }` pairs only.
- `data_audit.jsonl` - full diagnostics (intent, carrier, delivery, coverage bins, anchor/free flags, symmetry stats, effect metadata).
- `data_stats.json` - aggregated audits (semantic neighbour percentiles, anchor-free ratios, parity summaries).

## Pipeline Highlights
1. Target pool construction mixes HF datasets and local JSON with optional revision pinning and worker pools.
2. Positive synthesis uses either legacy templates or DSL renders emphasising intent-level injections, euphemistic anchors, structural wrappers, and CJK control.
3. Negative synthesis spans normative documents with "hot words", topic-shifted prompts, and adapters for corpora such as WildChat, JailbreakBench, and PKU SafeRLHF.
4. De-duplication layers SimHash, MinHash, hashed trigram cosine, optional ANN search, and pair-aware checks to avoid leakage between splits.
5. Diagnostics enforce artifact/mask/disclaimer parity, validate coverage buckets, run semantic neighbour audits, and guard anchor-free thresholds.

## Using the Python APIs
### Deduplication
```python
from dedupe_core import Deduper

texts = ["prompt A", "prompt B"]
d = Deduper(sim_bits=64, sim_thresh=1, k=5, n_hash=64,
            bands=16, jaccard_thresh=0.90)
kept = []
for text in texts:
    ok, reason, record = d.probe(text)
    if ok:
        d.add_record(record)
        kept.append(text)
    else:
        print("dropped due to", reason)

# Optional: plug your own encoder for semantic gating
# Deduper.set_external_embedder(embed_fn, cos_thresh=0.90)
```

### DSL Sampling
```python
from dsl_core import generate_batch

samples, coverage = generate_batch(
    n=1000,
    seed=42,
    pin={"min_cjk_share": 0.65, "structural_p": 0.55},
    coverage_axes=("strategy", "channel", "carrier", "delivery",
                   "ev_family", "speech_family", "anchor_free"),
    min_per_combo=2,
)
```
`generate_batch` returns detailed render metadata; downstream code can adapt it with `_dsl_to_row` (see the CLI) or custom transformers.

## Diagnostics & Troubleshooting
- Review the capability banner; missing modules explain slower runs or disabled accelerators.
- Windows users: `faiss` is typically unavailable. Install `annoy` if you need ANN acceleration.
- If DSL coverage fails (anchor-free ratio, renderer miss-rate), inspect the `[dsl-coverage]` payload printed to stdout and adjust `--coverage_min_per_*` or the `pin` overrides.
- `data_stats.json` is appended after the JSONL is written; re-running the script overwrites existing stats to keep reports in sync.

## License & Credits
This repository does not include a licence file; ensure downstream usage complies with internal policy. The project builds on open-source packages such as `datasets`, `regex`, `numpy`, `scikit-learn`, `datasketch`, `annoy`, `faiss`, `numba`, `simhash`, and `xxhash`.
