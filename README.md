# LLM-Defend: Chinese Prompt-Injection Dataset Composer

This repository provides a modular pipeline for generating Chinese-heavy jailbreak and safety datasets. The tooling focuses on intent-level coverage, controlled lexical artifacts, and reproducible sampling so the resulting corpora are suitable for benchmarking LLM defenses.

## Repository layout

| Path | Description |
| --- | --- |
| `v3.py` | End-to-end CLI that composes Chinese prompt-injection samples (positives) plus hard negatives, runs audits, and writes dataset/statistics files. Invokes the refactored modules in `compose/`. |
| `compose/` | Package that houses the refactor of the legacy monolithic script. Submodules cover CLI parsing, attack families, leakage checks, balancing, RNG wiring, multiprocessing workers, serialization, and auditing utilities. |
| `dedupe/` & `dedupe_core.py` | Shared SimHash + MinHash-LSH + hashed trigram cosine deduplication logic with optional Annoy/FAISS/Numba accelerators. |
| `dsl_core/` | Domain specific language (DSL) for rendering intents, invariants, anchors, sandboxed execution, and soft evidence hooks. |
| `micro_grammar.py` | Deterministic soft-evidence grammar bank used to refresh DSL categories and provide anchor-free hints. |
| `extract_and_bank_zh_phrases.py` | SentenceTransformer-based miner that discovers high-signal Chinese phrases from corpora and feeds them back into the DSL soft-evidence banks. |
| `stable_random.py` | Deterministic RNG helpers shared across the composer and extractor. |
| `source/` | Bundled seed corpora and loaders (e.g., JailBench, MultiJail_zh, combined safety prompts) plus `load_dataset.py` for HuggingFace/local ingestion. |

## Installation

The toolchain targets **Python 3.9+ (3.10 recommended)**.

1. Create and activate a virtual environment.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   * Core runtime: `numpy`, `regex`, `orjson`, and `datasets` power the composer and DSL modules.
   * Optional dedupe accelerators: `simhash`, `datasketch`, `annoy`, `faiss-cpu`, `numba`, and `xxhash` unlock faster similarity checks in `dedupe/`.
   * Soft-evidence extraction: `jieba`, `sentence-transformers`, `scikit-learn`, `torch`, and `psutil` are needed only when running `extract_and_bank_zh_phrases.py` (install `jieba-fast` when wheels are available).

## Generating a dataset

Run the CLI directly via `v3.py`. It prints a capability report summarizing which optional modules were detected, then composes the dataset while enforcing CN-share, coverage, artifact, and audit policies.

```bash
python v3.py \
  --out data.jsonl \
  --n 6000 --seed 42 \
  --family_max_frac 0.08 \
  --artifact_rate 0.25 \
  --min_CN_share_auto
```

Key outputs:

* `data.jsonl` – prompt-injection attempts and hard negatives with `text`, `label`, and metadata fields.
* `data_audit.jsonl` – per-row diagnostics covering intents, carriers, delivery strategies, and audit reasons.
* `data_stats.json` – aggregated coverage, symmetry, leakage, and distribution reports.

The full set of arguments is defined in `compose/cli.py`; groups include I/O controls, CN-share policy, sampling strategy, dedupe thresholds, DSL toggles, and optional adversarial mutation/effects modules. Defaults are centralized in `compose/constants.py`.

### Coverage & balancing

The composer enforces coverage across intents, carriers, and delivery channels using weights and quotas defined in `compose/state.py`. Symmetry audits in `compose/symmetry.py` and leakage checks in `compose/leakage.py` guard against obvious artifacts. Hard negatives originate from curated sources (`compose/sources.py`) and can be extended with local JSON/JSONL via `--targets_json`.

### Deduplication

Deduping relies on `dedupe.core.Deduper`, which combines SimHash, MinHash-LSH, and cosine similarity over hashed trigrams. Optional FAISS and Annoy indices (configured in `dedupe/backends/`) accelerate near-duplicate search when the corresponding packages are installed. Defaults can be overridden through CLI flags (`--simhash_bits`, `--simhash_thresh`, `--minhash_n`, `--minhash_bands`, `--jaccard_thresh`, `--cosine_thresh`).

### DSL and soft evidence

When `--use_dsl` or related toggles are enabled, the pipeline loads `dsl_core` to render structured attack templates, apply anchor invariants, and inject soft evidence. `micro_grammar.py` maintains deterministic slot permutations and category-specific thresholds. For larger corpora, run the extractor to refresh the DSL banks:

```bash
python extract_and_bank_zh_phrases.py \
  --input source/zh_lines.txt \
  --cluster_thr 0.85 \
  --bank_all \
  --cuda
```

The extractor streams documents, mines multi-token Chinese phrases with `BAAI/bge-small-zh-v1.5` embeddings, dedupes candidates with the shared `Deduper`, clusters them, and updates the in-memory DSL before writing JSON summaries of coverage and cache hits.

## Reproducibility

`stable_random.py` and `compose/rng.py` centralize seed handling so repeated runs with the same configuration yield consistent samples. Thread affinities and BLAS environment variables are tuned in `v3.py` to keep CPU usage predictable. Soft evidence caches and audit sidecars live next to the requested output path; reruns reuse them when signatures match.

## Extending the pipeline

* Add new attack templates or carriers by extending the registries in `compose/attacks.py`, `compose/carriers.py`, and the delivery utilities in `compose/state.py` and `compose/payload.py`.
* Integrate new safety corpora through `compose/sources.py` or by placing datasets under `source/` and updating `load_dataset.py`.
* Customize audits by editing `compose/audit.py`, `compose/leakage.py`, or `compose/effects_eval.py`.

Each submodule is importable directly from the `compose` package (`from compose import attacks, audit, utils, ...`), enabling interactive experimentation and downstream tooling without running the CLI.

