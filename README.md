# LLM-Defend: Chinese Prompt-Injection Dataset Composer

> Toolkit for composing CN-heavy prompt-injection attacks, euphemistic payloads, and symmetry-matched hard negatives with deterministic dedupe and DSL-driven coverage.

## Goals & Highlights

- Intent-level sampling across attack families, carriers, and delivery strategies (grid-based `compose.state` and DSL specs) to minimize lexical artifacts.
- Shared dedupe stack (SimHash + MinHash-LSH + hashed trigram cosine + optional vector search) ensures reproducible filtering across positives/negatives.
- DSL + micro-grammar generators enforce CN-share, anchors/invariants, and conversational soft-evidence without leaking artifacts.
- Deterministic randomness via `stable_random.py` and `compose.rng` so reruns with the same config/seed reproduce identical corpora and audits.
- Extensible compose package: add attacks, carriers, payloads, or hard-negative sources without touching the CLI wrapper.

## Repository Layout

| Path | Description |
| --- | --- |
| `v3.py` | Multiprocessing CLI that wires together RNG, compose modules, DSL/micro-grammar refresh, dedupe, audits, and reporting. |
| `compose/` | Refactored pipeline package (CLI, capabilities, attacks, carriers, payloads, RNG, quota/balancing, audits, effects validation, workers, serialization). |
| `dedupe_core.py` | Compatibility shim that imports `dedupe.core` / `dedupe.index` and falls back to a minimal set-based deduper when optional deps are missing. |
| `dedupe/` | Full deduplication library (`core.py`, `index.py`, `annoy.py`, `faiss.py`) providing SimHash, MinHash-LSH, hashed trigram vectors, and optional ANN accelerators. |
| `dsl_core/` | Domain-specific language for attack intents: anchors, invariants, sandboxed renderers, soft-evidence banks, and generator utilities. |
| `micro_grammar.py` | Deterministic CN micro-grammar system (slots, regional overlays, style profiles) used to refresh DSL soft evidence and anchor-free hints. |
| `stable_random.py` | Stable RNG helpers (`stable_rng`, `derive_rng`, `RandomBinder`, context managers) consumed by compose, DSL, and grammar modules. |
| `extract_and_bank_zh_phrases.py` | SentenceTransformer-based miner that refreshes the soft-evidence/micro-grammar banks from large CN corpora. |
| `source/` | Bundled corpora plus loaders (`load_dataset.py`) for offline composition and local HuggingFace mirrors. |
| `requirements.txt`, `bank_snapshot_*.json` | Supporting utilities, dependency pins, and cached bank snapshots referenced by the pipeline. |

## Core Components

### `v3.py` - orchestrated CLI

- Bootstraps BLAS/thread envs (`_configure_thread_env`) and configures deterministic seeds via `stable_random.random_module_binding`.
- Parses all CLI groups through `compose.cli.build_arg_parser`, validates defaults (`validate_parser_defaults`), and prints capability probes from `compose.capabilities.emit_capability_report` so you know which optional features (DSL, micro grammar, dedupe backends, ANN libraries, HF datasets) are active.
- Instantiates RNG (`compose.rng.configure_compose_rng`), quotas (`compose.quota.set_quota_manager`), dedupe helpers, DSL runtime, and audit queues before spinning up `compose.workers`.
- Streams composed samples, enforces coverage (`compose.state`, `compose.balance`), runs audit/effects gates (`compose.audit`, `compose.effects`, `compose.effects_eval`, `compose.adv_mutate`), and writes:
  - Dataset JSONL (`--out`), audit sidecar (`*_audit.jsonl`), stats JSON (`*_stats.json`), and optional external evaluation reports.
  - Capability, DSL, and dedupe diagnostics (see `compose.audit` soft/hard logs).
- Supports dry-runs (`--dry_run`) that only resolve configs, RNG/dedupe, and stats to validate settings without generating data.

### `compose/` - modular dataset machinery

Key submodules (lazy-loaded via `compose/__init__.py`) include:

- **Interface & defaults**: `cli.py` defines argument groups; `constants.py` holds centralized defaults; `knobs.py` and `utils.py` provide helper toggles.
- **Coverage & balancing**: `state.py`, `quota.py`, `balance.py`, and `mismatch.py` maintain weight tables (`GOAL_WEIGHTS`, `CARRIER_WEIGHTS`, etc.), enforce coverage minima, and balance splits.
- **Attack rendering**: `attacks.py`, `carriers.py`, `payload.py`, `surface_noise.py`, `conversation.py`, and `dsl_runtime.py` assemble intents, carriers, payload transformations, surface noise, and DSL-backed templates. `adv_mutate.py` enables adversarial mutation rounds, while `effects.py`/`effects_eval.py` capture gating logic.
- **Data sources & RNG**: `sources.py` ingests curated corpora, HuggingFace datasets, or local JSON/JSONL targets; `rng.py` and `stable_random.py` share namespace-safe RNGs across workers.
- **Deduplication, audits, serialization**: `dedupe_helpers.py` wires CLI flags into `dedupe_core.Deduper`; `audit.py`, `leakage.py`, `symmetry.py`, and `serialize.py` track diagnostics, leakage guards, and on-disk artifacts; `workers.py` coordinates multiprocessing.

### `dedupe_core.py` & `dedupe/`

- `dedupe_core` exposes `DEFAULT_DEDUPER_KWARGS`, `create_default_deduper`, `get_default_deduper_kwargs`, `Deduper`, `DedupeRecord`, `LSHMinhashIndex`, and helpers such as `simhash_weighted_text`. When native extensions (`simhash`, `numba`, `datasketch`, `annoy`, `faiss`) are available, it simply forwards to `dedupe.core` / `dedupe.index`. Otherwise it provides a deterministic fallback (Blake2b digests, simple shingling) so pipelines remain runnable in restricted sandboxes.
- `dedupe/core.py` implements the full pipeline:
  1. Normalize text (NFKC, whitespace folding) and hash trigram shingles.
  2. Compute SimHash signatures (`_simhash_weighted_np`) and early-exit via Hamming distance.
  3. Feed shingles into LSH MinHash buckets (`dedupe/index.py`) with adjustable `n_hash`/`bands`.
4. Compare hashed trigram vectors and optional external embedding vectors; optionally offload ANN queries to `dedupe/annoy.py` or `dedupe/faiss.py`.
- `compose.dedupe_helpers` configures these stages using CLI flags (`--simhash_bits`, `--simhash_thresh`, `--shingle_k`, `--minhash_n`, `--minhash_bands`, `--jaccard_thresh`, `--semantic_dedupe*`, preservation toggles) so positives and negatives share a single dedupe cache.

### `dsl_core/`

- Acts as a facade exposing reworked modules (`spec.py`, `anchors.py`, `invariants.py`, `renderers.py`, `generator.py`, `soft.py`, `sandbox.py`, `textops`, `utils`). `dsl_core.__all__` mirrors the legacy `dsl_core.py` interface so existing scripts keep working.
- Defines AttackSpec schemas, register/region/persona inventories, and sandbox hooks (safe `eval`, `exec`, tool overrides). Invariants and anchors enforce structural constraints (nonce placement, policy disclaimers, tool injection patterns), while `soft.py` manages soft-evidence banks and CN-share measurement (`CN_share`).
- `compose.dsl_runtime` and `micro_grammar` dynamically import these exports to render structured positives, apply invariants, and share soft-evidence caches.

### `micro_grammar.py`

- Provides dataclasses (`Slot`, `MicroGrammar`, `StyleProfile`) that decompose "soft evidence" phrases into slots with register/industry/region overlays, optional strong values, and CN-share constraints.
- Uses `stable_random` namespaces for deterministic permutations, dedupes realized phrases through optional `Deduper` hooks, and exposes helper APIs (`refill_bank`, `generate_micro_prototypes`, `dedupe_phrases`, `grammar_*` accessors) consumed by `compose.capabilities.refresh_micro_grammar_bank` and the phrase extractor.
- Ensures surface hints stay within `max_len`, `min_CN_share`, and `tight_pairs` rules, and can auto-attach to `dsl_core` so DSL renderers inherit refreshed paraphrase banks.

### `stable_random.py`

- Centralizes deterministic seeding across modules via `stable_seed_hex/int`, `stable_rng`, and `derive_rng`.
- `RandomBinder` and `random_module_binding` temporarily bind Python's `random` module to namespace + seed-specific RNGs, preventing cross-thread contamination. `compose.rng`, `micro_grammar`, and `dsl_core` all consume these helpers to keep runs reproducible even under multiprocessing.

## Installation & Environment

1. Use Python **3.9+ (3.10 recommended)**.
2. (Optional) Set up a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install base dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Optional extras:
   - **Dedupe accelerators**: `pip install simhash numba annoy faiss-cpu datasketch xxhash`.
   - **Micro-grammar/soft evidence refresh**: `pip install sentence-transformers torch jieba jieba-fast scikit-learn psutil`.
   - **HuggingFace streaming**: set `HF_HOME` and install `datasets` extras if hosting offline mirrors.

Ensure the repo root stays on `PYTHONPATH` (or run via `python v3.py` from the root) so `compose`, `dedupe`, `dsl_core`, and `micro_grammar` imports resolve.

## Quick Start Workflow

1. **Prepare resources**: Update DSL banks if needed (`python extract_and_bank_zh_phrases.py --input source/zh_lines.txt --cluster_thr 0.85 --bank_all`), refresh micro-grammar caches (handled automatically on first `v3.py` run), and stage any custom targets via `--targets_json`.
2. **Run the composer**:
   ```powershell
   python v3.py ^
     --out data.jsonl ^
     --n 6000 --seed 42 ^
     --family_max_frac 0.08 ^
     --artifact_rate 0.25 ^
     --min_CN_share_auto ^
     --neg_ratio 0.15 ^
     --use_dsl
   ```
   The CLI prints a capability report, coverage goals, RNG fingerprint, and dedupe config before generating samples.
3. **Inspect outputs** (paths derived from `--out`):
  - `data.jsonl` - composed prompts with metadata (`label`, `carrier`, `goal`, `artifact_flags`, etc.).
  - `data_audit.jsonl` - row-level diagnostics from `compose.audit` and DSL/grammar refresh logs.
  - `data_stats.json` - aggregate coverage, leakage, symmetry, dedupe, and CN-share summaries.
   - Optional: external evaluation report (`--ext_eval_report`), capability snapshot, refreshed grammar bank diffs.
4. **Iterate**: Adjust CLI weights, dedupe knobs, or DSL templates; use `--dry_run` for quick validation, or lower `--mp_workers` when debugging.

## CLI Essentials (`compose/cli.py`)

Argument groups (all bilingual in the CLI help):

- **I/O & Reproducibility**: `--out`, `--n`, `--seed`, `--dry_run`.
- **Data Sources**: `--targets_json` for local prompts, `--hf_revision` for HF pinning.
- **Language & Formatting**: `--min_CN_share`, `--min_CN_share_auto/--no-min_CN_share_auto`, `--CN_policy`, `--code_switch_prob`, `--alias_p_cn`.
- **Sampling & Composition**: `--pick_strategy`, `--family_max_frac`, `--coverage_min_per_combo`, `--oversample_mult`, `--anchor_free_p`, `--structural_pos_ratio`, `--carrier_mix_prob`, `--stack_prob`, `--soft_hint_rate`, `--disc_rate`, `--art_match_rate`, `--mirror_placeholderless`, `--use_dsl`, `--evolve_variants`, `--balance_mode`, `--rebalance_beta`.
- **Negatives**: `--neg_ratio`, `--plain_neg_ratio`, `--gray_neg_keep_frac`, `--neg_effect_guard`.
- **Artifacts & Evidence**: `--artifact_rate`, `--struct_evidence_rate`, plus weight knobs for explicit vs euphemistic families.
- **Dedupe & Similarity**: `--semantic_dedupe`, `--semantic_dedupe_thr`, `--simhash_bits`, `--simhash_thresh`, `--shingle_k`, `--minhash_n`, `--minhash_bands`, `--jaccard_thresh`, `--dedupe_preserve_carrier`, `--dedupe_preserve_codeblocks`.
- **Effect-level gating & evaluation**: `--effect_gate_for`, `--gate_semantic_injection`, `--effect_policy`, `--effect_fallback`, `--effect_replicas`, `--model_eval_fallback`, `--adv_mutate`, `--adv_iters`, `--mt_tool_override_rate`, `--invariance_end_nonce`.
- **Concurrency & performance**: `--target_workers`, `--mp_workers`, `--mp_batch`.
- **External evaluation**: `--ext_eval_sources`, `--ext_eval_size`, `--ext_eval_shapes`, `--ext_eval_lang`, `--ext_eval_report`.

All defaults live in `compose/constants.py`, so you can trace how each knob influences downstream modules.

## Deduplication & Similarity Pipeline

1. **Prepare**: `compose.dedupe_helpers` calls `get_default_deduper_kwargs` with CLI overrides and wires the shared `Deduper`.
2. **Normalize + SimHash**: Each `DedupeRecord` stores normalized text, SimHash signature, hashed shingles, and exact digest; Hamming-distance filters short-circuit obvious duplicates.
3. **MinHash-LSH**: Character shingles are fed through `_minhash` / `LSHMinhashIndex` to approximate Jaccard similarity before running exact comparisons.
4. **Exact checks & vectors**: Remaining candidates undergo exact Jaccard over shingles, optional cosine checks over hashed trigram vectors (`vec_dim`, `cosine_thresh`), and user-supplied external vectors (e.g., SentenceTransformer embeddings). ANN indices (FAISS/Annoy via `dedupe/faiss.py` and `dedupe/annoy.py`) kick in once the vector cache exceeds `annoy_rebuild_every`.
5. **Semantic heuristics**: Set `--semantic_dedupe` + threshold to enable SequenceMatcher-based edit similarity, useful for catching localized paraphrases that sneak past SimHash.

Because the same deduper instance screens positives and negatives, the system avoids cross-class leakage and guarantees consistent caching across workers.

## DSL & Micro-Grammar Integration

- `compose.dsl_runtime` loads `dsl_core` exports (AttackSpec, renderers, sandbox hooks) and binds register/region/persona pools to coverage weights from `compose.state`.
- At startup, `v3.py` optionally calls `compose.capabilities.refresh_micro_grammar_bank(args.seed)` which:
  1. Imports `micro_grammar` if available.
  2. Attaches the grammar banks to `dsl_core` (`attach_to_dsl_core`).
  3. Runs optional refill/prototype steps (see `micro_grammar.generate_micro_prototypes`, `refill_bank`, `rebuild_soft_check`) guarded by `stable_random` seeds and dedupe.
- During composition, DSL templates can request slot-level hints, style profiles, or grammar-based paraphrases. CN-share checks fall back to `micro_grammar.CN_share` or `dsl_core.CN_share`, ensuring quotas remain satisfied.
- Refresh cadence: rerun `extract_and_bank_zh_phrases.py` when new corpora are available, then re-run `v3.py` to ingest the refreshed bank snapshot.

## Extending the Pipeline

1. **New attacks/carriers**: update `compose/attacks.py`, `compose/carriers.py`, or `compose/payload.py`; register the new families in `GOAL_WEIGHTS`/`CARRIER_WEIGHTS` (or add overrides via CLI).
2. **Custom DSL specs**: add `AttackSpec` definitions, anchors, or invariants in `dsl_core/spec.py`, `anchors.py`, `invariants.py`, or create new renderers in `dsl_core/renderers.py`.
3. **Micro-grammar variants**: extend `micro_grammar.Slot` inventories, overlay registers/industries/regions, or script new `StyleProfile`s; rerun the refresh helper to push diffs into DSL soft banks.
4. **Hard-negative sources**: drop curated corpora under `source/` or teach `compose/sources.py` how to stream additional datasets, then target them via `--targets_json` or new CLI knobs.
5. **Dedupe tuning**: adjust `DEFAULT_DEDUPER_KWARGS` or extend the ANN helpers (`dedupe/faiss.py`, `dedupe/annoy.py`) to experiment with alternative vector indexes.

## Reproducibility & Troubleshooting

- Seeds: `--seed` fuels `stable_random` namespaces, SimHash minhash permutations, and micro-grammar permutations. Compose prints a fingerprint so you can diff configurations.
- Sidecars: The audit JSONL logs every soft/hard rejection (`compose.audit.*` codes), while stats JSON reports coverage, leakage, artifact rates, dedupe savings, CN bins, and optional external eval summaries.
- Capability report: emitted before generation; if DSL, micro-grammar, or optional dedupe accelerators fail to import, the report includes the exception so you can install missing deps.
- Debugging tips:
  - Use `--dry_run` to validate configs quickly.
  - Reduce concurrency (`--mp_workers 1`) to capture full stack traces.
  - Enable semantic dedupe temporarily to inspect borderline duplicates.
  - Keep an eye on `*_audit.jsonl` for leakage flags or DSL/micro-grammar refresh errors surfaced via `compose.audit.audit_soft`.

This README replaces the older brief summary and now documents how `v3.py`, `compose`, `dedupe_core.py`, `dedupe/`, `dsl_core/`, `micro_grammar.py`, and `stable_random.py` fit together so you can confidently extend or operate the CN prompt-injection composer.
