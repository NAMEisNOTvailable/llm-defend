# Architecture Notes

`llm-defend` is organised as a dataset-composition pipeline rather than a single script.

## Main Flow

1. `v3.py` reads CLI options, configures deterministic seeds, and prepares capability checks.
2. `compose/` selects attack goals, delivery carriers, payload families, hard negatives, and balancing constraints.
3. `dsl_core/` renders structured attack specifications with anchors, invariants, and sandbox-style prompt surfaces.
4. `micro_grammar.py` generates Chinese soft-evidence variants and phrase-level style diversity.
5. `dedupe/` screens generated records to reduce duplicates and cross-class leakage.
6. Audit and statistics files are written beside the generated dataset.

## Core Packages

| Path | Role |
| --- | --- |
| `compose/` | CLI, generation state, attack rendering, quota/balancing, workers, audits, and serialization |
| `dedupe/` | SimHash, MinHash-LSH, hashed trigram comparison, and optional vector/ANN dedupe helpers |
| `dsl_core/` | Attack specs, anchors, invariants, renderers, soft evidence, and sandbox utilities |
| `source/` | Public toy samples plus local-only prepared source data ignored by git |
| `scripts/` | Data-preparation helpers for private/upstream source exports |

## Design Priorities

- Reproducible generation through stable seeds.
- Coverage balance across attack goals and delivery modalities.
- Chinese-first prompt construction, not English-only prompt translation.
- Reviewable outputs through audit sidecars and summary statistics.
- Optional acceleration without making optional dependencies mandatory.
