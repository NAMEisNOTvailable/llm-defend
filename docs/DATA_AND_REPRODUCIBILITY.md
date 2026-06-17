# Data and Reproducibility

## Reproducibility

The project uses stable random helpers so runs with the same seed and configuration are repeatable.

Important files:

| Path | Purpose |
| --- | --- |
| `stable_random.py` | Stable seed derivation and namespace-safe RNG helpers |
| `compose/rng.py` | Composer-level RNG configuration |
| `compose/audit.py` | Audit sidecar logging |
| `compose/serialize.py` | Output serialization |

## Data Handling

Raw and derived source inputs are kept under `source/` so the repository root remains focused on executable entry points and package folders.

The current local sources are provenance-tracked in [`../DATA_PROVENANCE.md`](../DATA_PROVENANCE.md). They include Safety-Prompts, JailBench, JailJudge, and MultiJail-derived inputs, plus local compilation/extraction files such as `combined_prompts.jsonl` and `zh_lines.txt`.

The root-level duplicate `zh_lines.txt` was removed because the canonical local copy is `source/zh_lines.txt`.

For reproducible research, record the upstream dataset revision, accepted license/access terms, preprocessing script, random seed, and generated audit sidecar for each run. The bundled local snapshots are useful for portfolio review, but fresh upstream exports are preferred when running new quantitative experiments.

## Review Notes

Generated prompt data should be reviewed through:

- coverage by attack goal and delivery modality
- duplicate and near-duplicate rates
- Chinese-language share and code-switching behaviour
- leakage between positive and negative examples
- audit sidecar reasons for rejected or transformed samples
