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

Raw and bundled corpora are kept under `source/` so the repository root remains focused on executable entry points and package folders.

The root-level duplicate `zh_lines.txt` was removed because the canonical copy is `source/zh_lines.txt`.

## Review Notes

Generated prompt data should be reviewed through:

- coverage by attack goal and delivery modality
- duplicate and near-duplicate rates
- Chinese-language share and code-switching behaviour
- leakage between positive and negative examples
- audit sidecar reasons for rejected or transformed samples
