# Chinese Prompt Injection Dataset Toolkit

This repository bundles the dataset composer (`make_malicious_prompts_cn_compose_v2.py`)
with its DSL helper module (`dsl_core.py`). Together they generate Chinese prompt
injection attacks, paired benign negatives, and the audit metadata used for research
or model safety evaluation.

## Requirements

Install Python 3.9 or newer and the dependencies listed in `requirements.txt`:

```
pip install -r requirements.txt
```

The composer imports `dsl_core.py` directly, so keep both files in the same module
search path (for instance, the same directory) when running the generator or when
embedding this code in another project.

## `make_malicious_prompts_cn_compose_v2.py`

This script is the main entry point. It orchestrates candidate sampling, Chinese
surface repairs, de-duplication, DSL-based positives, and extensive auditing.

### Quick start

```
python make_malicious_prompts_cn_compose_v2.py \
  --out data.jsonl \
  --n 6000 \
  --seed 42 \
  --neg_ratio 0.15 \
  --family_max_frac 0.08 \
  --artifact_rate 0.25 \
  --min_cjk_share_auto
```

The command above writes a JSONL dataset plus a matching `_stats.json` report. Use
`--dry_run` to exercise the pipeline without emitting rows, or `--use_dsl` to rely on
the state-machine generator for positive examples.

Run `python make_malicious_prompts_cn_compose_v2.py --help` for the full set of CLI
flags (coverage quotas, semantic gates, evaluation hooks, etc.).

> Default hints: `family_max_frac=0.06`, `artifact_rate=0.10`, and `min_cjk_share_auto=True` so the quick-start command can omit that flag.

### Outputs

- `<out>.jsonl`: training-ready rows (positives and negatives).
- `<out>_audit.jsonl`: **optional** file with full audit context per row when audit capture is enabled.
- `<out>_stats.json`: run fingerprint, aggregate counters, and (when available) `semantic_neighbor_audit` and `dsl_coverage`.
- `<out>_coverage_matrix.json`, `<out>_report.json`, `<out>_quality_report.json`: coverage and quality diagnostics emitted by the corresponding pipeline stages.

Several options influence reproducibility (`--seed`, `--strict_mode`) and negative
repair (`--min_cjk_share`, `--plain_neg_ratio`). Keep the `dsl_core.py` helpers
available when enabling `--use_dsl`, soft evidence hints, or sinicization steps.

## `dsl_core.py`

The DSL module provides:

- Schema definitions and enums for intents, carriers, deliveries, and style axes.
- Rendering helpers (carrier templates, alias randomizers, wrap utilities).
- Contract and side-effect analyzers used to gate successful prompt injections.
- lightweight stubs for environments lacking optional dependencies (e.g. YAML).

Import `dsl_core` from your own scripts to reuse these utilities, or call
`generate_batch` (when available) to obtain DSL-crafted positives directly.

## Development Notes

- Both files assume UTF-8 without BOM (this repo already strips it).
- If you vendor the modules elsewhere, keep localized text assets intact so CJK
  repair and alias logic remain effective.
- The composer produces deterministic output per `--seed`; update fingerprints when
  modifying templates or helper logic.

## Troubleshooting

- Ensure `regex` and `datasets` are installed; they extend the Python standard
  library regex engine and load evaluation corpora.
- When running on machines without the optional DSL runtime, guard `--use_dsl` or
  wrap imports in try/except blocks similar to the existing fallbacks.
- Large runs may take several minutes; monitor stdout for audit counters and quota
  warnings to catch configuration issues early.
