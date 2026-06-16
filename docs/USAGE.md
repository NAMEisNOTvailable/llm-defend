# Usage and Outputs

## Basic Generation

Run commands from the repository root:

```bash
python v3.py --out outputs/demo.jsonl --n 500 --seed 42 --use_dsl
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--out` | Output JSONL path |
| `--n` | Number of samples to generate |
| `--seed` | Deterministic seed for reproducibility |
| `--use_dsl` | Enable DSL-backed prompt rendering |
| `--neg_ratio` | Include hard-negative examples |
| `--dry_run` | Validate configuration without generating full output |

## Output Files

For an output path such as `outputs/demo.jsonl`, the pipeline can produce:

| File | Purpose |
| --- | --- |
| `demo.jsonl` | Generated prompt records |
| `demo_audit.jsonl` | Row-level audit decisions and diagnostics |
| `demo_stats.json` | Aggregate coverage, language-share, and dedupe statistics |

## Phrase Bank Refresh

The bundled Chinese text source lives under `source/`.

```bash
python extract_and_bank_zh_phrases.py --input source/zh_lines.txt --bank_all
```

The extractor is intentionally separate from the main composer so reviewers can inspect generation and phrase-bank refresh as different workflow stages.
