# Usage and Outputs

## Basic Generation

Run commands from the repository root:

```bash
python v3.py --out outputs/demo.jsonl --n 500 --seed 42 --use_dsl
```

For a public-repository-only smoke run, use the toy sample explicitly:

```bash
python v3.py --out outputs/demo.jsonl --n 50 --seed 42 --use_dsl \
  --targets_json source/samples/public_toy_prompts.jsonl
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

## Source Data Preparation

The public repository does not ship the full third-party or derived prompt
source corpus. Prepare it from an access-controlled local export before running
experiments that need the full data:

```bash
python scripts/prepare_source_data.py --private-source ../llm-defend-private-data/source
```

## Phrase Bank Refresh

After full source data has been prepared locally, refresh Chinese phrase banks
from the local private `zh_lines.txt` file:

```bash
python extract_and_bank_zh_phrases.py --input source/zh_lines.txt --bank_all
```

The extractor is intentionally separate from the main composer so reviewers can inspect generation and phrase-bank refresh as different workflow stages.
