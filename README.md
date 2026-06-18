# LLM Defend

Chinese-first prompt-injection dataset composer for building coverage-balanced adversarial prompts, hard negatives, and reproducible LLM safety evaluation inputs.

This repository is presented as an AI safety and LLM security portfolio project. It focuses on deterministic dataset generation, Chinese prompt-injection coverage, deduplication, and research-grade auditability.

## Project Snapshot

| Area | Summary |
| --- | --- |
| Domain | LLM security, prompt-injection data generation, Chinese adversarial prompting |
| Main entry point | `v3.py` |
| Core packages | `compose/`, `dedupe/`, `dsl_core/` |
| Data sources | Public toy samples plus prepare scripts/provenance for private or upstream source data |
| Key outputs | Generated JSONL datasets, audit sidecars, coverage/statistics reports |

## What This Demonstrates

- Designed a modular prompt-injection composer with deterministic random seeding.
- Built a DSL-driven generation system for attack families, carriers, payloads, anchors, and hard negatives.
- Implemented multi-stage deduplication using SimHash, MinHash-LSH, hashed trigram similarity, and optional ANN/vector backends.
- Added Chinese-language micro-grammar tooling to improve lexical variety while controlling CN-share and prompt style.
- Produced audit and statistics outputs that make generated datasets easier to inspect and compare.

## Repository Structure

```text
compose/       Dataset composition pipeline, CLI options, balancing, workers, audits
dedupe/        Similarity and deduplication backends
dsl_core/      DSL schemas, renderers, anchors, invariants, and sandbox helpers
source/        Public toy samples plus notes for preparing local/private source inputs
scripts/       Data-preparation helpers
docs/          Architecture, usage, and reproducibility notes
v3.py          Main orchestration CLI
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate a small dataset:

```bash
python v3.py --out outputs/demo.jsonl --n 500 --seed 42 --use_dsl \
  --targets_json source/samples/public_toy_prompts.jsonl
```

Prepare full source data from an access-controlled local export before running
experiments that require the complete third-party or derived corpora:

```bash
python scripts/prepare_source_data.py --private-source ../llm-defend-private-data/source
python extract_and_bank_zh_phrases.py --input source/zh_lines.txt --bank_all
```

The public repository intentionally does not mirror the full prompt-source
corpora. Review [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) before preparing or
reusing any non-toy data.

## Documentation

- [Architecture Notes](docs/ARCHITECTURE.md)
- [Usage and Outputs](docs/USAGE.md)
- [Data and Reproducibility](docs/DATA_AND_REPRODUCIBILITY.md)
- [Source Data Notes](source/README.md)
- [Data Provenance](DATA_PROVENANCE.md)

## Related Project

This repository complements [`llm-safety-evaluation`](https://github.com/NAMEisNOTvailable/llm-safety-evaluation), which evaluates Mandarin-English prompt-injection behaviour across multiple LLMs.

## License and Data

Original source code, documentation, and public toy samples are licensed under
the MIT License. Full third-party and locally derived source inputs are not
distributed in this public repository and are not relicensed by it. See [Data
Provenance](DATA_PROVENANCE.md) before preparing, reusing, or publishing any
non-toy data or generated outputs.

## Status

Research portfolio project. The code is intended to demonstrate AI safety tooling, reproducible dataset generation, and security-focused evaluation design.
