# LLM Defend

Chinese-first prompt-injection dataset composer for building coverage-balanced adversarial prompts, hard negatives, and reproducible LLM safety evaluation inputs.

This repository is presented as an AI safety and LLM security portfolio project. It focuses on deterministic dataset generation, Chinese prompt-injection coverage, deduplication, and research-grade auditability.

## Project Snapshot

| Area | Summary |
| --- | --- |
| Domain | LLM security, prompt-injection data generation, Chinese adversarial prompting |
| Main entry point | `v3.py` |
| Core packages | `compose/`, `dedupe/`, `dsl_core/` |
| Data sources | `source/` |
| Key outputs | Generated JSONL datasets, audit sidecars, coverage/statistics reports |

## What This Demonstrates

- Designed a modular prompt-injection composer with deterministic random seeding.
- Built a DSL-driven generation system for attack families, carriers, payloads, anchors, and hard negatives.
- Implemented multi-stage deduplication using SimHash, MinHash-LSH, hashed trigram similarity, and optional ANN/vector backends.
- Added Chinese-language micro-grammar tooling to improve lexical variety while controlling CN-share and prompt style.
- Produced audit and statistics outputs so generated datasets can be reviewed rather than treated as opaque synthetic data.

## Repository Structure

```text
compose/       Dataset composition pipeline, CLI options, balancing, workers, audits
dedupe/        Similarity and deduplication backends
dsl_core/      DSL schemas, renderers, anchors, invariants, and sandbox helpers
source/        Bundled corpora and source datasets used by the composer
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
python v3.py --out outputs/demo.jsonl --n 500 --seed 42 --use_dsl
```

Refresh Chinese phrase banks from the bundled source corpus:

```bash
python extract_and_bank_zh_phrases.py --input source/zh_lines.txt --bank_all
```

## Documentation

- [Architecture Notes](docs/ARCHITECTURE.md)
- [Usage and Outputs](docs/USAGE.md)
- [Data and Reproducibility](docs/DATA_AND_REPRODUCIBILITY.md)
- [Source Data Notes](source/README.md)
- [Data Provenance](DATA_PROVENANCE.md)

## Related Project

This repository complements [`llm-safety-evaluation`](https://github.com/NAMEisNOTvailable/llm-safety-evaluation), which evaluates Mandarin-English prompt-injection behaviour across multiple LLMs.

## License and Data

Original source code and documentation are licensed under the MIT License. Bundled corpora, mirrored datasets, benchmark prompts, generated prompt corpora, and other data files are not relicensed by this repository. See [Data Provenance](DATA_PROVENANCE.md) before reusing data or generated outputs.

## Status

Research portfolio project. The code is intended to demonstrate AI safety tooling, reproducible dataset generation, and security-focused evaluation design.
