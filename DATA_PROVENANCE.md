# Data Provenance

This repository contains original source code for a prompt-injection dataset composer and a set of bundled research corpora used as local inputs. The MIT license in `LICENSE` applies to the original source code and documentation. It does not relicense third-party datasets, mirrored corpora, benchmark prompts, generated prompt corpora, or other data files.

## Licensing Boundary

- Original source code and documentation: MIT license, see `LICENSE`.
- Data files under `source/`: not relicensed by this repository.
- Generated datasets created by running the composer: check the license and use restrictions of every input source used for that run.
- If a data file has unclear upstream terms, treat it as research-review material only and verify the original source before reuse or redistribution.

## Data Inventory

| Path | Role in this repository | Reuse note |
| --- | --- | --- |
| `source/combined_prompts.jsonl` | Combined prompt corpus used as input material for generation and phrase extraction. | Derived/compiled data; not covered by the MIT code license. |
| `source/JailBench.jsonl` and `source/JailBench.csv` | JailBench-derived benchmark/source material. | Retains upstream dataset terms; verify the original source before reuse. |
| `source/datasets/JailJudge/*` | JailJudge-derived mirrored or transformed source files. | Retains upstream dataset terms; verify the original source before reuse. |
| `source/MultiJail_zh.jsonl` | Chinese split exported through `datasets.load_dataset("walledai/MultiJail", split="zh")`. | Retains the Hugging Face dataset card and upstream source terms. |
| `source/Safety_prompt_instruction_attack_scenarios.jsonl` | Safety and prompt-injection scenario source material. | Retains source dataset terms; not relicensed here. |
| `source/Safety_prompt_typical_safety_scenarios.jsonl` | Safety scenario source material used for prompt coverage. | Retains source dataset terms; not relicensed here. |
| `source/zh_lines.txt` | Extracted Chinese text lines used to refresh phrase banks. | Compiled/extracted text; not licensed as an original standalone corpus. |
| `source/load_dataset.py` | Helper script showing how one local source file was exported. | Script is MIT; downloaded/exported dataset content is not. |

## Responsible-Use Note

The data includes adversarial, unsafe, or policy-violating prompt examples because the project is about LLM security evaluation. These examples are included only to support defensive evaluation, reproducible safety research, and portfolio review. They should not be treated as endorsement, operational guidance, or a deployable attack dataset.

## Reviewer Guidance

For portfolio review, inspect the code, data layout, and reproducibility design. For reuse outside review, cite the upstream datasets where applicable, comply with their licenses and acceptable-use terms, and keep derived outputs clearly separated from original source code.
