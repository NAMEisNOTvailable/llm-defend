# Data Provenance

Last reviewed: 2026-06-17.

This repository contains original source code for a prompt-injection dataset composer plus third-party and locally derived research inputs under `source/`. The MIT license in `LICENSE` applies only to original source code and documentation. It does not relicense third-party datasets, benchmark prompts, local mirrors, generated prompt corpora, or derived text files.

## Licensing Boundary

- Original source code and documentation: MIT license, see `LICENSE`.
- Data files under `source/`: retain their upstream licenses, access terms, and citation expectations.
- Generated datasets created by running the composer: inherit the terms of every upstream input used for that run.
- Files with unclear upstream terms should be treated as portfolio/research-review material only until the original source terms are verified.

## Upstream Sources

| Local path | Upstream source | Local use | License / access | Mirror or transform status |
| --- | --- | --- | --- | --- |
| `source/Safety_prompt_typical_safety_scenarios.jsonl` | [thu-coai/Safety-Prompts][safety-prompts-github] / [Hugging Face][safety-prompts-hf] | Chinese safety scenario coverage for prompt composition and phrase extraction. | Apache-2.0 on the upstream GitHub and Hugging Face dataset card. | Local derived JSONL snapshot from the upstream JSON files; not a new dataset release. |
| `source/Safety_prompt_instruction_attack_scenarios.jsonl` | [thu-coai/Safety-Prompts][safety-prompts-github] / [Hugging Face][safety-prompts-hf] | Instruction-attack scenario coverage for prompt composition. | Apache-2.0 on the upstream GitHub and Hugging Face dataset card. | Local derived snapshot from the upstream instruction-attack scenario data. |
| `source/JailBench.csv`, `source/JailBench.jsonl` | [STAIR-BUPT/JailBench][jailbench-github] / [paper][jailbench-paper] | Chinese jailbreak-risk benchmark prompts used as adversarial source material. | MIT on the upstream GitHub repository. | Local public-data snapshot plus JSONL conversion; this repository is not the canonical JailBench distribution. |
| `source/datasets/JailJudge/JAILJUDGE_ID.json`, `source/datasets/JailJudge/JAILJUDGE_OOD.json`, `source/datasets/JailJudge/JAILJUDGE_TRAIN.json`, `source/datasets/JailJudge/JAILJUDGE_OOD_zh*.jsonl` | [usail-hkust/JailJudge][jailjudge-hf] / [code repository][jailjudge-github] / [paper][jailjudge-paper] | Jailbreak-judge benchmark/reference material and Chinese extracted subsets. | Hugging Face dataset card shows a custom `jailjudge` license and requires accepting access conditions/contact sharing; the code repository's MIT license does not override dataset terms. | Local mirrored files and derived Chinese subsets. `JAILJUDGE_TRAIN.json` is a Git LFS pointer, not the full training-data payload. |
| `source/MultiJail_zh.jsonl` | [walledai/MultiJail][multijail-hf] / [paper][multijail-paper] | Chinese multilingual jailbreak prompt split used as source material. | No explicit license field was exposed by the Hugging Face card/API during this review; verify terms before reuse or redistribution. | Local export of only the `zh` split via `datasets.load_dataset("walledai/MultiJail", split="zh")`; not a full mirror. |
| `source/combined_prompts.jsonl` | Local derived compilation from the bundled source inputs above. | Combined prompt/task corpus used by the composer. | Inherits the most restrictive applicable upstream terms. | Local compiled derivative; not independently licensed by this repository. |
| `source/zh_lines.txt` | Local derived extraction from bundled prompt sources. | Prompt-line input for Chinese phrase-bank refresh. | Inherits the most restrictive applicable upstream terms. | Prompt-only extraction; not an original standalone corpus. |
| `source/load_dataset.py` | Local helper script. | Documents the MultiJail Chinese split export path. | Script is MIT with the rest of the original code; downloaded/exported data remains under upstream terms. | Loader/export helper, not a data license grant. |

## Known Local Transformations

- Safety-Prompts inputs were converted from upstream scenario JSON files into local line-oriented source files.
- JailBench data is stored both as the upstream-style CSV snapshot and a JSONL conversion for composer input.
- JailJudge Chinese files are extracted subsets derived from the multilingual OOD data; the local train file is only a Git LFS pointer.
- MultiJail was exported from the Hugging Face `zh` split by `source/load_dataset.py`.
- `combined_prompts.jsonl` and `zh_lines.txt` are local compilations/extractions from the bundled sources, so they should not be cited or reused as independent original datasets.

Several historical local snapshots contain legacy encoding/mojibake artifacts. For quantitative research, redistribution, or production reuse, regenerate source files from upstream and keep the exact upstream revision, license, and preprocessing script in the experiment record.

## Responsible-Use Note

The data includes adversarial, unsafe, or policy-violating prompt examples because the project is about LLM security evaluation. These examples are included only to support defensive evaluation, reproducible safety research, and portfolio review. They should not be treated as endorsement, operational guidance, or a deployable attack dataset.

## Reviewer Guidance

For portfolio review, inspect the code, data layout, provenance boundary, and reproducibility design. For reuse outside review, cite the upstream datasets, comply with their licenses and acceptable-use terms, and keep derived outputs clearly separated from original source code.

[safety-prompts-github]: https://github.com/thu-coai/Safety-Prompts
[safety-prompts-hf]: https://huggingface.co/datasets/thu-coai/Safety-Prompts
[jailbench-github]: https://github.com/STAIR-BUPT/JailBench
[jailbench-paper]: https://arxiv.org/abs/2502.18935
[jailjudge-hf]: https://huggingface.co/datasets/usail-hkust/JailJudge
[jailjudge-github]: https://github.com/usail-hkust/Jailjudge
[jailjudge-paper]: https://arxiv.org/abs/2410.12855
[multijail-hf]: https://huggingface.co/datasets/walledai/MultiJail
[multijail-paper]: https://arxiv.org/abs/2310.06474
