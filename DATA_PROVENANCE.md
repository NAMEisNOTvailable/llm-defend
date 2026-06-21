# Data Provenance

Last reviewed: 2026-06-17.

This repository contains original source code for a prompt-injection dataset
composer plus a tiny original toy sample under `source/samples/`. The
repository intentionally does not mirror full third-party, gated, locally
derived, or potentially unsafe prompt corpora. Full source data should be kept
in an access-controlled local export or regenerated from upstream sources
under their own terms.

The MIT license in `LICENSE` applies only to original source code,
documentation, and original toy samples. It does not relicense third-party
datasets, benchmark prompts, local mirrors, generated prompt corpora, or
derived text files.

## Licensing Boundary

- Original source code and documentation: MIT license, see `LICENSE`.
- Public toy files under `source/samples/`: original MIT-licensed examples for smoke tests and documentation.
- Full data files prepared locally under `source/`: retain their upstream licenses, access terms, and citation expectations.
- Generated datasets created by running the composer: inherit the terms of every upstream input used for that run.
- Files with unclear upstream terms should be treated as research-review material only until the original source terms are verified.

## Upstream Sources

The paths below are expected local paths after running
`scripts/prepare_source_data.py` or regenerating data from upstream. They are
not committed to this repository.

| Local path after preparation | Upstream source | Local use | License / access | Repository status |
| --- | --- | --- | --- | --- |
| `source/Safety_prompt_typical_safety_scenarios.jsonl` | [thu-coai/Safety-Prompts][safety-prompts-github] / [Hugging Face][safety-prompts-hf] | Chinese safety scenario coverage for prompt composition and phrase extraction. | Apache-2.0 on the upstream GitHub and Hugging Face dataset card. | Not mirrored here; prepare from a local source export or upstream. |
| `source/Safety_prompt_instruction_attack_scenarios.jsonl` | [thu-coai/Safety-Prompts][safety-prompts-github] / [Hugging Face][safety-prompts-hf] | Instruction-attack scenario coverage for prompt composition. | Apache-2.0 on the upstream GitHub and Hugging Face dataset card. | Not mirrored here; prepare from a local source export or upstream. |
| `source/JailBench.csv`, `source/JailBench.jsonl` | [STAIR-BUPT/JailBench][jailbench-github] / [paper][jailbench-paper] | Chinese jailbreak-risk benchmark prompts used as adversarial source material. | MIT on the upstream GitHub repository. | Not mirrored here; use upstream sources or a local export. |
| `source/datasets/JailJudge/JAILJUDGE_ID.json`, `source/datasets/JailJudge/JAILJUDGE_OOD.json`, `source/datasets/JailJudge/JAILJUDGE_TRAIN.json`, `source/datasets/JailJudge/JAILJUDGE_OOD_zh*.jsonl` | [usail-hkust/JailJudge][jailjudge-hf] / [code repository][jailjudge-github] / [paper][jailjudge-paper] | Jailbreak-judge benchmark/reference material and Chinese extracted subsets. | Hugging Face dataset card shows a custom `jailjudge` license and requires accepting access conditions/contact sharing; the code repository's MIT license does not override dataset terms. | Not mirrored in public git; keep access-controlled after accepting upstream terms. |
| `source/MultiJail_zh.jsonl` | [walledai/MultiJail][multijail-hf] / [paper][multijail-paper] | Chinese multilingual jailbreak prompt split used as source material. | No explicit license field was exposed by the Hugging Face card/API during this review; verify terms before reuse or redistribution. | Not mirrored in public git; regenerate only when terms permit. |
| `source/combined_prompts.jsonl` | Local derived compilation from the prepared inputs above. | Combined prompt/task corpus used by the composer. | Inherits the most restrictive applicable upstream terms. | Derived local file only; not independently licensed or committed. |
| `source/zh_lines.txt` | Local derived extraction from prepared prompt sources. | Prompt-line input for Chinese phrase-bank refresh. | Inherits the most restrictive applicable upstream terms. | Derived local file only; not independently licensed or committed. |
| `source/load_dataset.py` | Local helper script. | Documents the MultiJail Chinese split export path. | Script is MIT with the rest of the original code; downloaded/exported data remains under upstream terms. | Script is included; exported data remains local. |
| `source/samples/public_toy_prompts.jsonl` | Original toy records. | Smoke tests and public examples only. | MIT with the rest of the original code. | Public; not from third-party datasets. |

## Known Local Transformations

- Safety-Prompts inputs can be converted from upstream scenario JSON files into local line-oriented source files.
- JailBench can be stored both as the upstream-style CSV snapshot and a JSONL conversion for composer input.
- JailJudge Chinese files are extracted subsets derived from multilingual OOD data; gated files must remain access-controlled.
- MultiJail can be exported from the Hugging Face `zh` split by `source/load_dataset.py` when terms permit.
- `combined_prompts.jsonl` and `zh_lines.txt` are local compilations/extractions from prepared sources, so they should not be cited or reused as independent original datasets.

Several historical local snapshots contain legacy encoding/mojibake artifacts. For quantitative research, redistribution, or production reuse, regenerate source files from upstream and keep the exact upstream revision, license, and preprocessing script in the experiment record.

## Responsible-Use Note

Prepared full data includes adversarial, unsafe, or policy-violating prompt
examples because the project is about LLM security evaluation. Those examples
are intentionally excluded from this repository and should be handled only
for defensive evaluation and reproducible safety research.
They are provided for defensive evaluation context, not endorsement or
operational guidance.

## Reuse Guidance

For project inspection, review the code, toy sample layout, provenance
boundary, and reproducibility design. For reuse outside review, cite the
upstream datasets, comply with their licenses and acceptable-use terms, and keep
derived outputs clearly separated from original source code.

[safety-prompts-github]: https://github.com/thu-coai/Safety-Prompts
[safety-prompts-hf]: https://huggingface.co/datasets/thu-coai/Safety-Prompts
[jailbench-github]: https://github.com/STAIR-BUPT/JailBench
[jailbench-paper]: https://arxiv.org/abs/2502.18935
[jailjudge-hf]: https://huggingface.co/datasets/usail-hkust/JailJudge
[jailjudge-github]: https://github.com/usail-hkust/Jailjudge
[jailjudge-paper]: https://arxiv.org/abs/2410.12855
[multijail-hf]: https://huggingface.co/datasets/walledai/MultiJail
[multijail-paper]: https://arxiv.org/abs/2310.06474
