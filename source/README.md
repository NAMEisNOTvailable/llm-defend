# Source Data

This directory contains third-party and locally derived prompt-source files used by the composer. It is not an original dataset release, and the repository's MIT code license does not relicense these files.

Large source files are kept here instead of the repository root so the top-level project view stays readable.

| File or folder | Local rows / role | Upstream source | Handling note |
| --- | --- | --- | --- |
| `Safety_prompt_typical_safety_scenarios.jsonl` | 70,000 local lines for broad Chinese safety scenarios. | `thu-coai/Safety-Prompts` | Apache-2.0 upstream; local derived snapshot. |
| `Safety_prompt_instruction_attack_scenarios.jsonl` | 30,000 local lines for instruction-attack scenarios. | `thu-coai/Safety-Prompts` | Apache-2.0 upstream; local derived snapshot. |
| `JailBench.jsonl` / `JailBench.csv` | 2,160 local JSONL rows plus CSV source material. | `STAIR-BUPT/JailBench` | MIT upstream; local CSV snapshot and JSONL conversion. |
| `datasets/JailJudge/` | JailJudge ID/OOD files, a Git LFS pointer for train data, and 630-row Chinese OOD subsets. | `usail-hkust/JailJudge` | Custom/gated Hugging Face dataset terms; local mirror/subset files. |
| `MultiJail_zh.jsonl` | 315 local rows from the Chinese split. | `walledai/MultiJail` | Exported with `load_dataset.py`; no explicit license field found during review. |
| `combined_prompts.jsonl` | 102,273 local prompt/task rows used by the composer. | Local compilation from bundled inputs. | Derived file; inherits upstream terms. |
| `zh_lines.txt` | 102,273 extracted prompt lines for phrase-bank refresh. | Local extraction from bundled inputs. | Derived prompt-line file; inherits upstream terms. |
| `load_dataset.py` | Helper script for exporting MultiJail Chinese data. | Local script. | Script is MIT; exported data is not. |

Use `extract_and_bank_zh_phrases.py` from the repository root when refreshing phrase banks:

```bash
python extract_and_bank_zh_phrases.py --input source/zh_lines.txt --bank_all
```

These files are research inputs and are not relicensed by the repository's MIT code license. See [`../DATA_PROVENANCE.md`](../DATA_PROVENANCE.md) for source links, license/access notes, mirror status, and transformation notes.
