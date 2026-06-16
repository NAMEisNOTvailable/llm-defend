# Source Data

This directory contains local corpora and prompt-source files used by the composer.

Large source files are kept here instead of the repository root so the top-level project view stays readable.

| File or folder | Purpose |
| --- | --- |
| `combined_prompts.jsonl` | Combined prompt source corpus |
| `JailBench.jsonl` / `JailBench.csv` | JailBench-derived source material |
| `zh_lines.txt` | Chinese phrase/source text used by phrase-bank refresh |
| `datasets/` | Bundled dataset mirrors used by local loaders |

Use `extract_and_bank_zh_phrases.py` from the repository root when refreshing phrase banks:

```bash
python extract_and_bank_zh_phrases.py --input source/zh_lines.txt --bank_all
```
