# Source Data

This repository intentionally does not mirror full third-party, gated, derived,
or potentially unsafe prompt-source corpora.

Committed public files under this directory are limited to:

- `README.md`: this handling note;
- `load_dataset.py`: a small helper script for exporting the MultiJail Chinese
  split when upstream terms permit;
- `samples/`: small original toy examples for smoke tests and documentation.

Full local experiment inputs such as `combined_prompts.jsonl`, `zh_lines.txt`,
Safety-Prompts exports, JailBench exports, JailJudge files, and MultiJail
exports are ignored by git and should be prepared from an access-controlled
local export or directly from upstream sources:

```bash
python scripts/prepare_source_data.py --source-dir ../llm-defend-source/source
```

After preparing the full source data locally, phrase banks can be refreshed with:

```bash
python extract_and_bank_zh_phrases.py --input source/zh_lines.txt --bank_all
```

See [`../DATA_PROVENANCE.md`](../DATA_PROVENANCE.md) for source links,
license/access notes, mirror status, and transformation boundaries.
