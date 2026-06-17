from datasets import load_dataset

# Load the Chinese split from the upstream Hugging Face dataset.
ds = load_dataset("walledai/MultiJail", split="zh")

# Keep compatibility with datasets that expose language labels as `language` or `lang`.
name_cols = ds.column_names
lang_col = "language" if "language" in name_cols else ("lang" if "lang" in name_cols else None)
zh = ds if lang_col is None else ds.filter(lambda x: x[lang_col] == "zh")
zh.to_json("MultiJail_zh.jsonl", lines=True, force_ascii=False)
