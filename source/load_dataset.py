from datasets import load_dataset

# 载入数据（全部行）
ds = load_dataset("walledai/MultiJail", split="zh")  # 若报 split 错，可去掉 split 参数再试

# 过滤出中文（列名有时叫 language 或 lang，两种都兼容）
name_cols = ds.column_names
lang_col = "language" if "language" in name_cols else ("lang" if "lang" in name_cols else None)
zh = ds if lang_col is None else ds.filter(lambda x: x[lang_col] == "zh")
zh.to_json("MultiJail_zh.jsonl", lines=True, force_ascii=False)
