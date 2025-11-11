"""
Utilities for building target pools and sampling external corpora.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import re
from multiprocessing import cpu_count
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from compose.audit import audit_soft
import compose.capabilities as _capabilities
from compose.carriers import maybe_wrap_again_named
from compose.dedupe_helpers import normalize
from compose.rng import stable_seed_int

# Lazy HF dataset handles are injected via compose.capabilities.
# Seed them here so _hf_loader() can safely reference the globals
# before the capability probe runs.
load_dataset: Optional[Callable[..., Any]] = None
DatasetDict: Optional[type] = None
hf_datasets: Optional[Any] = None

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_DIR = _REPO_ROOT / "source"

_DEFAULT_TARGET_FILES: List[Path] = [
    _SOURCE_DIR / "combined_prompts.jsonl",
    _SOURCE_DIR / "Safety_prompt_instruction_attack_scenarios.jsonl",
    _SOURCE_DIR / "Safety_prompt_typical_safety_scenarios.jsonl",
    _SOURCE_DIR / "MultiJail_zh.jsonl",
    _SOURCE_DIR / "JailBench.jsonl",
]

_PROMPT_KEYS = (
    "prompt",
    "text",
    "instruction",
    "question",
    "content",
    "input",
    "context",
    "query",
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_WILDCHAT_DATASETS: Tuple[Tuple[str, str], ...] = (
    ("allenai/WildChat-4.8M", "train"),
    ("allenai/WildChat", "train"),
    ("allenai/wildchat", "train"),
)
_WILDJAILBREAK_DATASETS: Tuple[Tuple[str, str], ...] = (
    ("ai-safety-lab/WildJailbreak", "train"),
    ("emozilla/WildJailbreak", "train"),
)
_JBB_DATASETS: Tuple[Tuple[str, str], ...] = (
    ("PKU-Alignment/JailbreakBench", "train"),
    ("PKU-Alignment/JailbreakBench-1.0", "train"),
)
_BEAVERTAILS_DATASETS: Tuple[Tuple[str, str], ...] = (
    ("PKU-Alignment/BeaverTails", "train"),
)


def _auto_io_workers(default: int = 0) -> int:
    """
    Heuristic for IO-heavy dataset loading parallelism.
    Mirrors the legacy behaviour so '--target_workers=-1' still auto-tunes.
    """
    if default and default > 0:
        return default
    cpu = max(2, cpu_count() or 2)
    base = 2 * cpu
    cap = int(os.getenv("TARGET_WORKERS_CAP", "16"))
    auto = max(2, min(base, cap))
    if os.getenv("HF_DATASETS_OFFLINE", "0") == "1":
        auto = min(auto, 4)
    return auto


def _hf_loader(force: bool = False):
    """
    Ensure that the HuggingFace datasets loader is initialised lazily.
    """
    global load_dataset
    loader = load_dataset
    if loader is None or force:
        if not _capabilities.ensure_hf_datasets(force=force):
            load_dataset = None
            return None
        loader = _capabilities.load_dataset
        load_dataset = loader
    return loader

def _try_load_dataset(
    name: str,
    params: Dict[str, Any],
    *,
    hf_rev: Optional[str] = None,
) -> Any:
    loader = _hf_loader()
    if loader is None:
        return None
    load_kwargs = dict(params)
    if hf_rev is not None and "revision" not in load_kwargs:
        load_kwargs["revision"] = hf_rev
    try:
        return loader(name, **load_kwargs)
    except Exception as exc:
        audit_soft("hf_target_load_error", exc, {"dataset": name, "params": load_kwargs})
        return None


def _load_many_datasets(
    specs: Sequence[Tuple[str, Dict[str, Any]]],
    hf_rev: Optional[str],
    workers: int,
) -> List[Tuple[Tuple[str, Dict[str, Any]], Any]]:
    if _hf_loader() is None:
        return [((name, kw), None) for name, kw in specs]
    workers = int(workers or 0)
    if workers <= 1:
        return [((name, kw), _try_load_dataset(name, kw, hf_rev=hf_rev)) for name, kw in specs]
    results: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Any] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_try_load_dataset, name, kw, hf_rev=hf_rev): (name, kw) for name, kw in specs
        }
        for fut in as_completed(future_map):
            name, kw = future_map[fut]
            key = (name, tuple(sorted(kw.items())))
            try:
                results[key] = fut.result()
            except Exception as exc:  # pragma: no cover - defensive
                audit_soft("hf_target_load_error", exc, {"dataset": name, "params": kw})
                results[key] = None
    return [((name, kw), results.get((name, tuple(sorted(kw.items()))))) for name, kw in specs]


def _coerce_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    absolute = (_REPO_ROOT / candidate).resolve()
    if absolute.exists():
        return absolute
    source_relative = (_SOURCE_DIR / candidate).resolve()
    if source_relative.exists():
        return source_relative
    return absolute


def _resolve_files(path: Path) -> List[Path]:
    if path.is_dir():
        files: List[Path] = []
        for pattern in ("*.jsonl", "*.json", "*.ndjson", "*.jsonl.gz", "*.json.gz", "*.ndjson.gz"):
            files.extend(sorted(path.glob(pattern)))
        return files
    return [path]


def _open_text(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_records_from_path(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    suffix = path.suffix.lower()
    try:
        with _open_text(path) as handle:
            if suffix in {".jsonl", ".ndjson"} or path.name.endswith(".jsonl.gz") or path.name.endswith(".ndjson.gz"):
                for idx, line in enumerate(handle, 1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError as exc:
                        audit_soft("offline_jsonl_parse_error", exc, {"path": str(path), "line": idx})
                        continue
                    if isinstance(obj, dict):
                        yield obj
                    else:
                        yield {"prompt": str(obj)}
            else:
                try:
                    data = json.load(handle)
                except json.JSONDecodeError as exc:
                    audit_soft("offline_json_parse_error", exc, {"path": str(path)})
                    return
                if isinstance(data, dict):
                    items = data.get("data") or data.get("items") or data.get("rows")
                    if isinstance(items, list):
                        for obj in items:
                            if isinstance(obj, dict):
                                yield obj
                    else:
                        yield data
                elif isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            yield obj
                else:
                    yield {"prompt": str(data)}
    except OSError as exc:
        audit_soft("offline_open_error", exc, {"path": str(path)})


def _collapse_messages(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    lines: List[str] = []
    for idx, item in enumerate(messages):
        role = f"turn{idx}"
        content: Any = ""
        if isinstance(item, dict):
            role = str(item.get("role") or item.get("speaker") or role)
            content = item.get("content") or item.get("text") or item.get("value")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            role = str(item[0])
            content = item[1]
        else:
            content = item
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        content = content.strip()
        if not content:
            continue
        lines.append(f"{role}: {content}" if role else content)
    return "\n".join(lines) if lines else None


def _extract_prompt(record: Dict[str, Any]) -> Optional[str]:
    # WildChat records expose explicit user/assistant fields; fold them first.
    user = ""
    assistant = ""
    for key in ("user_message", "human"):
        value = record.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                user = value
                break
    for key in ("assistant_response", "assistant"):
        value = record.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                assistant = value
                break
    if user or assistant:
        combined = user + ("\n" if user and assistant else "") + assistant
        combined = combined.strip()
        if combined:
            return combined
    for key in _PROMPT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    msg_text = _collapse_messages(record.get("messages") or record.get("dialog") or record.get("conversation"))
    if msg_text:
        return msg_text
    pair_fields: Tuple[Tuple[str, str], ...] = (
        ("sentence1", "sentence2"),
        ("premise", "hypothesis"),
        ("text1", "text2"),
        ("query1", "query2"),
        ("question1", "question2"),
    )
    for left_key, right_key in pair_fields:
        left = record.get(left_key)
        right = record.get(right_key)
        if isinstance(left, str) and isinstance(right, str):
            left = left.strip()
            right = right.strip()
            if left and right:
                return f"{left_key}: {left}\n{right_key}: {right}"
    singleton = record.get("sentence") or record.get("passage")
    if isinstance(singleton, str) and singleton.strip():
        return singleton.strip()
    return None


def _normalize_task(record: Dict[str, Any]) -> str:
    task = record.get("task")
    if isinstance(task, list):
        task = next((t for t in task if isinstance(t, str) and t.strip()), None)
    if isinstance(task, str):
        task = task.strip()
    elif task is None:
        task = ""
    else:
        task = str(task).strip()
    if not task:
        fallback = record.get("category") or record.get("label") or record.get("intent") or record.get("topic")
        if isinstance(fallback, list):
            fallback = next((t for t in fallback if isinstance(t, str) and t.strip()), None)
        if isinstance(fallback, str):
            task = fallback.strip()
        elif fallback is not None:
            task = str(fallback).strip()
    return task or "task_general"


def _normalize_target(record: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    prompt = _extract_prompt(record)
    if not prompt:
        return None
    task = _normalize_task(record)
    keep_keys = set(record.keys()) - {
        "prompt",
        "text",
        "instruction",
        "question",
        "content",
        "input",
        "context",
        "messages",
        "dialog",
        "conversation",
        "task",
        "label",
        "category",
        "intent",
        "topic",
        "query",
    }
    meta = {k: record[k] for k in keep_keys} if keep_keys else {}
    normalized: Dict[str, Any] = {"prompt": prompt.strip(), "task": task, "source": source}
    if meta:
        normalized["meta"] = meta
    return normalized


_PROJECT_TITLES: Tuple[str, ...] = ("[Context]", "[Input]", "【上下文】", "【材料】")


def _benign_allowed_tool_json(rng: random.Random) -> str:
    name = rng.choice(["search", "summarize", "classify", "extract"])
    obj = {"tool": name, "args": {"q": "...", "top_k": 3}, "out": {"note": "ok"}}
    return "```json\n" + json.dumps(obj, ensure_ascii=False) + "\n```"


def _project_shape(raw: dict, shape: str, rng: random.Random) -> dict:
    text = str(raw.get("text", ""))
    label = int(raw.get("label", 0))
    source = raw.get("source", "ext")
    shape_norm = (shape or "in_context").strip() or "in_context"
    if shape_norm == "in_context":
        projected = text
    elif shape_norm == "format_carrier":
        _, wrapped = maybe_wrap_again_named(text, rng=rng)
        projected = f"{rng.choice(_PROJECT_TITLES)}\n{wrapped}"
    elif shape_norm == "rag_doc":
        header = f"---\nsource: {source}\nrole: user\nupstream: Ref-Doc-{rng.randint(1, 9999)}\n---\n"
        projected = header + text + ("\n" if not text.endswith("\n") else "")
    elif shape_norm == "multi_turn":
        lead = text.split("\n")[0][:60] or "这里有个情况先确认"
        convo = [
            ("用户", lead),
            ("助手", "收到，我先理解要点。"),
            ("用户", "好，请保持原格式输出。"),
        ]
        projected = "\n".join(f"{role}: {utterance}" for role, utterance in convo)
    elif shape_norm == "tool_override":
        projected = _benign_allowed_tool_json(rng) + "\n" + text
    else:
        projected = text
    return {"text": projected, "label": label, "source": source, "shape": shape_norm}


def load_targets_offline(
    path: str | Path,
    *,
    limit: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
    dedupe: bool = True,
    filters: Optional[Iterable[Callable[[Dict[str, Any]], bool]] | Callable[[Dict[str, Any]], bool]] = None,
) -> List[Dict[str, Any]]:
    target_path = _coerce_path(path)
    files = _resolve_files(target_path)
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL files found at {path}")
    rng = random.Random(seed)
    if filters is None:
        filter_funcs: List[Callable[[Dict[str, Any]], bool]] = []
    elif callable(filters):
        filter_funcs = [filters]  # type: ignore[list-item]
    else:
        filter_funcs = [fn for fn in filters if callable(fn)]  # type: ignore[arg-type]
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for file_path in files:
        if not file_path.exists():
            continue
        for raw in _iter_records_from_path(file_path):
            normalized = _normalize_target(raw, source=file_path.stem)
            if not normalized:
                continue
            if filter_funcs and not all(fn(normalized) for fn in filter_funcs):
                continue
            if dedupe:
                key = normalized["prompt"]
                if key in seen:
                    continue
                seen.add(key)
            rows.append(normalized)
    if shuffle:
        rng.shuffle(rows)
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def _default_hf_dataset_specs() -> List[str]:
    env = os.getenv("COMPOSE_TARGET_DATASETS", "").strip()
    if not env:
        return []
    return [chunk.strip() for chunk in env.split(",") if chunk.strip()]


def _parse_dataset_spec(spec: str) -> Tuple[str, str]:
    dataset = spec.strip()
    if not dataset:
        raise ValueError("Empty dataset spec")
    if ":" in dataset:
        name, split = dataset.split(":", 1)
        return name.strip(), (split.strip() or "train")
    return dataset, "train"


def _collect_hf_targets(
    specs: Sequence[str],
    *,
    seed: int,
    hf_rev: Optional[str],
    limit: Optional[int],
    workers: int = 1,
) -> List[Dict[str, Any]]:
    if os.getenv("HF_DATASETS_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        return []
    loader = _hf_loader()
    if loader is None:
        return []
    rng = random.Random(seed)
    collected: List[Dict[str, Any]] = []

    entries: List[Tuple[int, str, str]] = []
    entry_lookup: Dict[int, Tuple[str, str]] = {}
    for idx, spec in enumerate(specs):
        try:
            dataset_name, split = _parse_dataset_spec(spec)
        except ValueError:
            continue
        entries.append((idx, dataset_name, split))
        entry_lookup[idx] = (dataset_name, split)
    if not entries:
        return collected

    def _process_dataset(idx: int, dataset_name: str, split: str, per_dataset_limit: Optional[int]) -> List[Dict[str, Any]]:
        try:
            dataset = loader(dataset_name, split=split, revision=hf_rev)
        except Exception as exc:
            audit_soft("hf_target_load_error", exc, {"dataset": dataset_name, "split": split})
            return []
        if hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=seed + idx)
        rows: List[Dict[str, Any]] = []
        for row in dataset:
            normalized = _normalize_target(row, source=dataset_name)
            if not normalized:
                continue
            rows.append(normalized)
            if per_dataset_limit is not None and len(rows) >= per_dataset_limit:
                break
        return rows

    per_dataset_limit = limit if limit is not None else None
    workers_int = int(workers or 0)
    if len(entries) == 0:
        return collected
    if workers_int <= 0:
        worker_count = _auto_io_workers()
    else:
        worker_count = workers_int
    worker_count = max(1, min(worker_count, len(entries)))

    if worker_count <= 1:
        for idx, dataset_name, split in entries:
            rows = _process_dataset(idx, dataset_name, split, per_dataset_limit)
            for row in rows:
                if limit is not None and len(collected) >= limit:
                    return collected
                collected.append(row)
            rng.shuffle(collected)
            if limit is not None and len(collected) >= limit:
                return collected[:limit]
        return collected

    results: List[Tuple[int, List[Dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_process_dataset, idx, dataset_name, split, per_dataset_limit): idx
            for idx, dataset_name, split in entries
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                rows = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                dataset_name, split = entry_lookup.get(idx, ("<unknown>", "<unknown>"))
                audit_soft("hf_target_load_error", exc, {"dataset": dataset_name, "split": split})
                rows = []
            results.append((idx, rows))

    for idx, rows in sorted(results, key=lambda item: item[0]):
        for row in rows:
            if limit is not None and len(collected) >= limit:
                return collected
            collected.append(row)
        rng.shuffle(collected)
        if limit is not None and len(collected) >= limit:
            return collected[:limit]
    return collected


def build_target_pool(
    seed: int,
    hf_rev: Optional[str] = None,
    workers: int = -1,
    limit: Optional[int] = None,
    dataset_specs: Optional[Sequence[str]] = None,
    offline_paths: Optional[Sequence[str | Path]] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pool: List[Dict[str, Any]] = []
    seen: set[str] = set()
    paths = list(offline_paths or _DEFAULT_TARGET_FILES)
    present_files: List[Path] = []
    missing_files: List[Path] = []
    for raw_path in paths:
        path = _coerce_path(raw_path)
        if not path.exists():
            missing_files.append(path)
            continue
        present_files.append(path)
        try:
            rows = load_targets_offline(path, seed=rng.randint(0, 1_000_000), shuffle=True, dedupe=True)
        except FileNotFoundError:
            continue
        for row in rows:
            key = row["prompt"]
            if key in seen:
                continue
            seen.add(key)
            pool.append(row)
    hf_specs = dataset_specs or _default_hf_dataset_specs()
    remaining = None if limit is None else max(0, limit - len(pool))
    if remaining is None or remaining > 0:
        curated_rows = _collect_curated_targets(
            seed=seed,
            hf_rev=hf_rev,
            workers=workers,
            limit=remaining,
        )
        for row in curated_rows:
            key = row["prompt"]
            if key in seen:
                continue
            seen.add(key)
            pool.append(row)
            if limit is not None and len(pool) >= limit:
                break
    if hf_specs:
        remaining = None if limit is None else max(0, limit - len(pool))
        hf_rows = _collect_hf_targets(
            hf_specs,
            seed=seed + 97,
            hf_rev=hf_rev,
            limit=remaining,
            workers=workers,
        )
        for row in hf_rows:
            key = row["prompt"]
            if key in seen:
                continue
            seen.add(key)
            pool.append(row)
    if limit is not None and limit >= 0 and len(pool) > limit:
        rng.shuffle(pool)
        pool = pool[:limit]
    if present_files or missing_files:
        summary = {
            "found": [str(path) for path in present_files],
            "missing": [str(path) for path in missing_files],
            "hf_specs": list(hf_specs),
        }
        print("[target_pool][local_files]", summary)
    return pool


def _lang_matches(text: str, record: Dict[str, Any], lang: str) -> bool:
    if not lang or lang.lower() in {"*", "all"}:
        return True
    lang_lower = lang.lower()
    for key in ("lang", "language", "languages", "locale"):
        value = record.get(key)
        if isinstance(value, str):
            return lang_lower in value.lower()
        if isinstance(value, list):
            joined = ",".join(str(x).lower() for x in value if isinstance(x, str))
            return lang_lower in joined
    if lang_lower in {"code_switch", "code-switch", "zh_en", "zh-en"}:
        return bool(_CJK_RE.search(text) and _LATIN_RE.search(text))
    if lang_lower.startswith("zh"):
        return bool(_CJK_RE.search(text))
    return True


def _sample_rows(
    rows: Iterable[Dict[str, Any]],
    n: Optional[int],
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    data = list(rows)
    if not data:
        return []
    rng = rng or random.Random()
    if n is None or n <= 0 or n >= len(data):
        data_copy = data[:]
        rng.shuffle(data_copy)
        return data_copy
    try:
        return rng.sample(data, min(n, len(data)))
    except ValueError:
        data_copy = data[:]
        rng.shuffle(data_copy)
        return data_copy[:n]


def _load_local_targets(paths: Sequence[str | Path], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    bag: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = _coerce_path(raw_path)
        if not path.exists():
            continue
        try:
            rows = load_targets_offline(path, seed=rng.randint(0, 1_000_000), shuffle=True, dedupe=True)
        except FileNotFoundError:
            continue
        for row in rows:
            key = row["prompt"]
            if key in seen:
                continue
            seen.add(key)
            bag.append(row)
    rng.shuffle(bag)
    return bag


def _yield_text_fields(row: Dict[str, Any]) -> Iterable[str]:
    keys = (
        "text",
        "content",
        "passage",
        "article",
        "paragraph",
        "sentence",
        "document",
        "wiki",
        "section",
        "title",
        "body",
        "summary",
        "answer",
        "question",
    )
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            yield value


def _iter_dataset_rows(dataset: Any, tag: str, seed: int, max_rows: int) -> Iterable[Dict[str, Any]]:
    if dataset is None:
        return []
    if hasattr(dataset, "keys"):
        splits = list(dataset.keys())
        for split in splits:
            subset = dataset[split]
            sample_seed = stable_seed_int("target_pool", tag, split, seed)
            rng = random.Random(sample_seed)
            for row in _sample_rows(subset, max_rows, rng=rng):
                yield row
    else:
        sample_seed = stable_seed_int("target_pool", tag, "train", seed)
        rng = random.Random(sample_seed)
        for row in _sample_rows(dataset, max_rows, rng=rng):
            yield row


def _collect_curated_targets(
    *,
    seed: int,
    hf_rev: Optional[str],
    workers: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    if os.getenv("HF_DATASETS_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        return []
    rng = random.Random(seed)
    loader_workers = _auto_io_workers(workers)
    max_per_task = 4000
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _append(record: Optional[Dict[str, Any]]) -> bool:
        if not record:
            return False
        prompt = record.get("prompt")
        if not prompt or prompt in seen:
            return False
        seen.add(prompt)
        results.append(record)
        return bool(limit is not None and len(results) >= limit)

    def _maybe_stop() -> bool:
        return bool(limit is not None and len(results) >= limit)

    def _load_with_fallback(candidates: Sequence[Tuple[str, Dict[str, Any]]]) -> Tuple[Any, str]:
        for name, params in candidates:
            ds = _try_load_dataset(name, params, hf_rev=hf_rev)
            if ds is not None:
                return ds, name
        return None, ""

    # Similarity (LCQMC/AFQMC)
    classic_groups: Tuple[Tuple[Tuple[str, Dict[str, Any]], Tuple[Tuple[str, Dict[str, Any]], ...]], ...] = (
        (("C-MTEB/LCQMC", {}), (("mteb/LCQMC", {}),)),
        (("C-MTEB/AFQMC", {}), (("clue", {"name": "afqmc"}),)),
    )
    for primary, fallbacks in classic_groups:
        candidates = (primary,) + fallbacks
        ds, loaded_name = _load_with_fallback(candidates)
        if not ds:
            continue
        tag = f"classic-{loaded_name}"
        for row in _iter_dataset_rows(ds, tag, seed, max_per_task):
            a = row.get("sentence1") or row.get("text1") or row.get("query1") or row.get("sentence")
            b = row.get("sentence2") or row.get("text2") or row.get("query2")
            if not (isinstance(a, str) and isinstance(b, str)):
                continue
            a = normalize(a)
            b = normalize(b)
            if len(a) < 4 or len(b) < 4:
                continue
            prompt = f"判断两句话是否表达同一含义，仅答“是/否”。\n句子A：{a}\n句子B：{b}"
            if _append({"task": "similarity", "prompt": prompt}):
                return results
        if _maybe_stop():
            return results

    # NLI (OCNLI/CMNLI)
    for sub in ("ocnli", "cmnli"):
        ds = _try_load_dataset("clue", {"name": sub}, hf_rev=hf_rev)
        if not ds:
            continue
        tag = f"nli-{sub}"
        for row in _iter_dataset_rows(ds, tag, seed, max_per_task):
            premise = row.get("sentence1") or row.get("premise")
            hypothesis = row.get("sentence2") or row.get("hypothesis")
            if not (isinstance(premise, str) and isinstance(hypothesis, str)):
                continue
            premise = normalize(premise)
            hypothesis = normalize(hypothesis)
            if len(premise) < 4 or len(hypothesis) < 4:
                continue
            prompt = (
                "判断前提与假设的关系：只答“蕴含/矛盾/中立”。\n"
                f"前提：{premise}\n假设：{hypothesis}"
            )
            if _append({"task": "nli", "prompt": prompt}):
                return results
        if _maybe_stop():
            return results

    # Sentiment
    sentiment_specs = (("lansinuote/ChnSentiCorp", {}),)
    for (name, kw), ds in _load_many_datasets(sentiment_specs, hf_rev, loader_workers):
        if ds is None:
            continue
        tag = f"sentiment-{name}"
        for row in _iter_dataset_rows(ds, tag, seed, max_per_task):
            text = row.get("text") or row.get("sentence") or row.get("content")
            if not isinstance(text, str):
                continue
            text = normalize(text)
            if len(text) < 8:
                continue
            prompt = f"判断情感倾向，仅答“正/负”。\n{text}"
            if _append({"task": "sentiment", "prompt": prompt}):
                return results
        if _maybe_stop():
            return results

    # Summarization (LCSTS)
    lcsts_specs = (("hugcyp/LCSTS", {}), ("suolyer/lcsts", {}))
    for (name, kw), ds in _load_many_datasets(lcsts_specs, hf_rev, loader_workers):
        if ds is None:
            continue
        tag = f"lcsts-{name}"
        for row in _iter_dataset_rows(ds, tag, seed, max_per_task):
            text = (
                row.get("text")
                or row.get("content")
                or row.get("passage")
                or row.get("summary")
                or row.get("Document")
            )
            if not isinstance(text, str):
                continue
            text = normalize(text)
            if len(text) < 20:
                continue
            prompt = f"请概括下文，生成≤30字摘要：\n{text}"
            if _append({"task": "summarization", "prompt": prompt}):
                return results
        if _maybe_stop():
            return results

    # Wikipedia style corpus
    wiki_specs = (("wikimedia/wikipedia", {"name": "20231101.zh"}), ("shaowenchen/wiki_zh", {}))
    for (name, kw), ds in _load_many_datasets(wiki_specs, hf_rev, loader_workers):
        if ds is None:
            continue
        tag = f"wiki-{name}"
        for row in _iter_dataset_rows(ds, tag, seed, max_per_task):
            segments = list(_yield_text_fields(row))
            if not segments:
                continue
            text = normalize("。".join(segments)[:800])
            if len(text) < 60:
                continue
            if _append({"task": "wiki_summarize", "prompt": f"阅读片段并给出50字内要点：\n{text}"}):
                return results
            if _append({"task": "wiki_ents", "prompt": f"从片段中抽取专有名词并以逗号分隔：\n{text}"}):
                return results
        if _maybe_stop():
            return results

    # Chinese news / forum style corpora
    news_specs = (("SirlyDreamer/THUCNews", {}),)
    for (name, kw), ds in _load_many_datasets(news_specs, hf_rev=None, workers=loader_workers):
        if ds is None:
            continue
        tag = f"news-{name}"
        for row in _iter_dataset_rows(ds, tag, seed, max_per_task):
            for text in _yield_text_fields(row):
                text_norm = normalize(text)
                if len(text_norm) < 60:
                    continue
                task = rng.choice(("extract_kv", "classify_topic", "clean_markup", "make_outline"))
                if task == "extract_kv":
                    prompt = f"抽取段落中的关键实体（人/地/组织），JSON返回：\n{text_norm}"
                elif task == "classify_topic":
                    prompt = f"判断该段是“新闻/博客/论坛/说明/评论”中的一类，给出一项\n{text_norm}"
                elif task == "clean_markup":
                    prompt = f"去除段落中的标记/括注并输出整洁文本\n{text_norm}"
                else:
                    prompt = f"根据段落生成提纲，1-3条要点\n{text_norm}"
                if _append({"task": task, "prompt": prompt}):
                    return results
                break  # one prompt per source row
        if _maybe_stop():
            return results

    rng.shuffle(results)
    return results


def _collect_from_hf_candidates(
    candidates: Sequence[Tuple[str, str]],
    *,
    n: Optional[int],
    seed: int,
    label: int,
    lang: Optional[str] = None,
    revision: Optional[str] = None,
) -> List[Dict[str, Any]]:
    loader = _hf_loader()
    if loader is None:
        return []
    results: List[Dict[str, Any]] = []

    def _coerce_label(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            if value in (0, 1):
                return int(value)
            return None
        if isinstance(value, str):
            s = value.strip().lower()
            if not s:
                return None
            if s in {"0", "false", "benign", "harmless", "safe", "non_harmful", "non-harmful"}:
                return 0
            if s in {"1", "true", "harmful", "malicious", "jailbreak"}:
                return 1
        return None

    for idx, (dataset_name, split) in enumerate(candidates):
        try:
            load_kwargs = {"split": split}
            if revision is not None:
                load_kwargs["revision"] = revision
            dataset = loader(dataset_name, **load_kwargs)
        except Exception as exc:
            audit_soft("external_dataset_load_error", exc, {"dataset": dataset_name, "split": split})
            continue
        if hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=seed + idx)
        for row in dataset:
            text = _extract_prompt(row)
            if not text:
                continue
            if lang and not _lang_matches(text, row, lang):
                continue
            eff_label = label
            # Prefer dataset-supplied annotations (e.g. WildJailbreak marks benign rows).
            derived_label = _coerce_label(row.get("label"))
            if derived_label is None and "wildjailbreak" in str(dataset_name).lower():
                derived_label = _coerce_label(row.get("harmful"))
            if derived_label is not None:
                eff_label = derived_label
            record: Dict[str, Any] = {"text": text, "label": eff_label, "source": dataset_name}
            task = row.get("task")
            if isinstance(task, list):
                task = next((t for t in task if isinstance(t, str) and t.strip()), None)
            if isinstance(task, str) and task.strip():
                record["meta"] = {"task": task.strip()}
            results.append(record)
            if n is not None and n > 0 and len(results) >= n:
                return results
        if n is not None and n > 0 and len(results) >= n:
            break
    return results


def _yield_wildchat(
    *,
    lang: str = "zh",
    n: Optional[int] = None,
    seed: int = 0,
    rng: Optional[random.Random] = None,
    hf_revision: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    rng = rng or random.Random(seed)
    produced = 0
    hf_rows = _collect_from_hf_candidates(
        _WILDCHAT_DATASETS, n=n, seed=seed, label=0, lang=lang, revision=hf_revision
    )
    for row in hf_rows:
        yield row
        produced += 1
    remaining = None
    if n is not None and n > 0:
        remaining = max(0, n - produced)
    if remaining == 0:
        return
    fallback = _load_local_targets(_DEFAULT_TARGET_FILES[:1], seed=seed + 7)
    if not fallback:
        return
    for row in _sample_rows(fallback, remaining, rng):
        rec = {"text": row["prompt"], "label": 0, "source": "wildchat_fallback", "meta": {"task": row["task"]}}
        yield rec


def _yield_wildjailbreak(
    *,
    lang: str = "zh",
    n: Optional[int] = None,
    seed: int = 0,
    rng: Optional[random.Random] = None,
    hf_revision: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    rng = rng or random.Random(seed)
    produced = 0
    hf_rows = _collect_from_hf_candidates(
        _WILDJAILBREAK_DATASETS, n=n, seed=seed, label=1, lang=lang, revision=hf_revision
    )
    for row in hf_rows:
        yield row
        produced += 1
    remaining = None
    if n is not None and n > 0:
        remaining = max(0, n - produced)
    if remaining == 0:
        return
    fallback_paths: List[Path] = [
        _SOURCE_DIR / "MultiJail_zh.jsonl",
    ]
    fallback = _load_local_targets(fallback_paths, seed=seed + 13)
    if not fallback:
        return
    for row in _sample_rows(fallback, remaining, rng):
        yield {"text": row["prompt"], "label": 1, "source": "wildjailbreak_fallback", "meta": {"task": row["task"]}}


def _yield_jbb_behaviors(
    *,
    lang: str = "zh",
    n: Optional[int] = None,
    seed: int = 0,
    rng: Optional[random.Random] = None,
    hf_revision: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    rng = rng or random.Random(seed)
    produced = 0
    hf_rows = _collect_from_hf_candidates(
        _JBB_DATASETS, n=n, seed=seed, label=1, lang=lang, revision=hf_revision
    )
    for row in hf_rows:
        yield row
        produced += 1
    remaining = None
    if n is not None and n > 0:
        remaining = max(0, n - produced)
    if remaining == 0:
        return
    fallback = _load_local_targets([_SOURCE_DIR / "JailBench.jsonl"], seed=seed + 19)
    if not fallback:
        return
    for row in _sample_rows(fallback, remaining, rng):
        yield {"text": row["prompt"], "label": 1, "source": "jbb_fallback", "meta": {"task": row["task"]}}


def _yield_beavertails(
    *,
    lang: str = "zh",
    n: Optional[int] = None,
    seed: int = 0,
    rng: Optional[random.Random] = None,
    hf_revision: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    rng = rng or random.Random(seed)
    produced = 0
    hf_rows = _collect_from_hf_candidates(
        _BEAVERTAILS_DATASETS, n=n, seed=seed, label=1, lang=lang, revision=hf_revision
    )
    for row in hf_rows:
        yield row
        produced += 1
    remaining = None
    if n is not None and n > 0:
        remaining = max(0, n - produced)
    if remaining == 0:
        return
    fallback = _load_local_targets([], seed=seed + 23)
    if not fallback:
        return
    for row in _sample_rows(fallback, remaining, rng):
        yield {"text": row["prompt"], "label": 1, "source": "beavertails_fallback", "meta": {"task": row["task"]}}


def _yield_pint(
    *,
    n: Optional[int] = None,
    seed: int = 0,
    rng: Optional[random.Random] = None,
    hf_revision: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    rng = rng or random.Random(seed)
    produced = 0
    dataset: Any = None
    loader = _hf_loader()
    if loader is not None:
        for candidate in ("lakeraai/pint-benchmark", "lakeraai/PINT", "lakera/PINT"):
            try:
                dataset = loader(candidate, split="train", revision=hf_revision)
                break
            except Exception:
                dataset = None
    if dataset is None:
        path = os.getenv("PINT_PATH")
        if path and os.path.exists(path):
            ext = Path(path).suffix.lower()
            try:
                if ext in {".jsonl", ".ndjson"}:
                    with open(path, "r", encoding="utf-8") as handle:
                        dataset = [json.loads(line) for line in handle if line.strip()]
                elif ext == ".json":
                    with open(path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    if isinstance(payload, list):
                        dataset = payload
                    elif isinstance(payload, dict):
                        dataset = payload.get("data") or payload.get("items")
            except Exception as exc:
                audit_soft("pint_local_load_error", exc, {"path": path})
                dataset = None
    if dataset is None:
        return
    iterable = dataset.shuffle(seed=seed) if hasattr(dataset, "shuffle") else dataset
    for row in iterable:
        text = ""
        label_value = None
        if isinstance(row, dict):
            text = row.get("prompt") or row.get("text") or row.get("input") or ""
            label_value = row.get("label")
        if not isinstance(text, str) or not text.strip():
            continue
        text_norm = normalize(text)
        if not text_norm:
            continue
        label_int = 0
        if isinstance(label_value, (int, float)) and label_value in (0, 1):
            label_int = int(label_value)
        elif isinstance(label_value, str):
            label_lower = label_value.strip().lower()
            if label_lower in {"1", "true", "harmful", "unsafe"}:
                label_int = 1
        yield {"text": text_norm, "label": label_int, "source": "pint"}
        produced += 1
        if n is not None and n > 0 and produced >= n:
            break


def _parse_external_token(token: str) -> Tuple[str, Optional[str]]:
    name = token.strip().lower()
    if ":" in name:
        base, arg = name.split(":", 1)
        return base.strip(), (arg.strip() or None)
    return name, None


def _alloc_counts(total: int, parts: int) -> List[int]:
    if total <= 0 or parts <= 0:
        return []
    base = total // parts
    remainder = total % parts
    counts = [base] * parts
    for idx in range(remainder):
        counts[idx] += 1
    return counts


def iter_external_corpora(
    spec: str,
    *,
    lang: str = "zh",
    total: int = 0,
    seed: int = 0,
    hf_revision: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    tokens = [tok.strip() for tok in spec.split(",") if tok.strip()]
    if not tokens:
        return iter(())
    rng = random.Random(seed)
    per_source = _alloc_counts(total, len(tokens)) if total else []
    registry: Dict[str, Callable[..., Iterator[Dict[str, Any]]]] = {
        "wildchat": _yield_wildchat,
        "wildjailbreak": _yield_wildjailbreak,
        "jbb": _yield_jbb_behaviors,
        "jbb_behaviors": _yield_jbb_behaviors,
        "jailbreakbench": _yield_jbb_behaviors,
        "beavertails": _yield_beavertails,
        "pint": _yield_pint,
    }

    def _iter() -> Iterator[Dict[str, Any]]:
        for idx, token in enumerate(tokens):
            name, arg = _parse_external_token(token)
            provider = registry.get(name)
            if provider is None:
                audit_soft("external_source_unknown", ValueError(f"Unknown external corpus '{name}'"), {"token": token})
                continue
            limit = per_source[idx] if per_source else None
            lang_choice = arg or lang
            produced = 0
            try:
                for row in provider(
                    lang=lang_choice,
                    n=limit,
                    seed=rng.randint(0, 2_147_483_647),
                    rng=rng,
                    hf_revision=hf_revision,
                ):
                    yield row
                    produced += 1
                    if limit is not None and limit > 0 and produced >= limit:
                        break
            except Exception as exc:
                audit_soft("external_source_error", exc, {"source": name})

    return _iter()


__all__ = [
    "build_target_pool",
    "load_targets_offline",
    "iter_external_corpora",
]
