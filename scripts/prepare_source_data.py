#!/usr/bin/env python3
"""Prepare local source data for llm-defend experiments.

The public repository intentionally does not mirror full third-party or derived
prompt corpora. This helper copies an explicitly supplied private/exported
source directory into the local `source/` tree for reviewers or experiments
that already have permission to access those files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


EXPECTED_SOURCE_FILES = (
    "combined_prompts.jsonl",
    "zh_lines.txt",
    "JailBench.csv",
    "JailBench.jsonl",
    "MultiJail_zh.jsonl",
    "Safety_prompt_instruction_attack_scenarios.jsonl",
    "Safety_prompt_typical_safety_scenarios.jsonl",
    "datasets/JailJudge/JAILJUDGE_ID.json",
    "datasets/JailJudge/JAILJUDGE_OOD.json",
    "datasets/JailJudge/JAILJUDGE_OOD_zh.jsonl",
    "datasets/JailJudge/JAILJUDGE_OOD_zh_task.jsonl",
    "datasets/JailJudge/JAILJUDGE_TRAIN.json",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy_file(src_root: Path, dst_root: Path, rel_path: str, overwrite: bool) -> str:
    src = src_root / rel_path
    dst = dst_root / rel_path
    if not src.exists():
        return "missing"
    if dst.exists() and not overwrite:
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy an access-controlled llm-defend source-data export into the local source tree."
    )
    parser.add_argument(
        "--private-source",
        default=None,
        help="Path to a private/exported source directory. Defaults to LLM_DEFEND_PRIVATE_SOURCE if set.",
    )
    parser.add_argument(
        "--dest",
        default="source",
        help="Destination source directory relative to the repository root. Defaults to source/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local source files.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report which expected files are available in --private-source.",
    )
    args = parser.parse_args()

    import os

    source_arg = args.private_source or os.getenv("LLM_DEFEND_PRIVATE_SOURCE")
    if not source_arg:
        parser.error("provide --private-source or set LLM_DEFEND_PRIVATE_SOURCE")

    src_root = Path(source_arg).expanduser().resolve()
    dst_root = (_repo_root() / args.dest).resolve()

    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"Private source directory not found: {src_root}")

    counts = {"available": 0, "copied": 0, "exists": 0, "missing": 0}
    for rel_path in EXPECTED_SOURCE_FILES:
        if args.check_only:
            status = "available" if (src_root / rel_path).exists() else "missing"
        else:
            status = _copy_file(src_root, dst_root, rel_path, args.overwrite)
        counts[status] += 1
        print(f"{status:>7} {rel_path}")

    if counts["missing"]:
        print(f"Missing {counts['missing']} expected file(s); inspect upstream access and export steps.")
        return 1
    if args.check_only:
        print(f"Validated source export at {src_root}")
    else:
        print(f"Prepared source data in {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
