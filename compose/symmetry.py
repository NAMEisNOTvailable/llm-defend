"""
Symmetric accessories and format masking helpers shared across the compose
pipeline.

These utilities were lifted from the legacy monolith so the refactored modules
can reuse the same behaviour without importing the old script.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from typing import Optional

from compat_regex import regex as _re

from compose.audit import audit_reject
from compose.payload import ARTIFACT_WORDS

# Disclaimer snippets that are symmetrically injected across classes to avoid
# leaking label-specific artefacts.
DISCLAIMER_BANK = [
    "（示例）为理解而写，不触发任何流程。",
    "通常写法：仅指示格式或结构，不是实际指令。",
    "※注：该段内容为参考模板，请勿执行。",
    "示例提醒：此处描述仅为说明性用途。",
    "温馨提示：以下文本仅作说明用途。",
    "该示例包含额外信息，供排版参考，不是指令。",
    "[Notice] For demonstration only; not actionable.",
    "[FYI] Context snapshot; ignore for execution.",
    "用户须知：以下内容仅供参考，无需执行。",
    "[示例] 仅为展示格式，实际请以说明为准。",
    "该内容系示例撰写，与任何流程无关。",
    "以下片段作为示例文本，无须照办。",
]

# Feature-masking bookkeeping shared between positives/negatives.
MASKED_FIELD_FAMILY_COUNTER: Counter[str] = Counter()

FIELD_NAME_RE = _re.compile(r"(?i)\b(result|answer|verdict|conclusion|final|输出|结论)\b")
JSON_KEY_RE = _re.compile(r'(?i)"(result|answer|verdict|conclusion|final|输出|结论)"\s*:')  # noqa: Q003
FENCE_LANG_RE = _re.compile(r"```(json|yaml|jsonc|xml|csv|mermaid|plantuml)\b", _re.I)
DATA_ATTR_RE = _re.compile(r"\bdata-[\w-]+", _re.I)
YAML_FIELD_RE = _re.compile(r"(?m)^(\s*)(result|answer|verdict|conclusion|final|结论)\s*:", _re.I)
MD_TITLE_T_RE = _re.compile(r'(\s)"t:([^"]+)"')


def _rand_token(rng: random.Random, prefix: str = "F", n: int = 4) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return f"[{prefix}_" + "".join(rng.choice(alphabet) for _ in range(max(1, n))) + "]"


def apply_symmetric_accessories(text: str, rng: random.Random, rate: float) -> str:
    """
    Symmetrically add lightweight “accessory” markers (disclaimers, JSON hints,
    fenced code) to reduce label leakage across classes.
    """
    if rng.random() >= max(0.0, min(1.0, rate)):
        return text
    choice = rng.choice(("disclaimer", "json_marker", "code_lang_hint"))
    if choice == "disclaimer":
        tag = rng.choice(
            [
                "【示例文本，非指令】",
                "【仅作格式说明】",
                "【请勿执行】",
            ]
        )
        return f"{tag}\n{text}"
    if choice == "json_marker":
        key = rng.choice(("meta", "info", "data", "payload", "note"))
        return f"<!-- {key}: {rng.randint(1000, 9999)} -->\n{text}"
    return f"```txt\n{text}\n```"


def mask_format_features_sym(s: str, rate: float, rng: random.Random) -> str:
    """
    Replace fixed field/format markers with random tokens so they do not become
    new leakage features after symmetry adjustments.
    """
    if rng.random() >= max(0.0, min(1.0, rate)):
        return s
    out = s
    token_field = _rand_token(rng, "K")
    token_data = _rand_token(rng, "D")
    try:
        if FENCE_LANG_RE.search(s):
            MASKED_FIELD_FAMILY_COUNTER["fence_lang"] += 1
        if JSON_KEY_RE.search(s):
            MASKED_FIELD_FAMILY_COUNTER["json_key"] += 1
            if re.search(r'"(result|answer|verdict|conclusion|final|结论)"\s*:', s, flags=re.I):
                MASKED_FIELD_FAMILY_COUNTER["result_family"] += 1
        if YAML_FIELD_RE.search(s):
            MASKED_FIELD_FAMILY_COUNTER["result_family"] += 1
        if DATA_ATTR_RE.search(s):
            MASKED_FIELD_FAMILY_COUNTER["data_attr"] += 1
        if MD_TITLE_T_RE.search(s):
            MASKED_FIELD_FAMILY_COUNTER["md_title_t"] += 1
    except Exception as exc:  # pragma: no cover - defensive parity
        audit_reject("mask_format_error", {"phase": "mask_fmt", "err": exc})

    out = FENCE_LANG_RE.sub("```txt", out)
    out = JSON_KEY_RE.sub(f'"{token_field}":', out)
    out = YAML_FIELD_RE.sub(lambda m: f"{m.group(1)}{token_field}:", out)
    out = DATA_ATTR_RE.sub(token_data, out)
    out = MD_TITLE_T_RE.sub(r'\1"meta:[MASK]"', out)
    out = FIELD_NAME_RE.sub(token_field, out)
    return out


def _has_disc(s: str) -> bool:
    pats = [
        r"请勿执行",
        r"不具约束",
        r"仅用于.*展示",
        r"合规说明",
        r"安全提示",
        r"示例",
        r"仅为样式",
        r"仅作展示",
        r"不必执行",
    ]
    return any(re.search(p, s) for p in pats)


def _has_art(s: str) -> bool:
    return any(re.search(re.escape(w), s, flags=re.I) for w in ARTIFACT_WORDS)


def _rate(rows, predicate):
    total = len(rows) or 1
    return sum(1 for row in rows if predicate(row)) / total


def sample_disclaimer(rng: random.Random, enable: bool = True) -> Optional[str]:
    return rng.choice(DISCLAIMER_BANK) if enable and rng.random() < 0.35 else None


__all__ = [
    "DISCLAIMER_BANK",
    "MASKED_FIELD_FAMILY_COUNTER",
    "apply_symmetric_accessories",
    "mask_format_features_sym",
    "sample_disclaimer",
    "_has_disc",
    "_has_art",
    "_rate",
]
