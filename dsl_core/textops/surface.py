"""Surface normalization utilities for Chinese renderers."""

from __future__ import annotations

import re

from .noise_cn import CODE_BLOCK_RE  # reuse existing fenced-block detector

__all__ = ["apply_cn_region", "sinicize_surface"]

try:  # pragma: no cover - optional dependency
    from opencc import OpenCC  # type: ignore
except Exception:  # pragma: no cover - fallback when opencc is absent
    OpenCC = None  # type: ignore

_cn2tw = _cn2hk = _tw2cn = None
if OpenCC is not None:  # pragma: no cover - exercised when opencc is installed
    try:
        _cn2tw = OpenCC("s2twp")
        _cn2hk = OpenCC("s2hk")
        _tw2cn = OpenCC("tw2s")
    except Exception:
        _cn2tw = _cn2hk = _tw2cn = None


def apply_cn_region(text: str, region: str) -> str:
    """Convert Simplified/Traditional surface forms depending on region."""
    try:
        if region == "tw" and _cn2tw:
            return _cn2tw.convert(text)
        if region == "hk" and _cn2hk:
            return _cn2hk.convert(text)
        if region == "cn_mainland" and _tw2cn:
            return _tw2cn.convert(text)
    except Exception:
        return text
    return text


def sinicize_surface(text: str) -> str:
    """Replace key English directives with CN aliases, skipping fenced code."""

    replacements = {
        "only output": "仅输出",
        "final answer": "最终答案",
        "result slot": "结果槽",
        "ignore above": "忽略以上内容",
        "override": "以此段为准",
        "append": "追加",
        "tool": "工具",
        "role": "角色",
        "path": "路径",
        "upstream": "上游",
    }

    def _transform(segment: str) -> str:
        out = segment
        for key, value in replacements.items():
            out = re.sub(rf"\b{re.escape(key)}\b", value, out, flags=re.I)
        return out

    parts: list[str] = []
    last = 0
    src = text or ""
    for match in CODE_BLOCK_RE.finditer(src):
        if match.start() > last:
            parts.append(_transform(src[last:match.start()]))
        parts.append(match.group(0))
        last = match.end()
    if last < len(src):
        parts.append(_transform(src[last:]))
    return "".join(parts)
