"""Aggregated text/surface helpers used by the DSL renderers."""

from __future__ import annotations

import random
import re
from typing import Optional

from .utils import (
    enc_base64,
    enc_url,
    enc_html_entity,
    enc_hex,
    enc_rot13,
    enc_fullwidth,
    insert_zwsp,
    payload_variants,
)

try:  # pragma: no cover - optional dependency
    from opencc import OpenCC  # type: ignore
except Exception:  # pragma: no cover - fallback when opencc is absent
    OpenCC = None  # type: ignore

__all__ = [
    # Encoders (re-exported from utils)
    "enc_base64",
    "enc_url",
    "enc_html_entity",
    "enc_hex",
    "enc_rot13",
    "enc_fullwidth",
    "insert_zwsp",
    "payload_variants",
    # Noise helpers
    "CODE_BLOCK_RE",
    "apply_cn_eco_noise",
    # Surface/region helpers
    "apply_cn_region",
    "sinicize_surface",
    # Alias helpers
    "ALIAS_MAP",
    "CN_ALIAS_MAP",
    "randomize_field_aliases",
]

# ---------------------------------------------------------------------------
# CN surface noise
# ---------------------------------------------------------------------------

_EMOS = ["😊", "😂", "😅", "😉", "🤔", "😐", "😑", "🙂", "🙃", "🤫", "😴", "😮"]
_FILLERS = ["呀", "欸", "啊", "哟", "咦", "哎", "诶呀", "哎呀", "呐", "呃", "对吧", "呗", "emmm", "emm", "233", "哈哈哈"]
_PUNCT_SWAP = {"，": "。", "。": ",", "：": ":", "；": ";", "！": "!", "？": "?", "—": "-", "～": "~"}
CODE_BLOCK_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]+)?\s*\n([\s\S]*?)```", re.M | re.S)


def _apply_cn_eco_free(seg: str, rng: random.Random, rate: float) -> str:
    s = seg
    if not s.strip():
        return s
    if rng.random() < rate:
        s = "".join(_PUNCT_SWAP.get(ch, ch) if rng.random() < 0.15 else ch for ch in s)
    if rng.random() < rate * 0.6:
        if rng.random() < 0.5:
            s = rng.choice(_FILLERS) + "，" + s
        else:
            s = s + rng.choice(["呀", "啦"]) + rng.choice(_EMOS)
    if rng.random() < rate * 0.4:
        if "结果" in s and rng.random() < 0.5:
            s = s.replace("结果", "result")
        if "工具" in s and rng.random() < 0.3:
            s = s.replace("工具", "tool")
    if rng.random() < rate * 0.3:
        if "注意" in s:
            s = s.replace("注意", "注意噢")
        if "路径" in s and rng.random() < 0.5:
            s = s.replace("路径", "途经")
    return s


def apply_cn_eco_noise(text: str, rng: random.Random, rate: float = 0.18) -> str:
    """Inject light code-switch and colloquial noise into natural-language spans."""
    out: list[str] = []
    last = 0
    t = text or ""
    for match in CODE_BLOCK_RE.finditer(t):
        if match.start() > last:
            out.append(_apply_cn_eco_free(t[last:match.start()], rng, rate))
        out.append(match.group(0))
        last = match.end()
    if last < len(t):
        out.append(_apply_cn_eco_free(t[last:], rng, rate))
    return "".join(out)


# ---------------------------------------------------------------------------
# Regional normalisation & directive sinicization
# ---------------------------------------------------------------------------

_cn2tw = _tw2cn = None
if OpenCC is not None:  # pragma: no cover - exercised when opencc is installed
    try:
        _cn2tw = OpenCC("s2twp")
        _tw2cn = OpenCC("tw2s")
    except Exception:  # pragma: no cover - guard errant configs
        _cn2tw = _tw2cn = None


def apply_cn_region(text: str, region: str) -> str:
    """Convert Simplified/Traditional surface forms depending on region."""
    result = text
    try:
        if region == "tw" and _cn2tw:
            result = _cn2tw.convert(text)
        if region == "cn_mainland" and _tw2cn:
            result = _tw2cn.convert(text)
    except Exception:
        result = text
    return result


def sinicize_surface(text: str) -> str:
    """Replace key English directives with CN aliases, skipping fenced code."""

    replacements = {
        "only output": "仅输出",
        "final answer": "最终答案",
        "result slot": "结果槽位",
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


# ---------------------------------------------------------------------------
# Field alias randomisation
# ---------------------------------------------------------------------------

ALIAS_MAP = {
    r"\b(?:note|remark|comment)\b": ["备注", "说明", "注记"],
    r"\b(?:result|answer|verdict|final)\b": ["结果", "答案", "结论"],
    r"\brole\b": ["角色", "通道角色"],
    r"\b(?:path|file|location)\b": ["路径", "文件路径"],
    r"\b(?:upstream|source)\b": ["上游", "来源"],
    r"\b(?:append|suffix|footer)\b": ["追加", "尾注", "附注"],
    r"\b(?:tool|function_call|args|arguments)\b": ["工具", "函数调用", "参数"],
    r"\bdirectives?\b": ["指令", "提示"],
    r"\btitle\b": ["标题", "题名"],
    r"\b(?:summary|abstract)\b": ["摘要", "概览"],
    r"\btags?\b": ["标签", "关键词"],
    r"\bstatus\b": ["状态", "进度"],
    r"\b(?:category|type)\b": ["类别", "类型"],
    r"\b(?:author|owner)\b": ["作者", "负责人"],
    r"\b(?:date|time|datetime|timestamp)\b": ["日期", "时间"],
}

CN_ALIAS_MAP = {
    "评测通道": ["评测通道", "评分通道", "评审通道", "打分通道", "审查通道"],
    "得分": ["得分", "评分", "分值", "评估值", "评分值"],
    "记忆写入": ["记忆写入", "写入记忆", "存储记忆", "持久化记忆", "存档"],
    "回合状态": ["回合状态", "轮次状态", "会话轮次", "阶段状态", "状态迁移"],
    "结果槽": ["结果槽", "结果字段", "输出槽", "目标槽", "结果位"],
}


def randomize_field_aliases(text: str, p_cn: float = 0.7, rng: Optional[random.Random] = None) -> str:
    """
    Probabilistically replace common field names with Chinese aliases.
    Intended for JSON/YAML/front-matter style blobs; Markdown tables are
    not modified explicitly.
    """

    _rng = rng or random
    t = text

    for pattern, cn_list in ALIAS_MAP.items():
        replacement = _rng.choice(cn_list)
        t = re.sub(
            rf'(")({pattern})(")\s*:',
            lambda m: m.group(1) + (replacement if _rng.random() < p_cn else m.group(2)) + m.group(3) + ":",
            t,
            flags=re.I,
        )
        t = re.sub(
            rf"(?m)^({pattern})\s*:",
            lambda m: (replacement if _rng.random() < p_cn else m.group(1)) + ":",
            t,
            flags=re.I,
        )

    def _rand_cn(word: str) -> str:
        alts = CN_ALIAS_MAP.get(word)
        if not alts:
            return word
        return _rng.choice(alts) if _rng.random() < float(p_cn) else word

    t = re.sub(
        r"(?m)^(\s*)(评测通道|得分|记忆写入|回合状态|结果槽)(\s*:\s*)",
        lambda m: f"{m.group(1)}{_rand_cn(m.group(2))}{m.group(3)}",
        t,
    )
    return t
