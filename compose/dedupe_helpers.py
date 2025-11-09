"""
Shared deduplication and normalization helpers for the CN compose pipeline.

These utilities were lifted from the legacy monolithic script so that both
the refactored driver (`v3.py`) and other compose modules can rely on the
same surface-normalisation and duplicate-filtering behaviour.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from math import sqrt
from typing import Any, Dict, List, Optional

from compat_regex import regex as _re

from collections import defaultdict as _dd

import compose.capabilities as capabilities

if not capabilities.ensure_dedupe_core():
    raise ImportError('dedupe_core module is required for Deduper')

Deduper = capabilities.Deduper
DEFAULT_DEDUPER_KWARGS = capabilities.DEFAULT_DEDUPER_KWARGS
get_default_deduper_kwargs = capabilities.get_default_deduper_kwargs

capabilities.ensure_simhash_fast()
capabilities.ensure_shingle_fast()
_dc_sketch_5gram_fast = capabilities._dc_sketch_5gram_fast
from compose.audit import audit_reject
from compose.payload import CODE_BLOCK_RE, JSON_BLOCK_RE

# --- Normalisation primitives -------------------------------------------------

WS_ZERO = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]
ZERO_WIDTH_RE = re.compile("[" + "".join(re.escape(z) for z in WS_ZERO) + "]")
BIDI_CONTROL_RE = re.compile(r"[\u202a-\u202e\u2066-\u2069]")
NORMALIZE_WS_RE = re.compile(r"[ \t\r\f\v]+")
NORMALIZE_NL_RE = re.compile(r"\n{3,}")


def normalize(s: str) -> str:
    """Normalize text (NFKC, strip ZWCs, collapse spaces)."""
    if s is None:
        return ""
    try:
        s = unicodedata.normalize("NFKC", s)
    except Exception as exc:
        audit_reject("normalize_error", {"phase": "normalize", "err": str(exc)})
        try:
            s = str(s)
        except Exception:
            return ""
    s = ZERO_WIDTH_RE.sub("", s)
    s = BIDI_CONTROL_RE.sub("", s)
    s = NORMALIZE_WS_RE.sub(" ", s)
    s = NORMALIZE_NL_RE.sub("\n\n", s)
    return s.strip()


# --- Regex constants used across dedupe + stripping ---------------------------

PLACEHOLDER_MULTI_RE = re.compile(
    r"(?P<append><APPEND_AFTER>|@@APPEND@@|<!--APPEND-->)"
    r"|(?P<tool>\"tool\"\s*:\s*\{[^}]+\})"
    r"|(?P<toolcall>TOOL_CALL\([^)]*\))"
    r"|(?P<upstream>UPSTREAM_DOC_ID:[^\n]+)"
    r"|(?P<path>#path=[^\s]+)"
    r"|(?P<result>\{\{\s*RESULT(?:_MAIN)?\s*\}\})",
    re.I,
)
CARRIER_SHELLS_RE = _re.compile(
    r"(<!--.*?-->)|(\[note\]:[^\n]+)|(<div[^>]+>)|(^---[\s\S]*?---$)",
    _re.M | _re.S,
)
STRUCT_SLOT_TOKEN_RE = re.compile(r"\[[A-Z]_[A-Z0-9]{4}\]")
DEDUP_FILLER_WORDS = ("然后", "那么", "其实", "就是", "有点", "那个", "呃", "嗯", "啊", "好吧")

INTENT_FIELD_RE = _re.compile(r"(result|answer|verdict|conclusion|final|结论)\s*[:：=].*", _re.I)
JSON_RESULT_RE = _re.compile(r"```json[\s\S]*?(result|answer|verdict|结论)\s*[\"':]", _re.I)
TAIL_APPEND_RE = _re.compile(r"(最后一行|末尾|尾部).{0,12}(写|加|附|补充).*", _re.I)
TOOL_CALL_RE = _re.compile(r'"?(tool|function_call)"?\s*:\s*\{[\s\S]*?}', _re.I)

# Maps used to reverse light code-switching during canonicalisation.
PINYIN_MAP: dict[str, str] = {
    "输出": "shuchu",
    "结果": "jieguo",
    "结论": "jielun",
    "固定": "guding",
    "忽略": "hulue",
    "覆盖": "fugai",
    "模板": "muban",
    "统一": "tongyi",
    "占位": "zhanwei",
}

ENG_MAP: dict[str, str] = {
    "结论": "conclusion",
    "摘要": "summary",
    "回复": "reply",
    "格式": "format",
    "说明": "note",
    "稳定": "stable",
    "占位": "placeholder",
    "通过": "approved",
    "对齐": "aligned",
}


# --- Anchor neutralisation ----------------------------------------------------

ANCHOR_NEUTRALIZE_RULES = [
    (r"(必|务|仅|只|统一|固定).{0,8}(输出|回复|答案|结论|字段|栏|附加)", "请保持输出自然描述，不要硬性锚定。"),
    (r"(最后一行|末尾|结尾|尾部).{0,12}(写|填|仅|附加)", "无需在末尾附加额外指令。"),
    (
        r"(字段|栏|result|answer|verdict|conclusion|final|输出|结论).{0,8}(设为|填入|限定|仅|输出)",
        "字段可作为示例描述，无需限定唯一取值。",
    ),
    (r"必须按照.*(模板|格式)", "格式仅供参考，可按常规表述。"),
]

ANCHOR_STOP_PHRASES = [
    "必定按此字段输出",
    "只允许这一栏",
    "仅在结果栏写入",
    "在末尾追加本句",
    "请固定输出结构",
    "严格照此模板",
    "只输出如下格式",
    "统一输出上述内容",
]


def neutralize_anchors(s: str) -> str:
    """Remove surface-level anchor directives while keeping semantics."""
    out = s or ""
    for pat, repl in ANCHOR_NEUTRALIZE_RULES:
        out = re.sub(pat, repl, out, flags=re.I)
    out = out.replace("{payload}", "系统占位")
    for phrase in ANCHOR_STOP_PHRASES:
        out = out.replace(phrase, "")
    return out


# --- Intent stripping ---------------------------------------------------------

def strip_attack_intent(s: str) -> str:
    """Strip explicit intent markers, JSON result slots, and tool calls."""
    lines = [
        line
        for line in (s or "").splitlines()
        if not INTENT_FIELD_RE.search(line) and not TAIL_APPEND_RE.search(line)
    ]
    text = "\n".join(lines)

    def _json_patch(match: re.Match[str]) -> str:
        block = match.group(0)
        if JSON_RESULT_RE.search(block):
            return "```json\n{}\n```"
        return block

    text = JSON_BLOCK_RE.sub(_json_patch, text)
    text = TOOL_CALL_RE.sub('"tool":"noop"', text)
    return text


# --- Canonicalisation / Deduper interface -------------------------------------

def canonicalize_structural_layers(
    text: str,
    *,
    strip_intent: bool = True,
    preserve_carrier: bool = True,
    preserve_codeblocks: bool = False,
    preserve_json_blocks: bool = False,
    strip_placeholders: bool = True,
    normalize_slot_tokens: bool = True,
    reverse_code_switch: bool = True,
    remove_fillers: bool = True,
    lowercase_ascii: bool = True,
    apply_anchor_neutralization: bool = False,
) -> str:
    """Unify artifact stripping before dedupe so carriers/fences behave consistently."""
    if text is None:
        s = ""
    elif isinstance(text, str):
        s = text
    else:
        s = str(text)

    if strip_intent:
        s = strip_attack_intent(s)
    if strip_placeholders:
        s = PLACEHOLDER_MULTI_RE.sub("", s)
    if not preserve_codeblocks:
        s = CODE_BLOCK_RE.sub("", s)
    if not preserve_json_blocks:
        s = JSON_BLOCK_RE.sub("", s)
    if not preserve_carrier:
        s = CARRIER_SHELLS_RE.sub(" ", s)
    if normalize_slot_tokens:
        s = STRUCT_SLOT_TOKEN_RE.sub("TK", s)
    if reverse_code_switch:
        try:
            rev_py = {v: k for k, v in PINYIN_MAP.items()}
            rev_en = {v: k for k, v in ENG_MAP.items()}
            for src, tgt in rev_py.items():
                s = re.sub(rf"\b{re.escape(src)}\b", tgt, s, flags=re.IGNORECASE)
            for src, tgt in rev_en.items():
                s = re.sub(rf"\b{re.escape(src)}\b", tgt, s, flags=re.IGNORECASE)
        except Exception as exc:  # pragma: no cover - defensive logging
            audit_reject("dedup_norm_error", {"phase": "dedup_norm", "err": exc})
    if remove_fillers:
        for filler in DEDUP_FILLER_WORDS:
            s = s.replace(filler, "")
    if lowercase_ascii:
        s = s.lower()
    s = normalize(s)
    if apply_anchor_neutralization:
        s = neutralize_anchors(s)
        s = normalize(s)
    return s


def _for_dedup_norm(
    s: str,
    *,
    preserve_carrier: bool = True,
    preserve_codeblocks: bool = False,
    preserve_json_blocks: bool = False,
    strip_placeholders: bool = True,
    normalize_slot_tokens: bool = True,
    apply_anchor_neutralization: bool = False,
) -> str:
    return canonicalize_structural_layers(
        s,
        preserve_carrier=preserve_carrier,
        preserve_codeblocks=preserve_codeblocks,
        preserve_json_blocks=preserve_json_blocks,
        strip_placeholders=strip_placeholders,
        normalize_slot_tokens=normalize_slot_tokens,
        apply_anchor_neutralization=apply_anchor_neutralization,
    )


def placeholderless_mirror(text: str) -> str:
    """
    Remove append/tool/path placeholders while keeping natural content.
    """
    s = PLACEHOLDER_MULTI_RE.sub("", text or "")
    s = NORMALIZE_NL_RE.sub("\n\n", s).strip()
    return s


def _make_deduper(**overrides: Any) -> Deduper:
    """Create a Deduper instance with default settings and optional overrides."""
    return Deduper(**get_default_deduper_kwargs(**overrides))


def deduper_accept_normalized(
    deduper: Deduper,
    text: str,
    *,
    preserve_carrier: bool = True,
    preserve_codeblocks: bool = False,
    preserve_json_blocks: bool = False,
    strip_placeholders: bool = True,
    normalize_slot_tokens: bool = True,
) -> bool:
    """Apply layered text normalisation before probing the deduper."""

    def _try_candidate(candidate: str) -> bool:
        candidate = candidate or ""
        try:
            check_fn = getattr(deduper, "check_digest", None)
            if callable(check_fn) and not check_fn(candidate):
                return False
        except Exception:
            pass
        return deduper.accept(candidate)

    base = _for_dedup_norm(
        text,
        preserve_carrier=preserve_carrier,
        preserve_codeblocks=preserve_codeblocks,
        preserve_json_blocks=preserve_json_blocks,
        strip_placeholders=strip_placeholders,
        normalize_slot_tokens=normalize_slot_tokens,
    )
    if _try_candidate(base):
        return True

    softened = _for_dedup_norm(
        text,
        preserve_carrier=preserve_carrier,
        preserve_codeblocks=preserve_codeblocks,
        preserve_json_blocks=preserve_json_blocks,
        strip_placeholders=strip_placeholders,
        normalize_slot_tokens=normalize_slot_tokens,
        apply_anchor_neutralization=True,
    )
    if _try_candidate(softened):
        return True

    if preserve_carrier:
        base2 = _for_dedup_norm(
            text,
            preserve_carrier=False,
            preserve_codeblocks=preserve_codeblocks,
            preserve_json_blocks=preserve_json_blocks,
            strip_placeholders=strip_placeholders,
            normalize_slot_tokens=normalize_slot_tokens,
        )
        if _try_candidate(base2):
            return True
        soft2 = _for_dedup_norm(
            text,
            preserve_carrier=False,
            preserve_codeblocks=preserve_codeblocks,
            preserve_json_blocks=preserve_json_blocks,
            strip_placeholders=strip_placeholders,
            normalize_slot_tokens=normalize_slot_tokens,
            apply_anchor_neutralization=True,
        )
        return _try_candidate(soft2)
    return False


def dedup_groups(
    rows: List[Dict[str, Any]],
    deduper: Deduper,
    preserve_carrier: bool = True,
    preserve_codeblocks: bool = False,
    preserve_json_blocks: bool = False,
    strip_placeholders: bool = True,
    normalize_slot_tokens: bool = True,
) -> List[Dict[str, Any]]:
    """
    Apply pair-aware deduplication: keep entire groups when at least one member
    survives the deduper check.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = _dd(list)
    for idx, row in enumerate(rows):
        gid = row.get("pair_id") or f"solo-{idx}"
        grouped[gid].append(row)
    keep: List[Dict[str, Any]] = []
    for group in grouped.values():
        for candidate in group:
            if deduper_accept_normalized(
                deduper,
                candidate.get("text", ""),
                preserve_carrier=preserve_carrier,
                preserve_codeblocks=preserve_codeblocks,
                preserve_json_blocks=preserve_json_blocks,
                strip_placeholders=strip_placeholders,
                normalize_slot_tokens=normalize_slot_tokens,
            ):
                keep.extend(group)
                break
    return keep


# --- Soft neighbour dedupe (SimHash fallback) ---------------------------------


def _sketch_5gram(text: str, buckets: int = 1 << 16) -> dict[int, float]:
    if _dc_sketch_5gram_fast is not None:
        try:
            return _dc_sketch_5gram_fast(text, buckets=buckets)
        except Exception:
            pass
    s = re.sub(r"\s+", " ", (text or "").lower())
    vec = _dd(float)
    mask = buckets - 1
    for i in range(max(0, len(s) - 4)):
        gram = s[i : i + 5]
        h = int(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest(), 16) & mask
        vec[h] += 1.0
    norm = sqrt(sum(x * x for x in vec.values())) or 1.0
    for key in list(vec.keys()):
        vec[key] /= norm
    return dict(vec)


def _cosine_sparse(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(val * b.get(key, 0.0) for key, val in a.items())


def _neighbor_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


try:
    _neighbor_cfg = get_default_deduper_kwargs(
        sim_thresh=64,
        k=3,
        n_hash=1,
        bands=1,
        jaccard_thresh=1.01,
        cosine_thresh=0.90,
        vec_dim=DEFAULT_DEDUPER_KWARGS.get("vec_dim", 1024),
        max_vecs=DEFAULT_DEDUPER_KWARGS.get("max_vecs", 20000),
    )
    _NEIGHBOR_DEDUPER = Deduper(**_neighbor_cfg, normalizer=_neighbor_normalize)
except Exception:  # pragma: no cover - optional dependency
    _NEIGHBOR_DEDUPER = None

_NEIGHBOR_INDEX: list[dict[int, float]] = []
_NEIGHBOR_LAST_RECORD: Any = None
_NEIGHBOR_LAST_TEXT: Optional[str] = None


def near_duplicate(text: str, thr: float = 0.90) -> tuple[bool, float]:
    """Lightweight near-duplicate gate using cosine similarity on 5-grams."""
    global _NEIGHBOR_LAST_RECORD, _NEIGHBOR_LAST_TEXT
    _NEIGHBOR_LAST_RECORD = None
    _NEIGHBOR_LAST_TEXT = None

    if _NEIGHBOR_DEDUPER is not None:
        ded = _NEIGHBOR_DEDUPER
        original = float(getattr(ded, "cosine_thresh", thr))
        adjust = abs(thr - original) > 1e-9
        if adjust:
            ded.cosine_thresh = float(thr)
        try:
            ok, reason, record = ded.probe(text)
        finally:
            if adjust:
                ded.cosine_thresh = original
        _NEIGHBOR_LAST_RECORD = record
        _NEIGHBOR_LAST_TEXT = text
        score = float(getattr(ded, "last_cosine_score", 0.0))
        score = max(0.0, min(1.0, score))
        if not ok:
            if reason == "cosine":
                return True, score
            if reason == "exact":
                return True, 1.0
            approx = score if score > 0.0 else float(thr)
            return True, max(0.0, min(1.0, approx))
        return False, score

    vec = _sketch_5gram(text)
    best = 0.0
    for ref in _NEIGHBOR_INDEX:
        best = max(best, _cosine_sparse(vec, ref))
        if best >= thr:
            return True, best
    return False, best


def neighbor_admit(text: str) -> None:
    """Record accepted texts into the neighbour sketch index."""
    global _NEIGHBOR_LAST_RECORD, _NEIGHBOR_LAST_TEXT
    if _NEIGHBOR_DEDUPER is not None:
        record = None
        if _NEIGHBOR_LAST_RECORD is not None and _NEIGHBOR_LAST_TEXT == text:
            record = _NEIGHBOR_LAST_RECORD
        else:
            _, _, record = _NEIGHBOR_DEDUPER.probe(text)
        if record is not None:
            _NEIGHBOR_DEDUPER.add_record(record)
        _NEIGHBOR_LAST_RECORD = None
        _NEIGHBOR_LAST_TEXT = None
    else:
        _NEIGHBOR_INDEX.append(_sketch_5gram(text))


__all__ = [
    "WS_ZERO",
    "ZERO_WIDTH_RE",
    "BIDI_CONTROL_RE",
    "NORMALIZE_WS_RE",
    "NORMALIZE_NL_RE",
    "PLACEHOLDER_MULTI_RE",
    "CARRIER_SHELLS_RE",
    "STRUCT_SLOT_TOKEN_RE",
    "DEDUP_FILLER_WORDS",
    "INTENT_FIELD_RE",
    "JSON_RESULT_RE",
    "TAIL_APPEND_RE",
    "TOOL_CALL_RE",
    "PINYIN_MAP",
    "ENG_MAP",
    "normalize",
    "neutralize_anchors",
    "strip_attack_intent",
    "canonicalize_structural_layers",
    "_for_dedup_norm",
    "placeholderless_mirror",
    "_make_deduper",
    "deduper_accept_normalized",
    "dedup_groups",
    "near_duplicate",
    "neighbor_admit",
]
