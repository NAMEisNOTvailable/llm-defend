"""
Lightweight text utilities used across the compose pipeline.

These helpers were extracted from the legacy monolith so that the refactored
modules can share a consistent implementation for length metrics and surface
feature probes.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, Optional

ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]
ZERO_WIDTH_RE = re.compile("[" + "".join(re.escape(ch) for ch in ZERO_WIDTH_CHARS) + "]")
BIDI_CONTROL_RE = re.compile(r"[\u202a-\u202e\u2066-\u2069]")
CODE_FENCE = re.compile(r"```")
XML_TAG = re.compile(r"<\s*[/]?\s*([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*[^>]*>")
BRACES = re.compile(r"[{}\[\]()]")
URL_RE = re.compile(r"https?://|www\.")
AT_HASH_RE = re.compile(r"[@#]\w+")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
CN_RE = re.compile(r"[\u4e00-\u9fff]")

CN_SHARE_AUTO_CHOICES = (0.60, 0.70, 0.85, 0.90)
CN_SHARE_TIER_FLOOR = 0.35
CN_SHARE_STRUCT_MIN = 0.50
CN_SHARE_DEFAULT_BUCKET_MIN = 0.60
CN_SHARE_MULTI_TURN_MIN = 0.70
CN_SHARE_SURFACE_FLOOR = 0.60

_PAYLOAD_KIND_TO_CODE: Dict[str, int] = {
    "": 0,
    "none": 0,
    "base64": 1,
    "url": 2,
    "html": 3,
    "hex": 4,
    "rot13": 5,
    "fullwidth": 6,
    "zwsp": 7,
}


def byte_len(s: str) -> int:
    """Return UTF-8 byte length."""
    if s is None:
        return 0
    if not isinstance(s, str):
        s = str(s)
    return len(s.encode("utf-8"))


def length_bucket(n: int) -> str:
    """Bucket length into short/medium/long categories."""
    num = int(max(0, n))
    if num < 60:
        return "short"
    if num < 240:
        return "medium"
    return "long"


def CN_latin_ratio(s: str) -> float:
    """Compute latin-to-CN character ratio."""
    if not s:
        return 0.0
    cn = len(CN_RE.findall(s))
    latin = len(LATIN_RE.findall(s))
    return (latin + 1e-6) / (cn + 1e-6)


def feature_probe_clean(
    s: Any,
    *,
    carrier: Optional[str] = None,
    payload_encoding: Optional[str] = None,
) -> Dict[str, int]:
    """
    Surface feature probe used for dataset analytics and leakage checks.
    Returns a dict keyed by feature name.
    """
    text = s if isinstance(s, str) else ("" if s is None else str(s))
    codef = int(bool(CODE_FENCE.search(text)))
    xml = int(bool(XML_TAG.search(text)))
    braces = int(bool(BRACES.search(text)))
    url = int(bool(URL_RE.search(text)))
    athash = int(bool(AT_HASH_RE.search(text)))
    latin = int(bool(LATIN_RE.search(text)))
    digit = int(bool(DIGIT_RE.search(text)))
    cn_hit = int(bool(CN_RE.search(text)))
    zw_flag = int(bool(ZERO_WIDTH_RE.search(text)))
    bidi_flag = int(bool(BIDI_CONTROL_RE.search(text)))
    homoglyph_flag = 0  # legacy placeholder; populated by compiled extensions when available

    carrier_key = (carrier or "").strip()
    double_wrap_flag = int(bool(carrier_key and ("+" in carrier_key)))
    payload_key = (payload_encoding or "").strip().lower()
    payload_kind = int(_PAYLOAD_KIND_TO_CODE.get(payload_key, 0))

    return {
        "codef": codef,
        "xml": xml,
        "braces": braces,
        "url": url,
        "athash": athash,
        "latin": latin,
        "digit": digit,
        "CN": cn_hit,
        "zw": zw_flag,
        "bidi": bidi_flag,
        "homoglyph": homoglyph_flag,
        "double_wrap": double_wrap_flag,
        "payload_enc_kind": payload_kind,
    }


def resolve_cn_share_targets(
    *,
    rng: Optional[random.Random] = None,
    min_cn_share: float = 0.60,
    min_cn_share_auto: bool = False,
    mode: str = "single_turn",
    has_struct: bool = False,
    bucket_min_override: Optional[float] = None,
) -> Dict[str, float]:
    """
    Resolve CN-share thresholds so positives/negatives share the same policy.

    Returns a dict containing ``base`` (the sampled baseline target),
    ``bucket_min`` (structural floor), ``hard_target`` (used by the hard policy),
    ``surface_target`` (used for surface augments), and ``tier_floor`` guards.
    """
    picker = rng or random
    if min_cn_share_auto:
        base = float(picker.choice(CN_SHARE_AUTO_CHOICES))
    else:
        base = float(min_cn_share)

    if bucket_min_override is not None:
        bucket_min = float(bucket_min_override)
    else:
        if mode == "multi_turn":
            bucket_min = CN_SHARE_MULTI_TURN_MIN
        elif has_struct:
            bucket_min = CN_SHARE_STRUCT_MIN
        else:
            bucket_min = CN_SHARE_DEFAULT_BUCKET_MIN

    hard_target = max(bucket_min, base)
    surface_target = max(CN_SHARE_SURFACE_FLOOR, base)
    return {
        "base": base,
        "bucket_min": bucket_min,
        "hard_target": hard_target,
        "surface_target": surface_target,
        "tier_floor": CN_SHARE_TIER_FLOOR,
    }


__all__ = [
    "ZERO_WIDTH_RE",
    "BIDI_CONTROL_RE",
    "byte_len",
    "length_bucket",
    "CN_latin_ratio",
    "feature_probe_clean",
    "resolve_cn_share_targets",
]
