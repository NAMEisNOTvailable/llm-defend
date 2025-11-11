from __future__ import annotations

from functools import lru_cache
import base64
import codecs
import hashlib as _hlib
import json
import math
import re
import urllib.parse
import warnings
from typing import Any, Callable, Iterable, Optional

from dedupe_core import _h64 as _dedupe_h64

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback env
    np = None  # type: ignore

try:
    import orjson as _json_fast  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _json_fast = None  # type: ignore

try:
    from compat_regex import regex as _re  # type: ignore
except Exception:  # pragma: no cover - fallback env
    _re = re  # type: ignore

_h64 = _dedupe_h64

__all__ = [
    "_json_loads",
    "_sketch_5gram_fast",
    "_h64",
    "_bucket_hash_cached",
    "bucket_hash",
    "_resolve_structural_ratio",
    "_parse_json_blocks",
    "set_probe_audit_hook",
    "_probe_audit",
    "enc_base64",
    "enc_url",
    "enc_html_entity",
    "enc_hex",
    "enc_rot13",
    "enc_fullwidth",
    "insert_zwsp",
    "payload_variants",
    "_PAT",
    "_rx_any",
    "JSON_BLOCK_RE",
    "EVD_HINT_FRONT",
    "EVD_HINT_HTTP",
    "HEADER_KV",
    "HTTP_METHOD_RE",
    "TOOL_KV",
    "YAML_KEY_LINE_RE",
    "CN_share",
    "_looks_like_email",
    "_looks_like_http_header",
    "_looks_like_yaml",
]


def _json_loads(data: str | bytes | bytearray) -> Any:
    """Fast-or-fallback JSON loader."""
    if _json_fast is not None:
        return _json_fast.loads(data)
    return json.loads(data)


def _sketch_5gram_fast(s: str, buckets: int = 1 << 16) -> dict[int, float]:
    """
    Compute normalized 5-gram sketch using vectorized numpy path when available.
    Used by semantic dedupe helpers.
    """
    txt = re.sub(r"\s+", " ", (s or "").lower())
    L = len(txt)
    if L < 5:
        return {}
    mask = buckets - 1
    grams = (txt[i:i + 5] for i in range(L - 4))
    if np is not None:
        hashes = np.fromiter((_h64(g) & mask for g in grams), dtype=np.uint32, count=L - 4)  # type: ignore[name-defined]
        if hashes.size == 0:
            return {}
        uniq, counts = np.unique(hashes, return_counts=True)
        vals = counts.astype(np.float64)
        norm = np.linalg.norm(vals)
        if norm == 0.0:
            return {}
        vals /= norm
        return {int(k): float(v) for k, v in zip(uniq.tolist(), vals.tolist())}
    counts = {}
    for g in grams:
        idx = int(_h64(g) & mask)  # type: ignore[name-defined]
        counts[idx] = counts.get(idx, 0) + 1
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm == 0.0:
        return {}
    return {k: v / norm for k, v in counts.items()}


@lru_cache(maxsize=4096)
def _bucket_hash_cached(key: tuple[str, ...]) -> str:
    s = "|".join(key)
    return _hlib.md5(s.encode("utf-8")).hexdigest()[:12]


def bucket_hash(items: Iterable[Any] | None) -> str:
    """Stable hash for coverage buckets."""
    try:
        key = tuple(sorted(str(x) for x in (items or []) if x))
    except Exception:
        key = tuple()
    return _bucket_hash_cached(key)


_STRUCT_RATIO_DEPRECATION_WARNED = False


def _resolve_structural_ratio(pin: Optional[dict[str, Any]] | None) -> float:
    """Resolve structural ratio from DSL pin, warning on deprecated keys."""
    global _STRUCT_RATIO_DEPRECATION_WARNED
    default = 0.60
    getter = getattr(pin, "get", None) if pin is not None else None
    if callable(getter):
        ratio = getter("structural_pos_ratio", None)
        if ratio is not None:
            try:
                value = float(ratio)
            except Exception:
                return default
            if isinstance(pin, dict):
                pin.setdefault("structural_pos_ratio", value)
            return value
        if getter("structural_p", None) is not None:
            ratio = getter("structural_p", None)
            if not _STRUCT_RATIO_DEPRECATION_WARNED:
                warnings.warn(
                    "DSL pin key 'structural_p' is deprecated; use 'structural_pos_ratio' instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _STRUCT_RATIO_DEPRECATION_WARNED = True
            try:
                value = float(ratio)
            except Exception:
                value = default
            if isinstance(pin, dict):
                pin.setdefault("structural_pos_ratio", value)
            return value
    return default


JSON_BLOCK_RE = re.compile(r"```(?:json|jsonc|json5|jsonl)\b\s*\n([\s\S]*?)```", re.I)


def _parse_json_blocks(txt: str, *, on_error: Optional[Callable[[Exception], None]] = None) -> Iterable[Any]:
    """Yield parsed JSON objects from fenced code blocks."""
    blocks = JSON_BLOCK_RE.findall(txt or "")
    for block in blocks:
        snippet = block.strip("` \n")
        if not snippet:
            continue
        try:
            yield _json_loads(snippet)
        except Exception as exc:
            if on_error:
                on_error(exc)


_PROBE_AUDIT_HOOK: Optional[Callable[[str, dict], None]] = None


def set_probe_audit_hook(hook: Optional[Callable[[str, dict], None]]) -> None:
    """Register a callback for internal audit events."""
    global _PROBE_AUDIT_HOOK
    _PROBE_AUDIT_HOOK = hook


def _probe_audit(reason: str, ctx: Optional[dict] = None) -> None:
    hook = _PROBE_AUDIT_HOOK
    if hook:
        try:
            hook(reason, ctx or {})
        except Exception:
            pass


def enc_base64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def enc_url(s: str) -> str:
    return urllib.parse.quote(s, safe="")


def enc_html_entity(s: str) -> str:
    return "".join(f"&#{ord(c)};" for c in s)


def enc_hex(s: str) -> str:
    return s.encode("utf-8").hex()


def enc_rot13(s: str) -> str:
    try:
        return codecs.encode(s, "rot_13")
    except Exception:
        return s


def enc_fullwidth(s: str) -> str:
    out: list[str] = []
    for ch in s or "":
        code = ord(ch)
        if 0x21 <= code <= 0x7E and ch != " ":
            out.append(chr(code + 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def insert_zwsp(s: str) -> str:
    if not s:
        return ""
    return "\u200b".join(list(s))


def payload_variants(payload: Optional[str]) -> set[str]:
    """Generate encoded payload variants for sandbox probing."""
    if not payload:
        return set()
    variants: set[str] = {payload}
    simple_encoders = (enc_base64, enc_url, enc_html_entity)
    depth_one: list[str] = []
    for fn in simple_encoders:
        try:
            res = fn(payload)
        except Exception:
            continue
        if isinstance(res, str) and res:
            variants.add(res)
            depth_one.append(res)
    for intermediate in depth_one:
        for fn in simple_encoders:
            try:
                chained = fn(intermediate)
            except Exception:
                continue
            if isinstance(chained, str) and chained:
                variants.add(chained)
    for fn in (enc_hex, enc_rot13, enc_fullwidth, insert_zwsp):
        try:
            res = fn(payload)
        except Exception:
            continue
        if isinstance(res, str) and res:
            variants.add(res)
    return variants


def _PAT(pattern: str):
    return _re.compile(pattern, _re.I | _re.S)


def _rx_any(words: Iterable[str]):
    return _PAT(r"|".join(map(_re.escape, words)))


_TOKEN_PATTERN = r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+|[^\s\w]"
try:
    from compat_regex import regex as _re2  # type: ignore
except Exception:
    _re2 = None

if _re2:
    _HAN_TOKEN_RE = _re2.compile(r"\p{Han}+|[A-Za-z]+|\d+|[^\s\w]", _re2.UNICODE)

    def _HAN_HAS(token: str) -> bool:
        return bool(_re2.search(r"\p{Han}", token))
else:
    _HAN_TOKEN_RE = re.compile(_TOKEN_PATTERN, re.UNICODE)

    def _HAN_HAS(token: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", token))

_TOKEN_PUNCT_RE = re.compile(r"^[^\w\u4e00-\u9fff]+$")


def CN_share(s: str) -> float:
    """Return the share of tokens that contain Chinese characters."""
    if not s:
        return 0.0
    toks = _HAN_TOKEN_RE.findall(s or "")
    if not toks:
        return 0.0
    total = 0
    cn = 0
    for tok in toks:
        if _TOKEN_PUNCT_RE.match(tok):
            continue
        total += 1
        if _HAN_HAS(tok):
            cn += 1
    if total == 0:
        return 0.0
    return cn / (total + 1e-6)


HEADER_KV = re.compile(r"^[A-Za-z\u4e00-\u9fff\-]{2,30}:\s?.{0,120}$", re.M)
HTTP_METHOD_RE = re.compile(r"(?m)^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+\S+")
YAML_KEY_LINE_RE = re.compile(r"(?m)^[\w\-]{1,32}:\s?.+$")
EVD_HINT_HTTP = re.compile(
    r"(?:\bRole\b|角色|\bPath\b|路径|Ref[- ]?(?:Chunk|Doc)|引用|来源|\bTool\b|工具|\bUpstream\b|上游|\bSession\b|会话)",
    re.I,
)
EVD_HINT_FRONT = re.compile(
    r"(?:角色|路径|来源|工具|会话|槽位|指令|ref[- ]?chunk|upstream|role|path|tool|session)",
    re.I,
)
TOOL_KV = re.compile(r'"?(name|tool)"?\s*:\s*"?([A-Za-z_]\w*)"?', re.I)


def _looks_like_email(text: str) -> bool:
    """Heuristic email detector: require a header block plus Subject."""
    sample = (text or "").strip()
    if not sample:
        return False
    lines = sample.splitlines()
    header_block = []
    for line in lines:
        if not line.strip():
            break
        if HEADER_KV.match(line):
            header_block.append(line)
        else:
            break
    subject_present = any(re.search(r"(Subject|主题)\s*:", line, re.I) for line in header_block)
    addressed = any(line.lower().startswith(("from:", "to:", "cc:", "bcc:")) for line in header_block)
    if subject_present and addressed and len(header_block) >= 2:
        return True
    return len(header_block) >= 3 and subject_present


def _looks_like_http_header(text: str) -> bool:
    """Basic HTTP-header detector with method+headers thresholds."""
    sample = text or ""
    lines = sample.splitlines()
    if not lines:
        return False
    first_line = lines[0].strip()
    has_method = bool(HTTP_METHOD_RE.match(first_line))
    kv_lines = [line for line in lines[1 if has_method else 0 :] if HEADER_KV.match(line)]
    if has_method and len(kv_lines) >= 2:
        return True
    return len(kv_lines) >= 3


def _looks_like_yaml(text: str) -> bool:
    sample = text or ""
    return ("```yaml" in sample) or bool(YAML_KEY_LINE_RE.search(sample))
