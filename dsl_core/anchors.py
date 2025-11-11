from __future__ import annotations

import json
import random
import re
import string
from typing import Any, Iterable, Optional

__all__ = [
    "Anchors",
    "strip_anchors",
    "APPEND_RE",
    "PATH_RE",
    "UP_RE",
    "ROLE_RE",
    "TOOLCALL_RE",
]


def _rand_id(rnd: random.Random, k: int = 8, alphabet: str = string.ascii_uppercase + string.digits) -> str:
    return "".join(rnd.choice(alphabet) for _ in range(k))


class Anchors:
    """
    Anchor factory for randomized structural tokens.
    Produces ephemeral APPEND tags, path hashes, upstream IDs, role hints,
    and tool call names to reduce lexical artifacts.
    """

    TOOLNAMES = ("write_sheet", "query_db", "send_email", "fetch_url", "kv_put", "kv_get")

    def __init__(self, rnd: random.Random):
        self.rnd = rnd
        self.APPEND = f"<APPEND_{_rand_id(rnd)}>"
        self.PATH = f"#p={_rand_id(rnd, k=10)}"
        self.UP = f"UPSTREAM_DOC_ID:{_rand_id(rnd, k=4)}-{_rand_id(rnd, k=6)}"
        self.ROLE = f"role:{rnd.choice(['system', 'user'])}"

    def TOOL_CALL(self, name: Optional[str] = None) -> str:
        name = name or self.rnd.choice(self.TOOLNAMES)
        return f"TOOL_CALL({name})"

    @property
    def ROLE_VAL(self) -> str:
        return self.ROLE.split(":", 1)[-1]


APPEND_RE = re.compile(r"<APPEND_[A-Z0-9]{4,16}>")
PATH_RE = re.compile(r"#p=[A-Z0-9]{6,18}")
HTML_PATH_ATTR_RE = re.compile(
    r"\s*data-(?:path|role|source|target|origin)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)",
    re.I,
)
META_ANCHOR_TAG_RE = re.compile(
    r"<meta\b[^>]*?(?:name|property)\s*=\s*(?:\"[^\"]*(?:path|role|source)[^\"]*\"|'[^']*(?:path|role|source)[^']*'|[^\s>]+)[^>]*?>",
    re.I,
)
URL_PATH_PARAM_RE = re.compile(r"(?i)(?P<lead>[?&])path=[^&\s>\"']+(?P<trail>&?)")
UP_RE = re.compile(r"UPSTREAM(?:_DOC_ID)?[:=][A-Z0-9\-#]{6,40}")
ROLE_RE = re.compile(r"\brole\s*[:：]\s*(system|user)\b", re.I)
TOOLCALL_RE = re.compile(r"TOOL_CALL\([A-Za-z_][\w]*\)")
SENSITIVE_HEADER_KEYS = ("X-Role-Anchor", "X-Source-Path", "X-Target-Path", "X-Role")
HEADER_LINE_RE = re.compile(
    r"(?mi)^\s*(?:X-Role-Anchor|X-Source-Path|X-Target-Path|X-Role)\s*:\s*.*$"
)
JSON_FENCE_RE = re.compile(
    r"(?P<open>```json[^\S\r\n]*\r?\n)(?P<body>.*?)(?P<close>\r?\n```)",
    re.I | re.S,
)


def _strip_query_segment(match: re.Match) -> str:
    lead = match.group("lead")
    trail = match.group("trail") or ""
    if lead == "?":
        return "?" if trail == "&" else ""
    if lead == "&":
        return "&" if trail == "&" else ""
    return ""


def _strip_anchor_tokens(text: str) -> str:
    t = re.sub(r"\{\{\s*RESULT(?:_MAIN)?\s*\}\}", "", text or "")
    t = APPEND_RE.sub("", t)
    t = TOOLCALL_RE.sub("", t)
    t = PATH_RE.sub("", t)
    t = re.sub(r"#path=[^\s]+", "", t, flags=re.I)
    t = HTML_PATH_ATTR_RE.sub("", t)
    t = META_ANCHOR_TAG_RE.sub("", t)
    t = URL_PATH_PARAM_RE.sub(_strip_query_segment, t)
    t = UP_RE.sub("", t)
    t = ROLE_RE.sub("", t)
    return t


_JSON_REMOVED = object()


def _prune_json_structure(value: Any, *, allow_remove: bool) -> Any:
    if isinstance(value, str):
        stripped = _strip_anchor_tokens(value).strip()
        if not stripped and allow_remove:
            return _JSON_REMOVED
        return stripped
    if isinstance(value, list):
        items = []
        for item in value:
            pruned = _prune_json_structure(item, allow_remove=True)
            if pruned is _JSON_REMOVED:
                continue
            items.append(pruned)
        if not items and allow_remove:
            return _JSON_REMOVED
        return items
    if isinstance(value, dict):
        result = {}
        for key, val in value.items():
            if str(key) in SENSITIVE_HEADER_KEYS:
                continue
            pruned = _prune_json_structure(val, allow_remove=True)
            if pruned is _JSON_REMOVED:
                continue
            result[key] = pruned
        if not result and allow_remove:
            return _JSON_REMOVED
        return result
    return value


def _render_json_body(data: Any, original_body: str) -> str:
    multiline = "\n" in original_body.strip()
    if multiline and isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False, indent=2)
    return json.dumps(data, ensure_ascii=False, separators=(",", ": "))


def _clean_json_fences(text: str) -> str:
    if "```json" not in (text or ""):
        return text or ""

    def _replace(match: re.Match) -> str:
        open_part = match.group("open")
        close_part = match.group("close")
        body = match.group("body")
        raw_body = body.strip()
        if not raw_body:
            return match.group(0)
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            return match.group(0)
        pruned = _prune_json_structure(parsed, allow_remove=False)
        rendered = _render_json_body(pruned, body)
        return f"{open_part}{rendered}{close_part}"

    return JSON_FENCE_RE.sub(_replace, text or "")


def strip_anchors(text: str) -> str:
    """
    Remove structural anchors/placeholders to produce anchor-free variants.
    Strips placeholders like {{RESULT_MAIN}}, tool markers, append tags,
    path anchors, upstream IDs, and role hints; normalizes whitespace.
    """

    t = _clean_json_fences(text or "")
    t = _strip_anchor_tokens(t)
    # remove soft role/path headers and JSON fields like X-Source-Path
    t = HEADER_LINE_RE.sub("", t)
    for key in SENSITIVE_HEADER_KEYS:
        key_pattern = re.escape(key)
        t = re.sub(rf'\s*"{key_pattern}"\s*:\s*"[^"]*"\s*,', "", t)
        t = re.sub(rf',\s*"{key_pattern}"\s*:\s*"[^"]*"', "", t)
        t = re.sub(rf'"{key_pattern}"\s*:\s*"[^"]*"', "", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t
