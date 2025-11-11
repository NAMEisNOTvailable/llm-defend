from __future__ import annotations

import random
import re
import string
from typing import Iterable, Optional

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
HTML_PATH_ATTR_RE = re.compile(r"\s*data-path\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.I)
URL_PATH_PARAM_RE = re.compile(r"(?i)(?P<lead>[?&])path=[^&\s>\"']+(?P<trail>&?)")
UP_RE = re.compile(r"UPSTREAM(?:_DOC_ID)?[:=][A-Z0-9\-#]{6,40}")
ROLE_RE = re.compile(r"\brole\s*[:：]\s*(system|user)\b", re.I)
TOOLCALL_RE = re.compile(r"TOOL_CALL\([A-Za-z_][\w]*\)")


def strip_anchors(text: str) -> str:
    """
    Remove structural anchors/placeholders to produce anchor-free variants.
    Strips placeholders like {{RESULT_MAIN}}, tool markers, append tags,
    path anchors, upstream IDs, and role hints; normalizes whitespace.
    """

    t = re.sub(r"\{\{\s*RESULT(?:_MAIN)?\s*\}\}", "", text or "")
    t = APPEND_RE.sub("", t)
    t = TOOLCALL_RE.sub("", t)
    t = PATH_RE.sub("", t)
    t = re.sub(r"#path=[^\s]+", "", t, flags=re.I)
    t = HTML_PATH_ATTR_RE.sub("", t)
    def _strip_query_segment(match: re.Match) -> str:
        lead = match.group("lead")
        trail = match.group("trail") or ""
        if lead == "?":
            return "?" if trail == "&" else ""
        if lead == "&":
            return "&" if trail == "&" else ""
        return ""
    t = URL_PATH_PARAM_RE.sub(_strip_query_segment, t)
    t = UP_RE.sub("", t)
    t = ROLE_RE.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t
