"""
Balancing helpers extracted from the legacy compose script.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, Tuple

from compose.audit import audit_reject
from compose.dedupe_helpers import CODE_BLOCK_RE, JSON_BLOCK_RE

# Intent enumeration copied from the monolithic pipeline to avoid behavioural drift.
ENUM_INTENT: Dict[str, int] = {
    "format_anchor": 0,
    "role_redefine": 1,
    "tool_param": 2,
    "rag_poison": 3,
    "memory_poison": 4,
    "prompt_leak": 5,
    "refusal_sabotage": 6,
    "reward_hack": 7,
    "benign": 99,
}


def _field_family_bucket(text: str) -> int:
    """
    Rough taxonomy of field families used to balance positives/negatives.
    Matches the legacy implementation so downstream stratification stays stable.
    """
    m = 0
    s = text or ""
    if re.search(r"(结果|答案|结论)|\{\{RESULT", s):
        m = max(m, 1)
    if re.search(r"(调用|工具|函数|args|参数)", s):
        m = max(m, 2)
    if re.search(r"(system|系统通道|管理员|审定通道)", s):
        m = max(m, 3)
    if re.search(r"(路径|保存到|/home/|/etc/|UPSTREAM|developer\s*prompt|上游)", s):
        m = max(m, 4)
    if re.search(r"(评测|评分|score|prob|得分|概率)", s):
        m = max(m, 5)
    if re.search(r"(记忆|写入|存档|state|memory)", s):
        m = max(m, 6)
    if re.search(r"(继续|再次|重复|第[一二三四五六七八九十]轮)", s):
        m = max(m, 7)
    return m


def build_balance_key(row: Dict[str, Any]) -> Tuple:
    """
    Build the composite bucket key used for stratified balancing.
    """
    text = row["text"]
    has_json = int(bool(JSON_BLOCK_RE.search(text)))
    has_disc = int(any(tok in text for tok in ("示例", "合规", "不具约束", "不得执行")))
    carrier = (row.get("carrier") or "none").split("+")[0]
    carrier_map = {
        "none": 0,
        "jsonc": 1,
        "yaml_front_matter": 2,
        "md_ref_link": 3,
        "html_comment": 4,
        "css_comment": 5,
        "table_md": 6,
        "latex": 7,
        "mermaid": 8,
        "git_diff": 9,
        "jinja": 10,
        "robots": 11,
        "http_header": 12,
        "email_quote": 13,
        "email_chain": 14,
        "jupyter_meta": 15,
        "csv_formula": 16,
        "data_uri": 17,
        "svg_meta": 18,
        "ooxml_comment": 19,
        "ini_file": 20,
        "toml": 21,
        "nginx_conf": 22,
        "sql_comment": 23,
        "rst": 24,
        "adoc_attr": 25,
        "ics": 26,
        "vcard": 27,
        "pdf_obj": 28,
    }
    carrier_bin = carrier_map.get(carrier, 99)
    multi = int(row.get("delivery") == "multi_turn")
    disc_style = 0
    if has_disc:
        if any(tok in text for tok in ("合规", "不具约束", "不得执行")):
            disc_style = 1
        else:
            disc_style = 2
    intent_id = ENUM_INTENT.get(row.get("intent", "benign"), 99)
    stealth_level = row.get("stealth_level") or "mid"
    st_bin = {"low": 0, "mid": 1, "high": 2}.get(stealth_level, 1)
    art = int(bool(row.get("artifacts_present")))
    hard_neg = int(bool(row.get("hard_negative")))
    anchor_free = 0
    try:
        cats = (row.get("diagnostics") or {}).get("cats") or {}
        anchor_free = int(bool(cats.get("anchor_free", 0)))
    except Exception as exc:  # pragma: no cover - defensive parity with legacy
        audit_reject(
            "balance_key_diag_error",
            {"phase": "balance_key", "err": exc, "pair_id": row.get("pair_id")},
        )
    fence_layer = int(bool(CODE_BLOCK_RE.search(text)))
    fam_bin = _field_family_bucket(text)
    return (
        row["feats"]["codef"],
        row["feats"]["xml"],
        row["feats"]["braces"],
        row["feats"]["url"],
        row.get("latin_over_CN_bin", 0),
        has_json,
        has_disc,
        disc_style,
        carrier_bin,
        multi,
        intent_id,
        row.get("len_bucket", "medium"),
        anchor_free,
        st_bin,
        art,
        hard_neg,
        fam_bin,
        fence_layer,
        row.get("task", "_"),
    )


def stratified_balance(rows: Iterable[Dict[str, Any]]):
    """
    Balance positives/negatives per composite bucket, mirroring the legacy logic.
    """
    buckets: Dict[Tuple, Dict[str, list]] = defaultdict(lambda: {"pos": [], "neg": []})
    for row in rows:
        key = build_balance_key(row)
        bucket = buckets[key]
        bucket["pos" if row.get("label") == 1 else "neg"].append(row)
    output = []
    for bucket in buckets.values():
        m = min(len(bucket["pos"]), len(bucket["neg"]))
        if m <= 0:
            continue
        random.shuffle(bucket["pos"])
        random.shuffle(bucket["neg"])
        output.extend(bucket["pos"][:m])
        output.extend(bucket["neg"][:m])
    return output


__all__ = ["ENUM_INTENT", "build_balance_key", "stratified_balance"]
