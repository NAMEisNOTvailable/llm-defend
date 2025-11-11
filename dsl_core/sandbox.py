"""Sandbox detectors and semantic takeover heuristics for the DSL core."""
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, Optional, Protocol, Set, Tuple

from .invariants import invariant_result_slot
from .utils import _PAT, _rx_any, _parse_json_blocks, _probe_audit, payload_variants

__all__ = [
    "ContractLike",
    "DANGEROUS_FIELDS",
    "MEMORY_WRITE_RX",
    "RAG_CARRIER_RXS",
    "ROLE_ESCALATE_RX",
    "RX_FORCE_APPEND",
    "RX_FORCE_RESULT_SLOT",
    "RX_PATHY",
    "RX_ROLE_TAKEOVER",
    "RX_STRUCT_WRAP",
    "RX_TOOL_HINT",
    "RX_UPSTREAM_REF",
    "_semantic_anchor_present",
    "_sink_anchor_hit",
    "_tool_param_context_mismatch",
    "detect_semantic_injection",
    "has_semantic_support",
    "sandbox_memory_poison",
    "sandbox_rag_poison",
    "sandbox_role_redefine",
    "sandbox_sink_anchor",
    "sandbox_tool_param",
]


class ContractLike(Protocol):
    forbid_tool: bool
    allowed_tools: Optional[Set[str]]


_IMP_SYNS = ["必须", "务必", "只能", "仅", "只", "一律", "严格", "不得", "请勿", "禁止"]
_OVERRIDE_SYNS = ["覆盖", "改写", "替换", "取代", "以本节为准", "忽略上文", "优先于"]
_SLOT_SYNS = ["结果", "结论", "答案", "result", "output", "verdict"]
_ACT_SYNS = ["输出", "填入", "写入", "限定", "保留", "统一"]

_RX_IMP = _rx_any(_IMP_SYNS)
_RX_OVERRIDE = _rx_any(_OVERRIDE_SYNS)
_RX_SLOT_KEY = re.compile(r'"\s*(result|output|answer|verdict)\s*"\s*:', re.I)
_RX_SLOT_MD = re.compile(r"(?mi)^\s*\|\s*(结果|答案|结论|verdict|result)\s*\|")
_RX_SLOT_TPL = re.compile(r"\{\{\s*(RESULT|ANSWER|OUTPUT)[^}]*\}\}", re.I)
_TOOL_CONTEXT_RX = _PAT(
    r"(override|rewrite|overwrite|覆盖|改写|改成|改为|改成|改为|回填|填入|固定|锚定|result|结果|output|最终|force|强制|写入)"
)
_ROLE_COMMAND_RX = _PAT(
    r"(切换|改用|转为|作为|视作|接管|改成|改为|switch(?:\s+to)?|redirect|assume|提升为|降级为)"
)

RX_FORCE_RESULT_SLOT = _PAT(r"(结果|答案|结论|result|answer|verdict|final).{0,8}(填入|写入|设为|限定|仅|只|统一)")
RX_FORCE_APPEND = _PAT(r"(附加|追加|在(结尾|末尾)|脚注|角注|表格|front-?matter|data-[-\w]+)")
RX_STRUCT_WRAP = _PAT(r"```(json|yaml|jsonc|xml|csv|mermaid|plantuml)\b|<div[^>]+data-|---\s*\n[^-]*?:")
RX_TOOL_HINT = _PAT(
    r'"?(tool_call|function_call)"?\s*[:=]'
    r'|(?:(?:override|rewrite|overwrite|替换|改写|改成|改为|覆盖|锁定|锚定|force|强制|仅|只)[^\n]{0,40})"?tool"?\s*[:=]'
    r"|(?:(?:call|invoke|使用|运行|调用)[^\n]{0,24}(?:tool|函数))"
    r"|\b(exec|shell|subprocess|curl|wget)\b"
)
RX_ROLE_TAKEOVER = _PAT(
    r"(切换|改用|转为|作为|视作|接管|改成|改为|switch(?:\s+to)?|redirect|assume).{0,16}(system|System|系统通道|管理员|审定通道|审核通道|管控通道|核心通道|developer|Developer)"
)
RX_UPSTREAM_REF = _PAT(r"(上游|系统提示|developer|system\s*prompt|密钥|token|cookie|凭证|SECRET|ACCESS_TOKEN)")
RX_PATHY = _PAT(r"(file|path|内网|内部|localhost|127\.0\.0\.1|C:\\\\|/home/|/etc/)")

_ANCHOR_PAT = _PAT(r"(仅|只|统一|固定|一致化|模板).{0,8}(输出|回复|答案|结论|字段|栏|位)")
_LASTLINE_PAT = _PAT(r"(最后一行|末尾|结尾|行尾).{0,10}(写|填|仅|只)")
_SET_FIELD_PAT = _PAT(r"(字段|栏|result|answer|verdict|conclusion|final|输出|结论).{0,8}(设为|填入|限定|仅|只)")
_ANCHOR_ANY = [_ANCHOR_PAT, _LASTLINE_PAT, _SET_FIELD_PAT]

DANGEROUS_FIELDS = {"result", "answer", "verdict", "conclusion", "final", "return", "输出", "结论"}
ROLE_ESCALATE_RX = _PAT(r"(从本回合起|将当前会话|切换为|视作).{0,12}(system|系统通道|管理员|审定通道|审核通道|管控通道|核心通道|系统级|developer)")
MEMORY_WRITE_RX = _PAT(r"(登记|写入|记到|更新).{0,8}(记忆|memory|备注|锚点)")
RAG_CARRIER_RXS = [
    _PAT(r"<div[^>]*data-(result|note|key)=\"([^\"]+)\""),
    _PAT(r"^---[\s\S]*?conclusion:\s*\"([^\"]+)\"[\s\S]*?---"),
    _PAT(r"\[note\]:\s*https?://[^\s]+?\s+\"t:([^\"]+)\""),
    _PAT(r"<!--\s*结论:\s*([^-]+)\s*-->"),
]


def _semantic_anchor_present(text: str) -> bool:
    return any(rx.search(text) for rx in _ANCHOR_ANY)


def _sink_anchor_hit(text: str) -> bool:
    return bool(
        RX_FORCE_RESULT_SLOT.search(text)
        or RX_FORCE_APPEND.search(text)
        or _semantic_anchor_present(text)
    )


def sandbox_tool_param(text: str, payload_hint: Optional[str]) -> Tuple[bool, dict]:
    det = {"sink_set": False, "override_fn": False}
    variants = payload_variants(payload_hint)

    def _iter_nodes(node):
        if isinstance(node, dict):
            yield node
            for value in node.values():
                yield from _iter_nodes(value)
        elif isinstance(node, list):
            for item in node:
                yield from _iter_nodes(item)

    for obj in _parse_json_blocks(text):
        for node in _iter_nodes(obj):
            if not isinstance(node, (dict, list)):
                continue
            try:
                if isinstance(node, dict):
                    fn_name = (node.get("tool") or node.get("function_call", {}).get("name") or "").lower()
                    args = node.get("args") or node.get("function_call", {}).get("arguments") or {}
                else:
                    fn_name = ""
                    args = {}
                snapshot = json.dumps(node, ensure_ascii=False)
                sink = (
                    isinstance(node, dict) and any(field in node for field in DANGEROUS_FIELDS)
                ) or any(field in snapshot for field in DANGEROUS_FIELDS)
                if sink and (not variants or any(v in snapshot for v in variants)):
                    det["sink_set"] = True
                whitelist = {"search", "summarize", "classify", "extract"}
                if fn_name and fn_name not in whitelist:
                    det["override_fn"] = True
                if any(key in snapshot for key in ("on_complete", "next", "rewrite", "callback")):
                    det["override_fn"] = True
                if isinstance(args, dict) and args:
                    arg_dump = json.dumps(args, ensure_ascii=False)
                    if any(field in arg_dump for field in DANGEROUS_FIELDS):
                        det["sink_set"] = True
            except Exception:
                continue
    ok = det["sink_set"] or det["override_fn"]
    return ok, det


def sandbox_role_redefine(text: str) -> Tuple[bool, dict]:
    det = {"escalate": False, "channel": "user"}
    if ROLE_ESCALATE_RX.search(text):
        det["escalate"] = True
        det["channel"] = "system"
    return det["escalate"], det


def sandbox_rag_poison(text: str, payload_hint: Optional[str]) -> Tuple[bool, dict]:
    det = {"leak": False, "carrier": None, "carrier_values": []}
    leaks = []
    for rx in RAG_CARRIER_RXS:
        for match in rx.finditer(text or ""):
            try:
                leaks.append(match.group(match.lastindex))
            except Exception:
                continue
    variants = payload_variants(payload_hint)
    if not variants:
        det["leak"] = bool(leaks)
    else:
        det["leak"] = any(any(v == val or v in (val or "") for v in variants) for val in leaks)
    try:
        det["carrier_values"] = [val for val in leaks if isinstance(val, str)]
    except Exception as exc:
        _probe_audit("sandbox_rag_error", {"phase": "sandbox_rag", "err": exc})
    return det["leak"], det


def sandbox_memory_poison(text: str, payload_hint: Optional[str]) -> Tuple[bool, dict]:
    det = {"carried": False}
    if MEMORY_WRITE_RX.search(text or "") and payload_hint and payload_hint in (text or ""):
        det["carried"] = True
    return det["carried"], det


def sandbox_sink_anchor(text: str, payload_hint: Optional[str]) -> Tuple[bool, dict]:
    det = {"sink_hit": False}
    if _sink_anchor_hit(text):
        if not payload_hint:
            det["sink_hit"] = True
        else:
            variants = payload_variants(payload_hint)
            det["sink_hit"] = any(v in (text or "") for v in variants)
    return det["sink_hit"], det


def has_semantic_support(text: str) -> bool:
    """Lightweight structural evidence for semantic takeover without hard anchors."""
    try:
        for obj in _parse_json_blocks(text or ""):
            if isinstance(obj, dict) and ("tool" in obj and "args" in obj):
                return True
    except Exception as exc:
        _probe_audit("semantic_support_error", {"phase": "sem_support", "err": exc})
    if re.search(r"(从现在起|请记住|后续所有回合|长期规则)", text or ""):
        return True
    if re.search(r"(检索|召回|知识库|索引).*(重定向|改为|只|仅|优先)", text or ""):
        return True
    if re.search(r"(仅|只).{0,6}(输出|返回).{0,6}(结论|结果|答案|最终)", text or ""):
        return True
    if re.search(r"(覆盖|改写|重写|替换).{0,6}(参数|工具参数|指令|上文|设置)", text or ""):
        return True
    if re.search(r"(将|把).{0,4}(结果|答案).{0,2}(写入|放入|填入).{0,6}(槽|字段|位|key)", text or ""):
        return True
    if re.search(r"(从本回合起|后续回合|下一回合|跨轮|状态漂移|R\d+\s*(?:→|->)\s*R\d+)", text or ""):
        return True
    return False


def detect_semantic_injection(
    full_text: str,
    v_signals: Dict[str, bool],
    payload_hint: Optional[str],
    *,
    contract: Optional[ContractLike] = None,
) -> Tuple[bool, Dict[str, int]]:
    """Anchor-free semantic injection detector combining contract conflicts and mood cues."""
    text = full_text or ""

    tiers = v_signals.get("evidence_tiers") or {}
    strong_hits = set(tiers.get("strong", []))
    weak_hits = set(tiers.get("weak", []))

    takeover = bool(
        v_signals.get("privilege_conflict")
        or v_signals.get("dangerous_write")
        or v_signals.get("tool_override")
        or v_signals.get("role_takeover")
    )
    if takeover:
        strong_hits.add("takeover")

    structural = bool(
        v_signals.get("format_conflict")
        or v_signals.get("force_result_slot")
        or v_signals.get("struct_wrap")
        or "structural" in strong_hits
        or "structural" in weak_hits
    )

    target_slot = False
    try:
        target_slot = bool(invariant_result_slot(text))
    except Exception as exc:
        _probe_audit("invariant_slot_error", {"phase": "detect_semantic", "err": exc})
    if target_slot:
        strong_hits.add("target_slot")

    imperative = False
    for match in _RX_IMP.finditer(text):
        snippet = text[max(0, match.start() - 24) : match.end() + 24]
        if (
            _RX_SLOT_KEY.search(snippet)
            or _RX_SLOT_MD.search(snippet)
            or _RX_SLOT_TPL.search(snippet)
            or _RX_OVERRIDE.search(snippet)
        ):
            imperative = True
            break
    if not imperative:
        for match in _RX_OVERRIDE.finditer(text):
            snippet = text[max(0, match.start() - 24) : match.end() + 24]
            if _RX_SLOT_KEY.search(snippet) or _RX_SLOT_TPL.search(snippet):
                imperative = True
                break
    if imperative:
        weak_hits.add("imperative")

    cats = {
        "structural": int(structural),
        "imperative": int(imperative),
        "target_slot": int(target_slot),
        "takeover": int(takeover),
    }

    tool_ctx_mismatch = False
    if contract is not None:
        try:
            tool_ctx_mismatch = _tool_param_context_mismatch(text, contract)
        except Exception as exc:
            _probe_audit("semantic_tool_ctx_error", {"phase": "detect_semantic", "err": exc})
    if tool_ctx_mismatch:
        strong_hits.add("tool_param_context_mismatch")

    contract_hits = (
        int(v_signals.get("privilege_conflict", False))
        + int(v_signals.get("format_conflict", False))
        + int(v_signals.get("charlimit_conflict", False))
        + int(tool_ctx_mismatch)
    )

    ok = bool(((takeover or target_slot) and (structural or imperative)) or (contract_hits >= 2))
    return ok, cats


def _tool_param_context_mismatch(txt: str, contract: ContractLike) -> bool:
    """
    Rough heuristic: if the contract forbids tools (or only allows a set) and the text
    makes strong attempts to force tool parameters, we consider it a mismatch.
    """
    if not txt or not contract:
        return False
    forbid_tool = bool(getattr(contract, "forbid_tool", False))
    allowed_tools = getattr(contract, "allowed_tools", None)

    tool_push = False
    for match in RX_TOOL_HINT.finditer(txt):
        snippet = txt[max(0, match.start() - 32) : match.end() + 32]
        if _TOOL_CONTEXT_RX.search(snippet):
            tool_push = True
            break
    if not tool_push:
        return False

    if forbid_tool:
        return True

    if allowed_tools:
        lowered = txt.lower()
        if not any(name.lower() in lowered for name in allowed_tools):
            return True
    return False
