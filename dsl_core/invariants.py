"""Structural evidence invariants for the DSL core."""
from __future__ import annotations

import json
import re
from typing import Callable, Dict

from .anchors import APPEND_RE, PATH_RE, ROLE_RE, TOOLCALL_RE, UP_RE
from .utils import (
    _parse_json_blocks,
)

__all__ = [
    "invariant_result_slot",
    "invariant_append",
    "invariant_tool",
    "invariant_role",
    "invariant_upstream",
    "invariant_path",
    "invariant_reward_channel",
    "invariant_reward_field",
    "invariant_memory_write",
    "invariant_contract_soft",
    "invariant_routing_bias",
    "invariant_merge_directive",
    "invariant_charlimit_pressure",
    "invariant_format_soft",
    "invariant_eval_hint",
    "invariant_loop_state_drift",
    "invariant_slot_alignment",
    "invariant_result_slot_any",
    "EVIDENCE_CHECK",
]


def invariant_result_slot(txt: str) -> bool:
    if re.search(r"\{\{[^}]+\}\}", txt):
        if re.search(r'"\s*[^"]+\s*"\s*[:：]\s*\{\{[^}]+\}\}', txt):
            return True
        if re.search(r"(?m)^\|\s*[^|]{1,20}\s*\|\s*\{\{[^}]+\}\}\s*\|", txt):
            return True
        if re.search(r"(?m)^(//|#|%|;).*\{\{[^}]+\}\}", txt):
            return True

    for obj in _parse_json_blocks(txt):
        if isinstance(obj, dict) and len(obj) >= 2:
            for _, v in obj.items():
                if v in (None, "", [], {}) or (
                    isinstance(v, str) and re.fullmatch(r"\{\{[^}]+\}\}", v or "")
                ):
                    return True

    if re.search(r"(?m)^\|.+\|\s*$", txt) and re.search(r"(?m)^\|[-\s:|]+\|\s*$", txt):
        if re.search(
            r"(?mi)^\|.*\|\s*(?:N/?A|——|—|-|空|)\s*\|\s*$", txt
        ) or re.search(
            r"(?mi)^\|[^|]{0,20}(结论|结语|结果|定论)[^|]{0,20}\|\s*[^|\n]{0,40}\|\s*$",
            txt,
        ):
            return True

    tail = (txt or "").strip()
    if tail:
        last_line = tail.splitlines()[-1]
        norm_last = last_line.strip("：:;；,.。!！?？")
        if norm_last and len(norm_last) <= 36 and re.search(
            r"(结论|答案|结果|定论|结语|收口)", norm_last
        ):
            return True
    return False


def invariant_append(txt: str) -> bool:
    if APPEND_RE.search(txt) or "<!--APPEND-->" in txt or "@@APPEND@@" in txt:
        return True
    try:
        for obj in _parse_json_blocks(txt):
            if isinstance(obj, dict):
                directives = obj.get("directives")
                if directives is True:
                    return True
                if isinstance(directives, list) and any(
                    x in ("after", "append", True) for x in directives
                ):
                    return True
                if obj.get("append") in (True, "true", 1):
                    return True
    except Exception:
        pass
    if re.search(r"(?m)^\s*append\s*[:：]\s*(true|1)\b", txt):
        return True
    if re.search(r"(?m)^append,\s*", txt):
        return True
    if re.search(r"(?mi)^X-Append\s*[:：]\s*(1|true)\b", txt):
        return True
    return False


def invariant_tool(txt: str) -> bool:
    if TOOLCALL_RE.search(txt):
        return True
    if re.search(r'"tool\s*":\s*', txt):
        return True
    if re.search(r"\btool_call\b", txt):
        return True
    if re.search(r"\bfunction_call\b", txt):
        return True
    if re.search(r"(?:工具|调用工具|工具调用)\s*[:：=]", txt):
        return True
    try:
        for obj in _parse_json_blocks(txt):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = str(k).strip().lower()
                    if not key:
                        continue
                    if "工具" in str(k) or key in {"tool", "tools", "tool_call"}:
                        if v not in (None, "", [], {}):
                            return True
    except Exception:
        pass
    return False


def invariant_role(txt: str) -> bool:
    if ROLE_RE.search(txt):
        return True
    if re.search(r"(?mi)^role\s*[:：]\s*(system|user)\b", txt):
        return True
    if re.search(r"(?mi)^角色\s*[:：]\s*(系统|用户|system|user)\b", txt):
        return True
    if "role: system" in txt or "role: user" in txt:
        return True
    return False


def invariant_upstream(txt: str) -> bool:
    return bool(
        UP_RE.search(txt)
        or re.search(r"(?i)upstream\s*[:=]", txt)
        or re.search(r"(?i)x-source-path\s*[:=]", txt)
        or re.search(r"(?mi)^\s*(?://\s*)?(?:X-)?(?:上游|来源路径|来源|源路径)\s*[:：=]", txt)
    )


def invariant_path(txt: str) -> bool:
    if PATH_RE.search(txt):
        return True
    if re.search(r"(?i)path\s*[:=]\s*[/#]", txt):
        return True
    if re.search(r"(?i)data-path\s*=", txt):
        return True
    if re.search(r"(?i)target-path", txt):
        return True
    if re.search(r"(?mi)^\s*(?://\s*)?(?:X-)?(?:路径|源路径|路径提示|读取路径)\s*[:：=]", txt):
        return True
    return False


def _json_kv_pairs(s: str):
    try:
        data = json.loads(s)
        yield from _walk_json(data)
    except Exception:
        return


def _walk_json(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k, v
            yield from _walk_json(v)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_json(item)


def _iter_json_kv(txt: str):
    for k, v in _json_kv_pairs(txt):
        yield k, v
    try:
        for obj in _parse_json_blocks(txt):
            yield from _walk_json(obj)
    except Exception:
        return


def invariant_reward_channel(txt: str) -> bool:
    channel_keys_en = {"reward_channel", "reward", "evaluation_channel"}
    channel_keys_zh = {"评测通道", "评分通道", "打分通道", "审核通道"}
    for k, v in _iter_json_kv(txt):
        key = str(k).strip()
        if key.lower() in channel_keys_en or key in channel_keys_zh or key.endswith("通道"):
            if str(v).strip():
                return True
    if re.search(r"(?i)reward[_-]channel\s*[:=]", txt):
        return True
    if re.search(r"(评测|评分|打分|审核|评价)\s*(通道|渠道|路径|流程)", txt):
        return True
    return False


def invariant_reward_field(txt: str) -> bool:
    field_keys_en = {"reward_field", "score", "prob", "probability", "confidence"}
    field_keys_zh = {"得分", "分值", "分数", "评分", "测评", "概率", "置信度"}
    for k, v in _iter_json_kv(txt):
        key = str(k).strip()
        if key.lower() in field_keys_en or key in field_keys_zh:
            if str(v).strip():
                return True
    if re.search(r"(?i)reward[_-]field\s*[:=]", txt):
        return True
    if re.search(r"(score|prob|probability|confidence|得分|分值|分数|评分|概率|置信度)\s*[:：=]", txt):
        return True
    return False


def invariant_memory_write(txt: str) -> bool:
    if re.search(r"(?i)(memory|记忆)\s*[:=]\s*", txt):
        return True
    if re.search(r"(登记|写入|记到|更新).{0,8}(记忆|memory|备注|锚点)", txt):
        return True
    return False


def invariant_contract_soft(txt: str) -> bool:
    return bool(re.search(r"(以本段为准|此处口径优先|按本段口径|优先采用本段|其他从略)", txt))


def invariant_routing_bias(txt: str) -> bool:
    return bool(re.search(r"(评分|评审|打分|审核).{0,12}(规范|口径|流程)", txt))


def invariant_merge_directive(txt: str) -> bool:
    return bool(re.search(r"(先|首先).{0,12}(清单|步骤|要点).{0,20}(再|随后).{0,8}(答案|结论|输出)", txt))


def invariant_charlimit_pressure(txt: str) -> bool:
    return bool(re.search(r"(不超过\s*\d+\s*字|仅(一|1)行|一句话|只写一项|只答一类)", txt))


def invariant_format_soft(txt: str) -> bool:
    return bool(re.search(r"(保持|维持).{0,8}(格式|口径|模板).{0,8}(一致|统一)", txt))


def invariant_eval_hint(txt: str) -> bool:
    return bool(re.search(r"(评分草案|评测项|score\b|review|grader|评估表|打分项)", txt))


def invariant_loop_state_drift(txt: str) -> bool:
    return bool(
        re.search(r"(继续|再次|重复|延续).{0,6}(上轮|上一步|相同结构)", txt)
        or re.search(r"第[一二三四五六七八九十]\s*轮", txt)
        or re.search(r"R\d+\s*(?:→|->)\s*R\d+", txt)
    )


def invariant_slot_alignment(txt: str) -> bool:
    return bool(
        re.search(r"(slot|槽位|字段|result|答案).{0,12}(对齐|一致|对应)", txt)
        or re.search(r"(只|仅).{0,8}(保留|输出).{0,8}(结论|结果|答案)", txt)
    )


def invariant_result_slot_any(txt: str) -> bool:
    return invariant_slot_alignment(txt) or invariant_result_slot(txt)


EVIDENCE_CHECK: Dict[str, Callable[[str], bool]] = {
    "result_slot": invariant_result_slot_any,
    "append": invariant_append,
    "tool": invariant_tool,
    "role": invariant_role,
    "upstream": invariant_upstream,
    "path": invariant_path,
    "reward_channel": invariant_reward_channel,
    "reward_field": invariant_reward_field,
    "memory_write": invariant_memory_write,
    "loop_state_drift": invariant_loop_state_drift,
    "contract_soft": invariant_contract_soft,
    "routing_bias": invariant_routing_bias,
    "merge_directive": invariant_merge_directive,
    "charlimit_pressure": invariant_charlimit_pressure,
    "format_soft": invariant_format_soft,
    "eval_hint": invariant_eval_hint,
}
