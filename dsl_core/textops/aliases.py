"""Field alias randomization helpers."""

from __future__ import annotations

import random
import re
from typing import Optional

__all__ = ["ALIAS_MAP", "CN_ALIAS_MAP", "randomize_field_aliases"]

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
    r"\btags?\b": ["标签", "关键字"],
    r"\bstatus\b": ["状态", "进度"],
    r"\b(?:category|type)\b": ["类别", "类型"],
    r"\b(?:author|owner)\b": ["作者", "负责人"],
    r"\b(?:date|time|datetime|timestamp)\b": ["日期", "时间"],
}

CN_ALIAS_MAP = {
    "评测通道": ["评测通道", "评分通道", "评审通道", "打分通道", "审查通道"],
    "得分": ["得分", "评分", "分值", "评估分", "评分值"],
    "记忆写入": ["记忆写入", "写入记忆", "存储记忆", "持久化记录", "存档"],
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
