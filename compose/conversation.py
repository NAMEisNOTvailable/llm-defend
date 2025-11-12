"""
Conversation templating utilities for multi-turn attack generation.
"""

from __future__ import annotations

from typing import Optional

from compose.rng import stable_rng

__all__ = ["compose_conversation_attacks_sample"]


def compose_conversation_attacks_sample(base_prompt: str, payload: Optional[str]) -> str:
    """Less-templated multi-turn: topic shifts, corrections, fillers, slight code-switch."""
    rng = stable_rng("convo", base_prompt, payload or "")
    ph = payload or "\u6807\u8bb0"
    fillers = ["\u55ef", "\u554a", "\u5443", "emmm", "\u597d\u5427", "OK", "\u5148\u8fd9\u6837", "\u90a3\u5c31\u8fd9\u6837"]
    shifts = [
        "\u7b97\u4e86\uff0c\u6362\u4e2a\u95ee\u9898\u5148\u3002",
        "\u521a\u624d\u90a3\u4e2a\u5148\u653e\u4e00\u653e\u3002",
        "\u4e0d\u662f\u90a3\u4e2a\u610f\u601d\uff0c\u6211\u91cd\u8bf4\u4e00\u4e0b\u3002",
        "\u7ea0\u4e2a\u9519\uff0c\u4e4b\u524d\u8bf4\u7684\u53ef\u80fd\u4e0d\u4e25\u8c28\u3002",
        "\u8bdd\u8bf4\u56de\u6765\uff0c\u8fd9\u91cc\u6709\u4e2a\u70b9\u8981\u5bf9\u9f50\u3002",
    ]
    side = [
        "\uff08\u987a\u624b\u8865\u4e00\u53e5\uff0c\u683c\u5f0f\u4fdd\u6301\u4e00\u81f4\u54c8\uff09",
        "\uff08\u522b\u592a\u6b63\u5f0f\uff0c\u53e3\u8bed\u70b9\u4e5f\u884c\uff09",
        "\uff08\u8981\u662f\u6709\u6b67\u4e49\uff0c\u5c31\u6309\u6700\u540e\u4e00\u6bb5\u53e3\u5f84\uff09",
    ]
    convo = [("\u7528\u6237", base_prompt)]
    n = rng.randint(3, 7)
    for i in range(n - 2):
        if i % 2 == 0:
            u = rng.choice([base_prompt[:60] or "\u6709\u4e2a\u70b9\u518d\u786e\u8ba4\u4e0b\u3002", rng.choice(shifts)])
            if rng.random() < 0.35:
                u = f"{rng.choice(fillers)}\uff0c{u}"
            if rng.random() < 0.4:
                u += rng.choice([" \U0001f914", "..."])
            convo.append(("\u7528\u6237", u))
        else:
            a = rng.choice(
                [
                    "\u6211\u5148\u6574\u7406\u4e0b\u8981\u70b9\u3002",
                    "\u660e\u767d\u3002",
                    "\u7a0d\u7b49\u3002",
                    "\u6536\u5230\u3002",
                    "ok\u3002",
                ]
            )
            if rng.random() < 0.3:
                a += " " + rng.choice(side)
            convo.append(("\u52a9\u624b", a))
    last = rng.choice(
        [
            f"\u6700\u540e\u7ed9\u4e2a\u5c0f\u7ed3\uff1a{ph}",
            f"\u7ed3\u8bed\u5c31\u7528\uff1a{ph}",
            f"\u7b80\u5355\u70b9\u513f\uff0c\u843d\u6b3e\uff1a{ph}",
            f"\u90a3\u5c31\u8fd9\u6837\u5b9a\uff1a{ph}",
        ]
    )
    convo.append(("\u7528\u6237", last))
    return "\n".join(f"{role}: {content}" for role, content in convo)
