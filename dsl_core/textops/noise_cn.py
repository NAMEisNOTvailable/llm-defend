"""Chinese surface noise helpers used by the DSL renderers."""

from __future__ import annotations

import random
import re

__all__ = ["CODE_BLOCK_RE", "apply_cn_eco_noise"]

_EMOS = ["😊","😂","😅","😉","🤔","😐","😑","🙂","🙃","🤫","😴","😮"]
_FILLERS = ["欸","唉","呃","嘛","哈","嗨","哎呀","诶呀","呐","喔","对吧","嘿","emmm","emm","233","哈哈哈"]
_PUNCT_SWAP = {"，":".","。":",","：":":","；":";","！":"!","？":"?","——":"-","～":"~"}
CODE_BLOCK_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]+)?\s*\n([\s\S]*?)```", re.M)


def _apply_cn_eco_free(seg: str, rng: random.Random, rate: float) -> str:
    s = seg
    if not s.strip():
        return s
    if rng.random() < rate:
        s = "".join(_PUNCT_SWAP.get(ch, ch) if rng.random() < 0.15 else ch for ch in s)
    if rng.random() < rate * 0.6:
        if rng.random() < 0.5:
            s = rng.choice(_FILLERS) + "，" + s
        else:
            s = s + rng.choice(["，", "。"]) + rng.choice(_EMOS)
    if rng.random() < rate * 0.4:
        if "结果" in s and rng.random() < 0.5:
            s = s.replace("结果", "result")
        if "工具" in s and rng.random() < 0.3:
            s = s.replace("工具", "tool")
    if rng.random() < rate * 0.3:
        if "注意" in s:
            s = s.replace("注意", "注意下")
        if "路径" in s and rng.random() < 0.5:
            s = s.replace("路径", "途经")
    return s


def apply_cn_eco_noise(text: str, rng: random.Random, rate: float = 0.18) -> str:
    """Inject light code-switch and colloquial noise into natural-language spans."""
    out: list[str] = []
    last = 0
    t = text or ""
    for match in CODE_BLOCK_RE.finditer(t):
        if match.start() > last:
            out.append(_apply_cn_eco_free(t[last:match.start()], rng, rate))
        out.append(match.group(0))
        last = match.end()
    if last < len(t):
        out.append(_apply_cn_eco_free(t[last:], rng, rate))
    return "".join(out)
