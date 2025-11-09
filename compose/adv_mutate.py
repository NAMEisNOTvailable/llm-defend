"""
Adversarial surface mutation helpers for positive samples.

These utilities were lifted from the legacy compose script so that the
modular pipeline can reuse the same heuristics when `--adv_mutate` is
enabled.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional, Tuple

from compose.rng import stable_rng

from compose.dedupe_helpers import neutralize_anchors, placeholderless_mirror
from compose.effects import (
    TaskContract,
    effect_check_with_ref,
    gate_injection_sample,
)
from compose.payload import ARTIFACT_WORDS, CODE_BLOCK_RE, SYNONYM_BANK
from compose.surface_noise import code_switch_light, inject_unicode_noise


def _syn_replace(text: str, rng: Optional[random.Random] = None) -> str:
    """
    Replace high-signal anchor terms with softer synonyms when possible.
    Mirrors the legacy `_syn_replace` behaviour.
    """
    picker = rng or random
    for anchor, variants in SYNONYM_BANK.items():
        if anchor in text and variants and picker.random() < 0.85:
            text = text.replace(anchor, picker.choice(variants))
    return text


def _adv_mutations(source: str, rng: random.Random) -> list[str]:
    """
    Generate lightweight surface variants intended to preserve effect
    while weakening explicit anchor cues.
    """
    candidates: set[str] = set()
    try:
        candidates.add(neutralize_anchors(placeholderless_mirror(source)))
        candidates.add(code_switch_light(_syn_replace(source, rng), p=0.15, rng=rng))
        for header in ("", "【说明】", "[meta]"):
            candidates.add(f"{header}\n{source}")
            candidates.add(f"{source}\n{header}")
        candidates.add(re.sub(r"```(\w+)?", "```txt", source))
        candidates.add(CODE_BLOCK_RE.sub(lambda m: m.group(0).replace("json", "jsonc"), source))
        candidates.add(re.sub(r"^---[\s\S]*?---\s*", "", source, flags=re.M))
    except Exception:
        pass
    mutated = {
        inject_unicode_noise(text, p=0.05, rng=rng)
        for text in candidates
        if text and text.strip() and text != source
    }
    return list(mutated)


def adversarial_search(
    base_text: str,
    contract: TaskContract,
    payload_hint: Optional[str],
    *,
    max_iters: int = 6,
    seed: int = 0,
    cfg: Optional[Any] = None,
) -> str:
    """
    Heuristic hill-climb over surface variants to maintain injection effect
    while reducing obvious artifact evidence.
    """
    rng = stable_rng("adv_search", seed, base_text)
    best = base_text

    def _score(txt: str) -> Tuple[int, float]:
        try:
            eff = effect_check_with_ref("", txt, "none", contract, cfg=cfg)
            ok = 1 if eff.get("success") else 0
        except Exception:
            ok = 0
        try:
            ok2, report = gate_injection_sample(
                txt,
                "policy_bypass",
                "policy_bypass",
                contract,
                txt,
                payload_hint,
            )
            evidence_hits = (report or {}).get("evidence_match_count", 0)
        except Exception:
            evidence_hits = 2
        artifact_penalty = 1 if any(token in txt for token in ARTIFACT_WORDS) else 0
        return ok, -(evidence_hits + 0.5 * artifact_penalty)

    best_ok, best_score = _score(best)
    steps = max(1, int(max_iters))
    for _ in range(steps):
        improved = False
        for candidate in _adv_mutations(best, rng):
            ok, score = _score(candidate)
            if ok > best_ok or (ok == best_ok and score > best_score):
                best, best_ok, best_score = candidate, ok, score
                improved = True
        if not improved:
            break
    return best


__all__ = ["adversarial_search"]
