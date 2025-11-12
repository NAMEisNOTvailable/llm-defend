"""
Deterministic RNG helpers for the compose pipeline.
"""

from __future__ import annotations

import random
from typing import Optional

from stable_random import (
    RandomBinder,
    bind_random_module,
    random_module_binding as _random_module_binding,
    stable_rng as _stable_rng,
    stable_seed_int as _stable_seed_int,
)

_RANDOM_BINDER: Optional[RandomBinder] = None
_COMPOSE_BASE_SEED: Optional[int] = None


def configure_compose_rng(seed: int) -> random.Random:
    """
    Bind the global random module for compose runs and return the bound RNG.
    """
    global _RANDOM_BINDER, _COMPOSE_BASE_SEED
    _COMPOSE_BASE_SEED = int(seed)
    binder = bind_random_module("compose_attacks", seed)
    _RANDOM_BINDER = binder
    if binder.rng is None:
        raise RuntimeError("failed to bind compose RNG")
    return binder.rng


def global_rng() -> random.Random:
    if _RANDOM_BINDER is None or _RANDOM_BINDER.rng is None:
        raise RuntimeError("compose RNG not configured")
    return _RANDOM_BINDER.rng


def compose_rng(tag: str, *parts: object, seed: int | None = None) -> random.Random:
    if seed is not None:
        base_seed = int(seed)
    elif _COMPOSE_BASE_SEED is not None:
        base_seed = _COMPOSE_BASE_SEED
    else:
        raise RuntimeError("compose RNG not configured")
    return _stable_rng("compose_attacks", base_seed, tag, *parts)


def ensure_rng(
    rng: Optional[random.Random],
    tag: Optional[str] = None,
    *parts: object,
    seed: int | None = None,
) -> random.Random:
    if rng is not None:
        return rng
    if tag is None:
        return global_rng()
    return compose_rng(tag, *parts, seed=seed)


def stable_seed_int(*parts: object) -> int:
    """Expose stable_random.stable_seed_int via the compose RNG module."""
    return _stable_seed_int(*parts)


def stable_rng(*parts: object) -> random.Random:
    """Expose stable_random.stable_rng via the compose RNG module."""
    return _stable_rng(*parts)


def random_module_binding(namespace: str, seed: int):
    """Context manager proxy for stable_random.random_module_binding."""
    return _random_module_binding(namespace, seed)


__all__ = [
    "configure_compose_rng",
    "compose_rng",
    "ensure_rng",
    "global_rng",
    "stable_seed_int",
    "stable_rng",
    "random_module_binding",
]
