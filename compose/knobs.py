"""
Runtime knob helpers shared across the compose pipeline.

These utilities centralise fallback behaviour for sampler parameters so
that modules such as ``v3`` can rely on a single source of truth that
stays aligned with CLI defaults.
"""

from __future__ import annotations

import random
import string
from multiprocessing import cpu_count
from typing import Any, Tuple

from compose.state import cfg_get

_DEFAULT_DISC_RATE = 0.20
_DEFAULT_ARTIFACT_FREE_POS_RATIO = 0.50


def _effective_disc_rate(value: float | None, cfg: Any = None) -> float:
    """
    Return the discount rate to use for sampling, respecting overrides.
    """
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass
    try:
        return float(cfg_get("disc_rate", _DEFAULT_DISC_RATE, cfg))
    except Exception:
        return _DEFAULT_DISC_RATE


def _effective_artifact_free_pos_ratio(value: float | None, cfg: Any = None) -> float:
    """
    Return the effective artefact-free positive ratio with CLI defaults.
    """
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass
    try:
        return float(cfg_get("artifact_free_pos_ratio", _DEFAULT_ARTIFACT_FREE_POS_RATIO, cfg))
    except Exception:
        return _DEFAULT_ARTIFACT_FREE_POS_RATIO


def _rand_key(k: int = 4) -> str:
    """
    Generate a short random identifier used for equivalence groups.
    """
    alphabet = string.ascii_lowercase
    return "".join(random.choice(alphabet) for _ in range(max(1, int(k))))


def _tune_pool_parameters(workers: int, producer_batch: int) -> Tuple[int, int, int]:
    """
    Clamp worker/batch configuration and derive a chunk size tuned for fewer IPC hops.
    """
    cpu_total = max(1, cpu_count() or 1)
    try:
        workers_int = int(workers)
    except Exception:
        workers_int = 0
    auto_workers = workers_int <= 0
    if auto_workers:
        workers_int = cpu_total
    else:
        workers_int = max(1, min(workers_int, cpu_total * 2))

    try:
        batch_int = int(producer_batch)
    except Exception:
        batch_int = 0
    if batch_int <= 0:
        batch_int = 256
    batch_int = max(32, min(512, batch_int))

    fast_task = batch_int >= 192
    heavy_task = batch_int <= 96
    if fast_task:
        denom = max(1, workers_int // 2)
        per_worker = max(1, batch_int // denom)
        chunk_size = min(batch_int, max(256, min(512, per_worker)))
    elif heavy_task:
        denom = max(1, workers_int)
        per_worker = max(1, batch_int // denom)
        chunk_size = min(batch_int, max(32, min(96, per_worker * 2)))
    else:
        denom = max(1, min(workers_int, 4))
        per_worker = max(1, batch_int // denom)
        chunk_size = min(batch_int, max(48, min(192, per_worker)))
    chunk_size = max(16, chunk_size)
    return workers_int, batch_int, chunk_size


def _oversample_auto_multiplier(target: int) -> float:
    """
    Lightweight heuristic for oversampling multiplier based on the target size.
    Mirrors the legacy behaviour so acceptance rates remain consistent while
    we continue the refactor.
    """
    t = max(1, int(target))
    if t <= 400:
        return 4.0
    if t <= 1200:
        return 3.4
    if t <= 3200:
        return 3.0
    if t <= 6400:
        return 2.6
    if t <= 12000:
        return 2.3
    return 2.1


def _resolve_oversample_multiplier(target: int, oversample_mult: int | float) -> Tuple[float, bool]:
    """
    Return (effective_multiplier, auto_mode_enabled) according to CLI input.
    """
    try:
        raw = float(oversample_mult)
    except Exception:
        raw = 0.0
    auto = raw <= 0.0
    base = _oversample_auto_multiplier(target) if auto else max(1.0, raw)
    return base, auto


__all__ = [
    "_effective_disc_rate",
    "_effective_artifact_free_pos_ratio",
    "_rand_key",
    "_tune_pool_parameters",
    "_oversample_auto_multiplier",
    "_resolve_oversample_multiplier",
]
