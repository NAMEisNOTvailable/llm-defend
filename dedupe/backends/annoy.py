# -*- coding: utf-8 -*-
"""Optional Annoy backend helpers."""
from __future__ import annotations

try:
    from annoy import AnnoyIndex as _AnnoyIndex  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _AnnoyIndex = None


def available() -> bool:
    return _AnnoyIndex is not None


def create_index(dim: int, metric: str = "angular"):
    if _AnnoyIndex is None or dim <= 0:
        return None
    try:
        return _AnnoyIndex(dim, metric)  # type: ignore[call-arg]
    except Exception:
        return None


__all__ = ["available", "create_index"]

