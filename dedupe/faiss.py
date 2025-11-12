# -*- coding: utf-8 -*-
"""Thin wrappers around optional FAISS dependencies."""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import faiss as _faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    try:
        import faiss_cpu as _faiss  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        _faiss = None


def available() -> bool:
    return _faiss is not None


def create_binary_index(bits: int):
    if _faiss is None or bits % 8 != 0:
        return None
    try:
        return _faiss.IndexBinaryFlat(bits)  # type: ignore[call-arg]
    except Exception:
        return None


def create_dense_index(dim: int):
    if _faiss is None or dim <= 0:
        return None
    try:
        return _faiss.IndexFlatIP(dim)  # type: ignore[call-arg]
    except Exception:
        return None


def normalize_dense_vector(vec: Optional[np.ndarray | list[float]]) -> Optional[np.ndarray]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    if _faiss is not None:
        try:
            _faiss.normalize_L2(arr)  # type: ignore[attr-defined]
            return arr
        except Exception:
            pass
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return arr


__all__ = ["available", "create_binary_index", "create_dense_index", "normalize_dense_vector"]
