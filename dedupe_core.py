# -*- coding: utf-8 -*-
"""Compatibility shim for the legacy dedupe_core module."""
from __future__ import annotations

try:  # pragma: no cover - executed at import time
    from dedupe.core import (  # type: ignore
        DEFAULT_DEDUPER_KWARGS,
        DEFAULT_EXT_VEC_LIMIT,
        Deduper,
        DedupeRecord,
        _simhash_tokens,
        _simhash_weighted_np,
        create_default_deduper,
        get_default_deduper_kwargs,
        simhash_weighted_text,
    )
    from dedupe.index import (  # type: ignore
        LSHMinhashIndex,
        _char_shingles,
        _h64,
        _minhash,
        char_shingles,
        compute_minhash,
        datasketch_available,
    )
except Exception as exc:  # pragma: no cover - propagate failures with context
    raise ImportError(
        "dedupe_core requires the dedupe package to be importable; no fallback is provided"
    ) from exc


__all__ = [
    "DEFAULT_DEDUPER_KWARGS",
    "DEFAULT_EXT_VEC_LIMIT",
    "Deduper",
    "DedupeRecord",
    "LSHMinhashIndex",
    "create_default_deduper",
    "get_default_deduper_kwargs",
    "simhash_weighted_text",
    "_h64",
    "_minhash",
    "_char_shingles",
    "_simhash_tokens",
    "_simhash_weighted_np",
    "char_shingles",
    "compute_minhash",
    "datasketch_available",
]
