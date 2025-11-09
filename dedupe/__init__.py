# -*- coding: utf-8 -*-
"""Public exports for the dedupe package."""
from .core import (
    DEFAULT_DEDUPER_KWARGS,
    DEFAULT_EXT_VEC_LIMIT,
    Deduper,
    DedupeRecord,
    create_default_deduper,
    get_default_deduper_kwargs,
    simhash_weighted_text,
)
from .index import LSHMinhashIndex, _h64, char_shingles, compute_minhash, datasketch_available

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
    "compute_minhash",
    "datasketch_available",
    "char_shingles",
]
