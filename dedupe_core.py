# -*- coding: utf-8 -*-
"""Compatibility shim for the legacy dedupe_core module."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

try:  # pragma: no cover - prefer native implementation when available
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
except Exception:  # pragma: no cover - sandbox fallback
    import hashlib
    from types import SimpleNamespace

    DEFAULT_DEDUPER_KWARGS: dict = {}
    DEFAULT_EXT_VEC_LIMIT = 0

    class DedupeRecord(SimpleNamespace):
        def __init__(self, text: str, vector: Sequence[float] | None = None):
            super().__init__(text=text, vector=list(vector or []))

    class Deduper:
        """Minimal set-based deduper used when numpy/dedup deps are unavailable."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._seen: Set[str] = set()

        @staticmethod
        def _digest(text: str) -> str:
            data = (text or "").encode("utf-8")
            return hashlib.blake2b(data, digest_size=16).hexdigest()

        def accept(self, text: str) -> bool:
            digest = self._digest(text)
            if digest in self._seen:
                return False
            self._seen.add(digest)
            return True

        def check_digest(self, text: str) -> bool:
            return self._digest(text) not in self._seen

        def prepare(self, text: str) -> DedupeRecord:
            return DedupeRecord(text, vector=None)

    def get_default_deduper_kwargs(**overrides):
        params = dict(DEFAULT_DEDUPER_KWARGS)
        params.update(overrides)
        return params

    def create_default_deduper(**kwargs):
        return Deduper(**kwargs)

    def _h64(token) -> int:
        data = str(token).encode("utf-8")
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

    def _simhash_tokens(tokens: Sequence[str]) -> int:
        return _h64(" ".join(tokens))

    def _simhash_weighted_np(tokens: Sequence[str], *_, **__) -> int:
        return _simhash_tokens(tokens)

    def simhash_weighted_text(text: str) -> int:
        return _h64(text)

    def char_shingles(text: str, k: int) -> Set[str]:
        if k <= 0:
            return {text}
        if len(text) < k:
            return {text}
        return {text[i : i + k] for i in range(len(text) - k + 1)}

    def _char_shingles(text: str, k: int) -> Set[str]:
        return char_shingles(text, k)

    def compute_minhash(shingles: Iterable[str], n_hash: int, seed_base: int = 2025):
        hashes: List[int] = sorted(_h64(s) for s in shingles)
        if n_hash <= 0:
            return tuple()
        if not hashes:
            return tuple(0 for _ in range(n_hash))
        step = max(1, len(hashes) // n_hash)
        return tuple(hashes[i] & 0xFFFFFFFF for i in range(0, len(hashes), step))[:n_hash]

    def _minhash(shingles: Iterable[str], n_hash: int) -> Tuple[int, ...]:
        return compute_minhash(shingles, n_hash)

    class LSHMinhashIndex:  # pragma: no cover - simplified fallback
        def __init__(self, *_, **__):
            self.items: List[Tuple[Tuple[int, ...], Set[str]]] = []

        def reset(self) -> None:
            self.items.clear()

        def add(self, sig, shingles: Set[str]) -> None:
            self.items.append((tuple(sig), set(shingles)))

        def query(self, sig, shingles: Set[str], jaccard_thresh: float | None = None) -> bool:
            threshold = float(jaccard_thresh if jaccard_thresh is not None else 0.9)
            for _sig, existing in self.items:
                inter = len(existing & shingles)
                union = len(existing | shingles) or 1
                if inter / union >= threshold:
                    return True
            return False

    def datasketch_available() -> bool:
        return False

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
