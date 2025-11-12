# -*- coding: utf-8 -*-
"""MinHash-LSH utilities and shared hashing helpers for the dedupe package."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Iterable, List, Optional, Set, Tuple, Union

import hashlib
import numpy as np

try:
    from datasketch import MinHash as _DSMinHash, MinHashLSH as _DSMinHashLSH  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _DSMinHash = None
    _DSMinHashLSH = None


def datasketch_available() -> bool:
    return _DSMinHash is not None

try:
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    njit = None

try:
    from xxhash import xxh64 as _xxh64  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _xxh64 = None

_UINT32_MASK = np.uint32(0xFFFFFFFF)
_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

if njit is not None:

    @njit(cache=True)  # type: ignore[misc]
    def _numba_minhash(hashes: np.ndarray, seeds: np.ndarray) -> np.ndarray:  # pragma: no cover - numba jit
        out = np.empty(seeds.shape[0], dtype=np.uint32)
        for i in range(seeds.shape[0]):
            seed = seeds[i]
            best = np.uint64(0xFFFFFFFFFFFFFFFF)
            for hv in hashes:
                mix = hv ^ seed
                mix ^= mix >> np.uint64(33)
                mix *= np.uint64(0xFF51AFD7ED558CCD)
                mix ^= mix >> np.uint64(33)
                mix *= np.uint64(0xC4CEB9FE1A85EC53)
                mix ^= mix >> np.uint64(33)
                if mix < best:
                    best = mix
            out[i] = np.uint32(best & np.uint32(0xFFFFFFFF))
        return out

else:
    _numba_minhash = None


def _hash_bytes(data: Union[str, bytes]) -> bytes:
    if isinstance(data, bytes):
        return data
    return str(data).encode("utf-8")


@lru_cache(maxsize=1 << 20)
def _h64(token: Union[str, bytes]) -> int:
    data = _hash_bytes(token)
    if _xxh64 is not None:
        try:
            return _xxh64(data).intdigest()  # type: ignore[operator]
        except Exception:
            pass
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def char_shingles(text: str, k: int) -> Set[str]:
    if k <= 0:
        return set()
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _build_seed_array(n_hash: int, seed_base: int) -> np.ndarray:
    seeds = np.empty(max(0, int(n_hash)), dtype=np.uint64)
    step = np.uint64(1315423911)
    base = np.uint64(seed_base)
    for i in range(seeds.shape[0]):
        seeds[i] = base + np.uint64(i) * step
    return seeds


def _minhash_numpy(hashes: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    out = np.empty(seeds.shape[0], dtype=np.uint32)
    if hashes.size == 0:
        if out.size:
            out.fill(0)
        return out
    with np.errstate(over="ignore"):
        for i in range(seeds.shape[0]):
            seed = seeds[i]
            best = np.uint64(0xFFFFFFFFFFFFFFFF)
            for hv in hashes:
                mix = hv ^ seed
                mix ^= mix >> np.uint64(33)
                mix *= np.uint64(0xFF51AFD7ED558CCD)
                mix ^= mix >> np.uint64(33)
                mix *= np.uint64(0xC4CEB9FE1A85EC53)
                mix ^= mix >> np.uint64(33)
                if mix < best:
                    best = mix
            out[i] = np.uint32(best & _UINT32_MASK)
    return out


def compute_minhash(shingles: Iterable[str], n_hash: int, seed_base: int = 2025):
    if _DSMinHash is not None:
        mh = _DSMinHash(num_perm=n_hash, hashfunc=_h64)
        for sh in shingles:
            mh.update(sh.encode("utf-8"))
        return mh
    sh_list = list(shingles)
    if not sh_list:
        return tuple(0 for _ in range(n_hash))
    hash_arr = np.fromiter(
        ((np.uint64(_h64(sh)) & _UINT64_MASK) for sh in sh_list),
        dtype=np.uint64,
        count=len(sh_list),
    )
    if hash_arr.size == 0:
        return tuple(0 for _ in range(n_hash))
    seed_arr = _build_seed_array(n_hash, seed_base)
    if _numba_minhash is not None:
        sig_arr = _numba_minhash(hash_arr, seed_arr)
    else:
        sig_arr = _minhash_numpy(hash_arr, seed_arr)
    return tuple(int(x) for x in sig_arr)


class LSHMinhashIndex:
    def __init__(self, n_hash: int, bands: int, threshold: float):
        self.n_hash = n_hash
        self.bands = max(1, bands)
        self.rows = max(1, n_hash // self.bands)
        self.threshold = float(threshold)
        self._use_datasketch = _DSMinHashLSH is not None
        if self._use_datasketch:
            self._lsh = _DSMinHashLSH(threshold=threshold, num_perm=n_hash)  # type: ignore[call-arg]
            self._store: dict[str, Set[str]] = {}
            self._counter = 0
        else:
            self.buckets = [defaultdict(list) for _ in range(self.bands)]
            self.items: List[Tuple[Tuple[int, ...], Set[str]]] = []

    def reset(self) -> None:
        if self._use_datasketch:
            self._lsh = (
                _DSMinHashLSH(threshold=self.threshold, num_perm=self.n_hash)  # type: ignore[call-arg]
                if _DSMinHashLSH is not None
                else None
            )
            self._store = {}
            self._counter = 0
        else:
            self.buckets = [defaultdict(list) for _ in range(self.bands)]
            self.items = []

    def _band_keys(self, sig: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        return [tuple(sig[i * self.rows : (i + 1) * self.rows]) for i in range(self.bands)]

    def add(self, sig, shingles: Set[str]) -> None:
        if self._use_datasketch:
            key = f"mh-{self._counter}"
            self._counter += 1
            self._store[key] = shingles
            if self._lsh is not None:
                self._lsh.insert(key, sig)
            return
        sig = tuple(sig)
        idx = len(self.items)
        self.items.append((sig, shingles))
        for band_idx, key in enumerate(self._band_keys(sig)):
            self.buckets[band_idx][key].append(idx)

    def _candidate_indices(self, sig: Tuple[int, ...]) -> Set[int]:
        cand: Set[int] = set()
        for band_idx, key in enumerate(self._band_keys(sig)):
            cand.update(self.buckets[band_idx].get(key, []))
        return cand

    def query(self, sig, shingles: Set[str], jaccard_thresh: Optional[float] = None) -> bool:
        thr = float(jaccard_thresh if jaccard_thresh is not None else self.threshold)
        if self._use_datasketch:
            if self._lsh is None:
                return False
            keys = self._lsh.query(sig)
            for key in keys:
                existing = self._store.get(key, set())
                inter = len(shingles & existing)
                union = len(shingles | existing) + 1e-9
                if inter / union >= thr:
                    return True
            return False
        sig = tuple(sig)
        cand_idx = self._candidate_indices(sig)
        for idx in cand_idx:
            osig, oshingles = self.items[idx]
            inter = len(shingles & oshingles)
            union = len(shingles | oshingles) + 1e-9
            if inter / union >= thr:
                return True
        return False


_minhash = compute_minhash
_char_shingles = char_shingles

__all__ = [
    "LSHMinhashIndex",
    "compute_minhash",
    "_minhash",
    "char_shingles",
    "_char_shingles",
    "_h64",
    "_UINT32_MASK",
    "_UINT64_MASK",
    "datasketch_available",
]
