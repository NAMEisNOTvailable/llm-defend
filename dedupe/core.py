# -*- coding: utf-8 -*-
"""Shared deduplication utilities (SimHash + MinHash-LSH + hashed trigram cosine)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, BinaryIO, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import hashlib
import math
import os
import pickle
import re
import threading

from collections import deque

import numpy as np

from .backends import annoy as annoy_backend
from .backends import faiss as faiss_backend
from .index import (
    LSHMinhashIndex,
    _UINT64_MASK,
    _h64,
    _char_shingles,
    _minhash,
    datasketch_available,
)

DEFAULT_DEDUPER_KWARGS: Dict[str, Any] = {
    "sim_bits": 64,
    "sim_thresh": 1,
    "k": 5,
    "n_hash": 64,
    "bands": 16,
    "jaccard_thresh": 0.90,
    "vec_dim": 1024,
    "cosine_thresh": 0.92,
    "max_vecs": 20000,
    "annoy_rebuild_every": 64,
    "annoy_n_trees": 16,
}

DEFAULT_EXT_VEC_LIMIT = DEFAULT_DEDUPER_KWARGS["max_vecs"]


def get_default_deduper_kwargs(**overrides: Any) -> Dict[str, Any]:
    cfg = dict(DEFAULT_DEDUPER_KWARGS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def create_default_deduper(**overrides: Any) -> "Deduper":
    cfg = get_default_deduper_kwargs(**overrides)
    return Deduper(**cfg)


try:
    from simhash import weighted_fingerprint as _simhash_weighted_fp, hamming_distance as _simhash_hamm_dist  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _simhash_weighted_fp = None
    _simhash_hamm_dist = None

try:
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    njit = None

if njit is not None:

    @njit(cache=True)  # type: ignore[misc]
    def _numba_gram_counts(hashes: np.ndarray, vec_dim: int) -> np.ndarray:  # pragma: no cover - numba jit
        out = np.zeros(vec_dim, dtype=np.float32)
        for hv in hashes:
            out[int(hv % vec_dim)] += 1.0
        return out

else:
    _numba_gram_counts = None


@dataclass
class DedupeRecord:
    """Container for per-text signature cache."""

    raw_text: str
    normalized: str
    simhash: int
    sig_bytes: Optional[np.ndarray]
    shingles: Set[str]
    minhash: object
    vector: Optional[List[float]]
    external_vector: Optional[List[float]] = None
    exact_hash: int = 0
    exact_len: int = 0


def _default_normalize(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    import unicodedata

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _md5_u128(text: str) -> int:
    try:
        data = text.encode("utf-8")
    except Exception:
        data = str(text or "").encode("utf-8")
    try:
        digest = hashlib.md5(data, usedforsecurity=False).digest()  # type: ignore[arg-type]
    except TypeError:
        digest = hashlib.md5(data).digest()
    return int.from_bytes(digest, "big")


@lru_cache(maxsize=1 << 20)
def _simhash_tokens(compact: str) -> Tuple[str, ...]:
    if not compact:
        return ()
    chars = [c for c in compact if not c.isspace()]
    bigrams = [compact[i : i + 2] for i in range(len(compact) - 1)]
    tokens = chars + bigrams
    if not tokens:
        tokens = [compact]
    return tuple(tokens)


def _simhash_weighted_np(tokens: Sequence[str], bits: int = 64) -> int:
    if bits <= 0 or not tokens:
        return 0
    hashes = np.fromiter((_h64(tok) for tok in tokens), dtype=np.uint64, count=len(tokens))
    if hashes.size == 0:
        return 0
    bit_bytes = hashes.view(np.uint8).reshape(-1, 8)
    bit_matrix = np.unpackbits(bit_bytes, axis=1, bitorder="little").astype(np.int16)
    weights = (bit_matrix * 2 - 1).sum(axis=0)
    if bits < weights.size:
        weights = weights[:bits]
    elif bits > weights.size:
        weights = np.pad(weights, (0, bits - weights.size), constant_values=0)
    bit_flags = np.where(weights >= 0, 1, 0).astype(np.uint8)
    padded_bits = ((bits + 7) // 8) * 8
    if bit_flags.size < padded_bits:
        bit_flags = np.pad(bit_flags, (0, padded_bits - bit_flags.size), constant_values=0)
    else:
        bit_flags = bit_flags[:padded_bits]
    packed = np.packbits(bit_flags, bitorder="little")
    byte_width = max(1, (bits + 7) // 8)
    return int.from_bytes(packed[:byte_width].tobytes(), "little")


def simhash_weighted_text(text: str, bits: int = 64) -> int:
    compact = _prepare_for_simhash(text)
    if not compact:
        return 0
    return _simhash_weighted_np(_simhash_tokens(compact), bits)


def _prepare_for_simhash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").lower()


def _simhash(text: str, bits: int, normalizer: Callable[[str], str]) -> int:
    norm = normalizer(text)
    compact = _prepare_for_simhash(norm)
    if not compact:
        return 0
    tokens = _simhash_tokens(compact)
    if bits == 64 and _simhash_weighted_fp is not None:
        pairs = [(_h64(tok), 1.0) for tok in tokens]
        try:
            return int(_simhash_weighted_fp(pairs))  # type: ignore[call-arg]
        except Exception:
            pass
    return _simhash_weighted_np(tokens, bits)


def _simhash_to_bytes(sig: int, bits: int) -> np.ndarray:
    width = max(1, (bits + 7) // 8)
    return np.frombuffer(int(sig).to_bytes(width, "little", signed=False), dtype=np.uint8).reshape(1, width)


def _as_float_list(vec: Any) -> Optional[List[float]]:
    if vec is None:
        return None
    if isinstance(vec, list):
        values = vec
    elif isinstance(vec, np.ndarray):
        try:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        values = arr.tolist()
    else:
        if isinstance(vec, (str, bytes)):
            return None
        try:
            values = list(vec)
        except Exception:
            return None
    try:
        converted = [float(x) for x in values]
    except Exception:
        return None
    return converted or None


class _HammingBKTree:
    def __init__(self, dist_fn: Callable[[int, int], int]):
        self.dist_fn = dist_fn
        self.root = None

    class Node:
        __slots__ = ("value", "children")

        def __init__(self, value: int):
            self.value = value
            self.children: dict[int, "_HammingBKTree.Node"] = {}

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = self.Node(value)
            return
        node = self.root
        while True:
            d = self.dist_fn(value, node.value)
            if d == 0:
                return
            if d not in node.children:
                node.children[d] = self.Node(value)
                return
            node = node.children[d]

    def search(self, value: int, radius: int) -> bool:
        if self.root is None:
            return False
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = self.dist_fn(value, node.value)
            if dist <= radius:
                return True
            low = max(0, dist - radius)
            high = dist + radius
            for d, child in node.children.items():
                if low <= d <= high:
                    stack.append(child)
        return False


class Deduper:
    """Combined SimHash + MinHash-LSH + hashed trigram cosine deduper."""

    def __init__(
        self,
        *,
        sim_bits: int = 64,
        sim_thresh: int = 1,
        k: int = 5,
        n_hash: int = 64,
        bands: int = 16,
        jaccard_thresh: float = 0.90,
        vec_dim: int = 1024,
        cosine_thresh: float = 0.92,
        max_vecs: int = 20000,
        normalizer: Optional[Callable[[str], str]] = None,
        annoy_rebuild_every: int = 64,
        annoy_n_trees: int = 16,
    ) -> None:
        self.sim_bits = int(sim_bits)
        self.sim_thresh = int(sim_thresh)
        self.k = max(1, int(k))
        eff_n_hash = int(n_hash) if datasketch_available() else min(int(n_hash), 32)
        eff_n_hash = max(1, eff_n_hash)
        eff_bands = int(bands) or eff_n_hash
        if eff_bands <= 0:
            eff_bands = 1
        if eff_n_hash % eff_bands != 0:
            eff_bands = math.gcd(eff_n_hash, eff_bands) or 1
        self.index = LSHMinhashIndex(n_hash=eff_n_hash, bands=eff_bands, threshold=jaccard_thresh)
        self.n_hash = eff_n_hash
        self.bands = eff_bands
        self.jaccard_thresh = float(jaccard_thresh)
        self.vec_dim = int(vec_dim)
        self.cosine_thresh = float(cosine_thresh)
        self.max_vecs = int(max_vecs)
        self._normalizer = normalizer or _default_normalize
        self._lock = threading.RLock()
        self._vec_lock = threading.RLock()
        self._annoy_lock = threading.RLock()
        self.sim_sigs: List[int] = []
        self._exact_hashes: Set[Tuple[int, int]] = set()
        self._bk_tree = None
        self._faiss_sim = None
        if faiss_backend.available() and self.sim_bits % 8 == 0:
            self._faiss_sim = faiss_backend.create_binary_index(self.sim_bits)
        if self._faiss_sim is None:
            self._bk_tree = _HammingBKTree(_hamm)
        self._faiss_dense = None
        if faiss_backend.available() and self.vec_dim > 0:
            self._faiss_dense = faiss_backend.create_dense_index(self.vec_dim)
        self._annoy = None
        self._annoy_ids = 0
        self._annoy_built = False
        if annoy_backend.available() and self.vec_dim > 0:
            self._annoy = annoy_backend.create_index(self.vec_dim, "angular")
        self._vecs_mat: Optional[np.ndarray] = None
        self._vecs_cursor: int = 0
        self._vecs_filled: int = 0
        self._last_cosine_score = 0.0
        self._annoy_pending = 0
        self._annoy_dirty = False
        self._record_archive: List[Dict[str, Any]] = []
        self._annoy_rebuild_every = max(1, int(annoy_rebuild_every or 1))
        self._annoy_n_trees = max(1, int(annoy_n_trees or 1))
        if self.vec_dim > 0 and self.max_vecs > 0:
            try:
                self._vecs_mat = np.zeros((self.max_vecs, self.vec_dim), dtype=np.float32)
            except Exception:
                self._vecs_mat = None
        if not hasattr(Deduper, "_ext_embed_fn"):
            Deduper._ext_embed_fn = None
            Deduper._ext_cos = 0.90
            Deduper._ext_vec_limit = int(DEFAULT_EXT_VEC_LIMIT)
            Deduper._ext_vecs = deque(maxlen=Deduper._ext_vec_limit)
            Deduper._ext_lock = threading.RLock()

    @classmethod
    def set_external_embedder(cls, fn, cos_thresh: float = 0.90, cache_limit: Optional[int] = None):
        cls._ext_embed_fn = fn
        cls._ext_cos = float(cos_thresh) if cos_thresh else 0.90
        if cache_limit is not None:
            try:
                limit = max(1, int(cache_limit))
            except Exception:
                limit = cls._ext_vec_limit if hasattr(cls, "_ext_vec_limit") else int(DEFAULT_EXT_VEC_LIMIT)
            current = getattr(cls, "_ext_vecs", None)
            existing = list(current) if current else []
            cls._ext_vec_limit = limit
            cls._ext_vecs = deque(existing[-limit:], maxlen=limit)
        elif not isinstance(getattr(cls, "_ext_vecs", None), deque):
            current = getattr(cls, "_ext_vecs", None)
            cls._ext_vecs = deque(list(current) if current else [], maxlen=getattr(cls, "_ext_vec_limit", int(DEFAULT_EXT_VEC_LIMIT)))
        if not hasattr(cls, "_ext_lock"):
            cls._ext_lock = threading.RLock()

    def check_digest(self, text: str) -> bool:
        try:
            norm = self._normalizer(text)
        except Exception:
            norm = text or ""
        try:
            fingerprint = (len(norm), _md5_u128(norm))
        except Exception:
            return True
        with self._lock:
            return fingerprint not in self._exact_hashes

    def _embed(self, normalized: str) -> Optional[List[float]]:
        if self.vec_dim <= 0:
            return None
        txt = (normalized or "").lower()
        length = len(txt)
        if length == 0:
            return None
        if length >= 3:
            gram_count = length - 2
            hash_iter = ((np.uint64(_h64(txt[i : i + 3])) & _UINT64_MASK) for i in range(gram_count))
        else:
            gram_count = 1
            hash_iter = ((np.uint64(_h64(txt)) & _UINT64_MASK) for _ in range(1))
        hash_arr = np.fromiter(hash_iter, dtype=np.uint64, count=gram_count)
        if hash_arr.size == 0:
            return None
        if _numba_gram_counts is not None:
            vec = _numba_gram_counts(hash_arr, self.vec_dim)
        else:
            vec = np.zeros(self.vec_dim, dtype=np.float32)
            idx = (hash_arr % self.vec_dim).astype(np.int64, copy=False)
            np.add.at(vec, idx, 1.0)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec.astype(np.float32).tolist()

    def prepare(self, text: str) -> DedupeRecord:
        normalized = self._normalizer(text or "")
        sig = _simhash(normalized, self.sim_bits, lambda s: s)
        sig_bytes = None
        if self._faiss_sim is not None:
            sig_bytes = _simhash_to_bytes(sig, self.sim_bits)
        shingles = _char_shingles(normalized, self.k)
        minhash = _minhash(shingles, self.n_hash)
        vector = self._embed(normalized)
        exact_len = len(normalized)
        exact_hash = _md5_u128(normalized)
        return DedupeRecord(
            raw_text=text,
            normalized=normalized,
            simhash=sig,
            sig_bytes=sig_bytes,
            shingles=shingles,
            minhash=minhash,
            vector=vector,
            exact_hash=exact_hash,
            exact_len=exact_len,
        )

    def _set_last_cosine_score(self, value: float) -> None:
        with self._vec_lock:
            self._last_cosine_score = float(value)

    def _cosine_near(self, vec: Optional[List[float]]) -> bool:
        if vec is None or self.vec_dim <= 0:
            self._set_last_cosine_score(0.0)
            return False
        self._set_last_cosine_score(0.0)
        best = 0.0
        if self._faiss_dense is not None and getattr(self._faiss_dense, "ntotal", 0) > 0:
            arr = faiss_backend.normalize_dense_vector(vec)
            k = min(32, self._faiss_dense.ntotal) or 1
            try:
                D, _ = self._faiss_dense.search(arr, k)
            except Exception:
                D = ()
            scores = ()
            if getattr(D, "size", 0):
                try:
                    scores = [float(x) for x in D[0]]
                except Exception:
                    scores = [float(x) for x in list(D[0])]
            for score in scores:
                if score > best:
                    best = score
                if score >= self.cosine_thresh:
                    self._set_last_cosine_score(score)
                    return True
        local_dists: Tuple[float, ...] = ()
        if self._annoy is not None:
            with self._annoy_lock:
                ids = self._annoy_ids
                if ids > 0:
                    min_build = max(32, self._annoy_rebuild_every)
                    need_build = False
                    if not self._annoy_built and ids >= min_build:
                        need_build = True
                    elif (
                        self._annoy_built
                        and self._annoy_dirty
                        and self._annoy_pending >= self._annoy_rebuild_every
                        and ids >= min_build
                    ):
                        need_build = True
                    if need_build:
                        try:
                            if self._annoy_built:
                                try:
                                    self._annoy.unbuild()
                                except Exception as exc_unbuild:
                                    self._log_annoy_failure("unbuild", exc_unbuild, ids=ids, built=self._annoy_built)
                            self._annoy.build(self._annoy_n_trees)
                            self._annoy_built = True
                            self._annoy_pending = 0
                            self._annoy_dirty = False
                        except Exception as exc_build:
                            self._log_annoy_failure(
                                "build",
                                exc_build,
                                ids=ids,
                                pending=self._annoy_pending,
                                trees=self._annoy_n_trees,
                                rebuilt=need_build,
                            )
                            self._annoy_built = False
                            self._annoy_pending = 0
                            self._annoy_dirty = False
                    if self._annoy_built:
                        try:
                            _, dists = self._annoy.get_nns_by_vector(
                                vec,
                                min(32, ids),
                                include_distances=True,
                            )
                            local_dists = tuple(float(d) for d in dists)
                        except Exception as exc_query:
                            self._log_annoy_failure(
                                "query", exc_query, ids=ids, k=min(32, ids), built=self._annoy_built
                            )
                            local_dists = ()
        for dist in local_dists:
            cos = 1.0 - (dist * dist) / 2.0
            if cos > best:
                best = float(cos)
            if cos >= self.cosine_thresh:
                self._set_last_cosine_score(cos)
                return True
        with self._vec_lock:
            mat = self._vecs_mat
            count = getattr(self, "_vecs_filled", 0)
            if mat is not None and mat.size > 0 and count > 0:
                query = np.asarray(vec, dtype=np.float32)
                active = mat if count >= mat.shape[0] else mat[:count, :]
                try:
                    scores = active @ query
                except Exception:
                    scores = np.dot(active, query)
                if getattr(scores, "size", 0):
                    try:
                        max_score = float(np.max(scores))
                    except Exception:
                        max_score = float(max((float(x) for x in scores), default=0.0))
                    if max_score > best:
                        best = max_score
                    if np.any(scores >= self.cosine_thresh):
                        self._last_cosine_score = max_score
                        return True
            self._last_cosine_score = float(best)
        return False

    @property
    def last_cosine_score(self) -> float:
        with self._vec_lock:
            return float(getattr(self, "_last_cosine_score", 0.0))

    def _cosine_ext(self, vec: Optional[List[float]]) -> bool:
        if vec is None:
            return False
        try:
            ext_vecs = getattr(Deduper, "_ext_vecs", [])
            thr = float(getattr(Deduper, "_ext_cos", 0.90))
            lock = getattr(Deduper, "_ext_lock", None)
        except Exception:
            return False
        if not ext_vecs:
            return False
        if lock is not None:
            with lock:
                candidates = list(ext_vecs)
                thr_local = thr
        else:
            candidates = list(ext_vecs)
            thr_local = thr
        for existing in candidates:
            try:
                other = _as_float_list(existing)
                if other is None:
                    continue
                score = sum(a * b for a, b in zip(vec, other))
            except Exception:
                continue
            if score >= thr_local:
                return True
        return False

    def check_record(self, record: DedupeRecord) -> Tuple[bool, Optional[str]]:
        with self._lock:
            fingerprint = (record.exact_len, record.exact_hash)
            if fingerprint in self._exact_hashes:
                return False, "exact"

            if self._faiss_sim is not None and getattr(self._faiss_sim, "ntotal", 0) > 0 and record.sig_bytes is not None:
                k = min(64, self._faiss_sim.ntotal) or 1
                try:
                    D, _ = self._faiss_sim.search(record.sig_bytes, k)
                except Exception:
                    D = ()
                if getattr(D, "size", 0):
                    for dist in D[0]:
                        if int(dist) <= self.sim_thresh:
                            return False, "simhash"
            elif self._bk_tree is not None and self._bk_tree.search(record.simhash, self.sim_thresh):
                return False, "simhash"
            elif self._faiss_sim is None and self._bk_tree is None:
                for old in self.sim_sigs:
                    if _hamm(record.simhash, old) <= self.sim_thresh:
                        return False, "simhash"

            if self.index.query(record.minhash, record.shingles, self.jaccard_thresh):
                return False, "jaccard"

            if self._cosine_near(record.vector):
                return False, "cosine"

            ext_fn = getattr(Deduper, "_ext_embed_fn", None)
            if ext_fn is not None:
                try:
                    if record.external_vector is None:
                        record.external_vector = ext_fn(record.normalized)
                    ext_vec = _as_float_list(record.external_vector)
                    if ext_vec is not None:
                        record.external_vector = ext_vec
                        if self._cosine_ext(ext_vec):
                            return False, "external"
                except Exception:
                    pass
            return True, None

    def size(self) -> int:
        with self._lock:
            return len(self._exact_hashes)

    def reset(self) -> None:
        with self._lock:
            self.sim_sigs = []
            self._exact_hashes = set()
            self._record_archive = []
            try:
                self.index.reset()
            except AttributeError:
                pass
            if self._bk_tree is not None:
                self._bk_tree = _HammingBKTree(_hamm)
            if self._faiss_sim is not None:
                try:
                    self._faiss_sim.reset()
                except Exception:
                    self._faiss_sim = faiss_backend.create_binary_index(self.sim_bits) if faiss_backend.available() else None
            if self._faiss_dense is not None:
                try:
                    self._faiss_dense.reset()
                except Exception:
                    self._faiss_dense = faiss_backend.create_dense_index(self.vec_dim) if faiss_backend.available() else None
        with self._annoy_lock:
            if annoy_backend.available() and self.vec_dim > 0:
                self._annoy = annoy_backend.create_index(self.vec_dim, "angular")
            else:
                self._annoy = None
            self._annoy_ids = 0
            self._annoy_built = False
            self._annoy_pending = 0
            self._annoy_dirty = False
        with self._vec_lock:
            if self.vec_dim > 0 and self.max_vecs > 0:
                try:
                    self._vecs_mat = np.zeros((self.max_vecs, self.vec_dim), dtype=np.float32)
                except Exception:
                    self._vecs_mat = None
            else:
                self._vecs_mat = None
            self._vecs_cursor = 0
            self._vecs_filled = 0
            self._last_cosine_score = 0.0

    def _snapshot_config(self) -> Dict[str, Any]:
        """Capture critical runtime parameters for compatibility validation."""
        return {
            "sim_bits": self.sim_bits,
            "sim_thresh": self.sim_thresh,
            "k": self.k,
            "n_hash": self.n_hash,
            "bands": self.bands,
            "jaccard_thresh": self.jaccard_thresh,
            "vec_dim": self.vec_dim,
            "cosine_thresh": self.cosine_thresh,
            "max_vecs": self.max_vecs,
        }

    def _mask_simhash(self, value: int) -> int:
        bits = max(1, int(self.sim_bits))
        mask = (1 << bits) - 1
        return int(value) & mask

    def _require_pickle_opt_in(self, allow_pickle: bool) -> None:
        if not allow_pickle:
            raise ValueError(
                "Loading Deduper snapshots from files requires allow_pickle=True. "
                "Pickle is unsafe for untrusted data; only enable this for trusted snapshots."
            )

    def _validate_loaded_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        if not cfg:
            return
        critical = ("sim_bits", "k", "n_hash", "bands", "vec_dim", "max_vecs")
        mismatches = []
        for key in critical:
            loaded_val = cfg.get(key)
            if loaded_val is None:
                continue
            current_val = getattr(self, key, None)
            if current_val is None:
                continue
            if int(loaded_val) != int(current_val):
                mismatches.append(f"{key} (saved={loaded_val} current={current_val})")
        if mismatches:
            details = ", ".join(mismatches)
            raise ValueError(
                f"Snapshot configuration mismatch: {details}. Instantiate Deduper with matching parameters."
            )

    def dump(self, path: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            exact_hashes = list(self._exact_hashes)
            record_snapshot = [
                {
                    "simhash": entry.get("simhash"),
                    "shingles": list(entry.get("shingles", ())),
                    "vector": list(entry["vector"]) if entry.get("vector") is not None else None,
                }
                for entry in self._record_archive
            ]
        with self._vec_lock:
            vecs_cursor = self._vecs_cursor
            vecs_filled = self._vecs_filled
            max_vecs = self.max_vecs
            vec_dim = self.vec_dim
            if self._vecs_mat is not None and vecs_filled > 0:
                vecs = self._vecs_mat[:vecs_filled].copy()
            else:
                vecs = None
        ext_vectors = None
        ext_vec_limit = int(getattr(Deduper, "_ext_vec_limit", int(DEFAULT_EXT_VEC_LIMIT)))
        ext_store = getattr(Deduper, "_ext_vecs", None)
        ext_lock = getattr(Deduper, "_ext_lock", None)

        def _snapshot_ext(store) -> Optional[List[List[float]]]:
            snapshot: List[List[float]] = []
            for vec in list(store):
                clean = _as_float_list(vec)
                if clean is not None:
                    snapshot.append(list(clean))
            return snapshot or None

        if ext_store:
            if ext_lock is not None:
                with ext_lock:
                    ext_vectors = _snapshot_ext(ext_store)
            else:
                ext_vectors = _snapshot_ext(ext_store)
        state: Dict[str, Any] = {
            "exact_hashes": exact_hashes,
            "vecs_cursor": vecs_cursor,
            "vecs_filled": vecs_filled,
            "max_vecs": max_vecs,
            "vec_dim": vec_dim,
            "vecs": vecs,
            "records": record_snapshot,
            "config": self._snapshot_config(),
            "state_version": 3,
            "ext_vectors": ext_vectors,
            "ext_vec_limit": ext_vec_limit,
        }
        if path:
            with open(path, "wb") as fh:
                pickle.dump(state, fh)
        return state
 
    def load(
        self,
        source: Union[str, bytes, os.PathLike, Dict[str, Any], BinaryIO],
        *,
        allow_pickle: bool = False,
    ) -> None:
        """
        Load a previously dumped deduper snapshot.

        Parameters
        ----------
        source:
            Either the dict returned by `dump`, a path to a snapshot file, or a
            binary file-like object positioned at the start of the snapshot.
        allow_pickle:
            Set to True only when the snapshot originates from a trusted source.
            File-based snapshots rely on `pickle`, which can execute arbitrary
            code during load. Passing False (default) prevents accidental use.
        """
        if isinstance(source, (str, bytes, os.PathLike)):
            self._require_pickle_opt_in(allow_pickle)
            with open(source, "rb") as fh:
                state = pickle.load(fh)
        elif hasattr(source, "read"):
            self._require_pickle_opt_in(allow_pickle)
            state = pickle.load(source)  # type: ignore[arg-type]
        else:
            state = dict(source)
        self._validate_loaded_config(state.get("config"))
        self.reset()
        exact = state.get("exact_hashes") or []
        with self._lock:
            self._exact_hashes = set(
                tuple(item) if isinstance(item, (list, tuple)) else tuple(item) for item in exact
            )
        vecs = state.get("vecs")
        cursor = int(state.get("vecs_cursor", 0))
        should_rebuild_vectors = False
        with self._vec_lock:
            if vecs is not None:
                arr = np.asarray(vecs, dtype=np.float32)
                if arr.ndim == 2 and self.vec_dim > 0 and arr.shape[1] == self.vec_dim:
                    if self.max_vecs > 0:
                        cap = min(arr.shape[0], self.max_vecs)
                        try:
                            self._vecs_mat = np.zeros((self.max_vecs, self.vec_dim), dtype=np.float32)
                            self._vecs_mat[:cap, :] = arr[:cap, :]
                        except Exception:
                            self._vecs_mat = arr[:cap, :].copy()
                            self.max_vecs = self._vecs_mat.shape[0]
                        self._vecs_filled = cap
                        if self.max_vecs > 0:
                            capacity = max(1, self.max_vecs)
                            next_cursor = int(cursor) % capacity
                            if cap < capacity:
                                next_cursor = min(next_cursor, cap)
                            self._vecs_cursor = next_cursor
                        else:
                            self._vecs_cursor = 0
                    else:
                        self._vecs_mat = arr.copy()
                        self._vecs_filled = arr.shape[0]
                        self._vecs_cursor = min(cursor, self._vecs_filled)
                else:
                    should_rebuild_vectors = self.vec_dim > 0
                    self._vecs_mat = None
                    self._vecs_filled = 0
                    self._vecs_cursor = 0
            else:
                should_rebuild_vectors = self.vec_dim > 0
                if self.vec_dim > 0 and self.max_vecs > 0:
                    try:
                        self._vecs_mat = np.zeros((self.max_vecs, self.vec_dim), dtype=np.float32)
                    except Exception:
                        self._vecs_mat = None
                else:
                    self._vecs_mat = None
                self._vecs_filled = 0
                self._vecs_cursor = 0
            self._last_cosine_score = 0.0
        records = state.get("records") or []
        if records:
            self._restore_records_from_state(records, rebuild_vectors=should_rebuild_vectors)
        else:
            self._record_archive = []
        self._restore_external_cache(state.get("ext_vectors"), state.get("ext_vec_limit"))

    def _log_annoy_failure(self, stage: str, exc: Exception, **extra: object) -> None:
        try:
            context = " ".join(f"{k}={v}" for k, v in extra.items() if v is not None)
        except Exception:
            context = ""
        msg = f"[annoy][warn] stage={stage} err={exc}"
        if context:
            msg = f"{msg} {context}"
        print(msg, flush=True)

    def _append_vector_buffer(self, vec_arr: np.ndarray) -> None:
        if self.vec_dim <= 0 or self.max_vecs <= 0:
            return
        if vec_arr.size != self.vec_dim:
            return
        vec_arr = np.asarray(vec_arr, dtype=np.float32).reshape(-1)
        with self._vec_lock:
            buf = self._vecs_mat
            if (
                buf is not None
                and self.max_vecs > 0
                and buf.shape[0] == self.max_vecs
                and vec_arr.size == buf.shape[1]
            ):
                idx = self._vecs_cursor
                try:
                    buf[idx, :] = vec_arr
                except Exception:
                    pass
                else:
                    self._vecs_cursor = (self._vecs_cursor + 1) % buf.shape[0]
                    self._vecs_filled = min(self._vecs_filled + 1, buf.shape[0])
                    return
            arr2 = vec_arr.reshape(1, -1)
            if buf is None:
                self._vecs_mat = arr2.copy()
            else:
                try:
                    self._vecs_mat = np.vstack((buf, arr2))
                except Exception:
                    return
                if self.max_vecs > 0 and self._vecs_mat.shape[0] > self.max_vecs:
                    self._vecs_mat = self._vecs_mat[-self.max_vecs :]
            if self._vecs_mat is not None:
                self._vecs_filled = self._vecs_mat.shape[0]
                self._vecs_cursor = (
                    self._vecs_filled % max(1, self.max_vecs) if self.max_vecs > 0 else 0
                )

    def _queue_annoy_vector(self, vector: List[float]) -> None:
        with self._annoy_lock:
            if self._annoy is None:
                return
            try:
                self._annoy.add_item(self._annoy_ids, vector)
                self._annoy_ids += 1
                self._annoy_pending += 1
                self._annoy_dirty = True
            except Exception as exc_add:
                self._log_annoy_failure("add_item", exc_add, idx=self._annoy_ids)

    def _snapshot_record(self, record: DedupeRecord) -> None:
        vector_snapshot: Optional[Tuple[float, ...]] = None
        if record.vector is not None:
            try:
                vector_snapshot = tuple(float(x) for x in record.vector)
            except Exception:
                vector_snapshot = tuple(float(x) for x in list(record.vector))
        self._record_archive.append(
            {
                "simhash": self._mask_simhash(int(record.simhash)),
                "shingles": tuple(sorted(record.shingles)),
                "vector": vector_snapshot,
            }
        )

    def _restore_records_from_state(
        self,
        entries: Iterable[Dict[str, Any]],
        *,
        rebuild_vectors: bool = False,
    ) -> None:
        restored: List[Dict[str, Any]] = []
        with self._lock:
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    simhash = self._mask_simhash(int(entry.get("simhash", 0)))
                except Exception:
                    continue
                shingles_raw = entry.get("shingles") or ()
                if isinstance(shingles_raw, (list, tuple, set)):
                    shingles_iter = shingles_raw
                else:
                    shingles_iter = (shingles_raw,)
                shingles_set: Set[str] = set()
                for sh in shingles_iter:
                    if sh is None:
                        continue
                    shingles_set.add(str(sh))
                vector_data = entry.get("vector")
                vector_list: Optional[List[float]] = None
                if vector_data is not None:
                    try:
                        vector_list = [float(x) for x in vector_data]
                    except Exception:
                        vector_list = [float(x) for x in list(vector_data)]
                restored.append(
                    {
                        "simhash": simhash,
                        "shingles": tuple(sorted(shingles_set)),
                        "vector": tuple(vector_list) if vector_list is not None else None,
                    }
                )
                rec = DedupeRecord(
                    raw_text="",
                    normalized="",
                    simhash=simhash,
                    sig_bytes=_simhash_to_bytes(simhash, self.sim_bits) if self._faiss_sim is not None else None,
                    shingles=shingles_set,
                    minhash=_minhash(shingles_set, self.n_hash),
                    vector=vector_list,
                    exact_hash=0,
                    exact_len=0,
                )
                self._ingest_record_no_lock(
                    rec,
                    snapshot=False,
                    update_vector_buffer=rebuild_vectors,
                )
        self._record_archive = restored

    def _restore_external_cache(
        self,
        vectors: Optional[Iterable[Any]],
        limit: Optional[int],
    ) -> None:
        try:
            limit_val = max(1, int(limit))
        except Exception:
            limit_val = int(getattr(Deduper, "_ext_vec_limit", int(DEFAULT_EXT_VEC_LIMIT)))
        lock = getattr(Deduper, "_ext_lock", None)
        if lock is None:
            Deduper._ext_lock = threading.RLock()
            lock = Deduper._ext_lock
        restored: deque = deque(maxlen=limit_val)
        if vectors:
            try:
                iterable = list(vectors)
            except Exception:
                iterable = []
            for vec in iterable[-limit_val:]:
                clean = _as_float_list(vec)
                if clean is not None:
                    restored.append(clean)
        with lock:
            Deduper._ext_vec_limit = limit_val
            Deduper._ext_vecs = restored

    def _ingest_record_no_lock(
        self,
        record: DedupeRecord,
        *,
        snapshot: bool = True,
        update_vector_buffer: bool = True,
    ) -> None:
        self.sim_sigs.append(record.simhash)
        if self._bk_tree is not None:
            self._bk_tree.add(record.simhash)
        self.index.add(record.minhash, record.shingles)
        if update_vector_buffer and self.vec_dim > 0 and record.vector is not None:
            arr = np.asarray(record.vector, dtype=np.float32).reshape(-1)
            self._append_vector_buffer(arr)
        if self._faiss_sim is not None and record.sig_bytes is not None:
            try:
                self._faiss_sim.add(record.sig_bytes)
            except Exception:
                pass
        if self._faiss_dense is not None and record.vector is not None:
            try:
                dense_vec = faiss_backend.normalize_dense_vector(record.vector)
                if dense_vec is not None:
                    self._faiss_dense.add(dense_vec)
            except Exception:
                pass
        if record.vector is not None:
            self._queue_annoy_vector(record.vector)
        if snapshot:
            self._snapshot_record(record)

    def add_record(self, record: DedupeRecord) -> None:
        with self._lock:
            self._exact_hashes.add((record.exact_len, record.exact_hash))
            self._ingest_record_no_lock(record)
            ext_fn = getattr(Deduper, "_ext_embed_fn", None)
            if ext_fn is not None:
                try:
                    ext_vec = _as_float_list(record.external_vector)
                    if ext_vec is not None:
                        record.external_vector = ext_vec
                        limit = max(1, int(getattr(Deduper, "_ext_vec_limit", int(DEFAULT_EXT_VEC_LIMIT))))
                        lock = getattr(Deduper, "_ext_lock", None)
                        target_vec = list(ext_vec)
                        if lock is not None:
                            with lock:
                                store = getattr(Deduper, "_ext_vecs", None)
                                raw = list(store) if store else []
                                if isinstance(store, deque) and store.maxlen == limit:
                                    target = store
                                else:
                                    target = deque(raw[-limit:], maxlen=limit)
                                    Deduper._ext_vecs = target
                                target.append(target_vec)
                        else:
                            store = getattr(Deduper, "_ext_vecs", None)
                            raw = list(store) if store else []
                            if isinstance(store, deque) and store.maxlen == limit:
                                target = store
                            else:
                                target = deque(raw[-limit:], maxlen=limit)
                                Deduper._ext_vecs = target
                            target.append(target_vec)
                except Exception:
                    pass

    def probe(self, text: str) -> Tuple[bool, Optional[str], DedupeRecord]:
        record = self.prepare(text)
        ok, reason = self.check_record(record)
        return ok, reason, record

    def accept(self, text: str) -> bool:
        ok, _, record = self.probe(text)
        if ok:
            self.add_record(record)
        return ok


def _hamm(a: int, b: int) -> int:
    if _simhash_hamm_dist is not None:
        try:
            return int(_simhash_hamm_dist(int(a), int(b)))
        except Exception:
            pass
    x = int(a) ^ int(b)
    try:
        return int(x.bit_count())
    except AttributeError:
        return bin(x).count("1")


__all__ = [
    "DEFAULT_DEDUPER_KWARGS",
    "DEFAULT_EXT_VEC_LIMIT",
    "Deduper",
    "DedupeRecord",
    "create_default_deduper",
    "get_default_deduper_kwargs",
    "simhash_weighted_text",
    "_h64",
]
