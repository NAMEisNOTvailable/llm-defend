# -*- coding: utf-8 -*-
"""Shared deduplication utilities (SimHash + MinHash-LSH + hashed trigram cosine).

The module exposes a Deduper class that mirrors the behaviour previously defined
in make_malicious_prompts_cn_compose_v2.py but can be reused by other
pipelines (e.g. DSL batch generation) to avoid repeated O(N^2) comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import hashlib
import math
import re

from collections import defaultdict

import numpy as np

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

def get_default_deduper_kwargs(**overrides: Any) -> Dict[str, Any]:
    cfg = dict(DEFAULT_DEDUPER_KWARGS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def create_default_deduper(**overrides: Any) -> "Deduper":
    cfg = get_default_deduper_kwargs(**overrides)
    return Deduper(**cfg)

try:
    from simhash import weighted_fingerprint as _simhash_weighted_fp, hamming_distance as _simhash_hamm_dist  # type: ignore
except Exception:
    _simhash_weighted_fp = None
    _simhash_hamm_dist = None

try:
    from datasketch import MinHash as _DSMinHash, MinHashLSH as _DSMinHashLSH  # type: ignore
except Exception:
    _DSMinHash = None
    _DSMinHashLSH = None

try:
    import faiss  # type: ignore
except Exception:
    try:
        import faiss_cpu as faiss  # type: ignore
    except Exception:
        faiss = None

try:
    from annoy import AnnoyIndex as _AnnoyIndex  # type: ignore
except Exception:
    _AnnoyIndex = None


try:
    from numba import njit  # type: ignore
except Exception:
    njit = None

_UINT32_MASK = np.uint32(0xFFFFFFFF)
_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

if "njit" in globals() and njit is not None:

    @njit(cache=True)
    def _numba_minhash(hashes: np.ndarray, seeds: np.ndarray) -> np.ndarray:  # type: ignore[no-any-unimported]
        out = np.empty(seeds.shape[0], dtype=np.uint32)
        for i in range(seeds.shape[0]):
            seed = seeds[i]
            best = np.uint64(0xFFFFFFFFFFFFFFFF)
            for hv in hashes:
                mix = hv ^ seed
                mix ^= mix >> np.uint64(33)
                mix *= np.uint64(0xff51afd7ed558ccd)
                mix ^= mix >> np.uint64(33)
                mix *= np.uint64(0xc4ceb9fe1a85ec53)
                mix ^= mix >> np.uint64(33)
                if mix < best:
                    best = mix
            out[i] = np.uint32(best & np.uint32(0xFFFFFFFF))
        return out

    @njit(cache=True)
    def _numba_gram_counts(hashes: np.ndarray, vec_dim: int) -> np.ndarray:  # type: ignore[no-any-unimported]
        out = np.zeros(vec_dim, dtype=np.float32)
        for hv in hashes:
            out[int(hv % vec_dim)] += 1.0
        return out
else:
    _numba_minhash = None
    _numba_gram_counts = None

try:
    from xxhash import xxh64 as _xxh64  # type: ignore
except Exception:
    _xxh64 = None


@dataclass
class DedupeRecord:
    """Container for per-text signature cache."""
    raw_text: str
    normalized: str
    simhash: int
    sig_bytes: Optional[np.ndarray]
    shingles: Set[str]
    minhash: object  # datasketch MinHash or tuple fallback
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
        digest = hashlib.md5(data, usedforsecurity=False).digest()
    except TypeError:
        digest = hashlib.md5(data).digest()
    return int.from_bytes(digest, "big")


@lru_cache(maxsize=1 << 20)
def _h64(token: Union[str, bytes]) -> int:
    if isinstance(token, bytes):
        data = token
    else:
        data = str(token).encode("utf-8")
    if _xxh64 is not None:
        try:
            return _xxh64(data).intdigest()
        except Exception:
            pass
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def _simhash_tokens(compact: str) -> List[str]:
    if not compact:
        return []
    chars = [c for c in compact if not c.isspace()]
    bigrams = [compact[i : i + 2] for i in range(len(compact) - 1)]
    tokens = chars + bigrams
    if not tokens:
        tokens = [compact]
    return tokens


def _simhash_weighted_np(tokens: List[str], bits: int = 64) -> int:
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
            return int(_simhash_weighted_fp(pairs))
        except Exception:
            pass
    return _simhash_weighted_np(tokens, bits)


def _simhash_to_bytes(sig: int, bits: int) -> np.ndarray:
    width = max(1, (bits + 7) // 8)
    return np.frombuffer(int(sig).to_bytes(width, "little", signed=False), dtype=np.uint8).reshape(1, width)


def _dense_to_np(vec: List[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype="float32").reshape(1, -1)
    norm = float(np.linalg.norm(arr))
    if faiss is not None:
        try:
            faiss.normalize_L2(arr)
            return arr
        except Exception:
            pass
    if norm > 0.0:
        arr /= norm
    return arr




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
    with np.errstate(over='ignore'):
        for i in range(seeds.shape[0]):
            seed = seeds[i]
            best = np.uint64(0xFFFFFFFFFFFFFFFF)
            for hv in hashes:
                mix = hv ^ seed
                mix ^= mix >> np.uint64(33)
                mix *= np.uint64(0xff51afd7ed558ccd)
                mix ^= mix >> np.uint64(33)
                mix *= np.uint64(0xc4ceb9fe1a85ec53)
                mix ^= mix >> np.uint64(33)
                if mix < best:
                    best = mix
            out[i] = np.uint32(best & _UINT32_MASK)
    return out

def _char_shingles(text: str, k: int) -> Set[str]:
    if k <= 0:
        return set()
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash(shingles: Iterable[str], n_hash: int, seed_base: int = 2025):
    if _DSMinHash is not None:
        mh = _DSMinHash(num_perm=n_hash, hashfunc=_h64)
        for sh in shingles:
            mh.update(sh.encode("utf-8"))
        return mh
    sh_list = list(shingles)
    if not sh_list:
        return tuple(0 for _ in range(n_hash))
    hash_arr = np.fromiter(((np.uint64(_h64(sh)) & _UINT64_MASK) for sh in sh_list), dtype=np.uint64, count=len(sh_list))
    if hash_arr.size == 0:
        return tuple(0 for _ in range(n_hash))
    seed_arr = _build_seed_array(n_hash, seed_base)
    if _numba_minhash is not None:
        sig_arr = _numba_minhash(hash_arr, seed_arr)
    else:
        sig_arr = _minhash_numpy(hash_arr, seed_arr)
    return tuple(int(x) for x in sig_arr)


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


class LSHMinhashIndex:
    def __init__(self, n_hash: int, bands: int, threshold: float):
        self.n_hash = n_hash
        self.bands = max(1, bands)
        self.rows = max(1, n_hash // self.bands)
        self.threshold = float(threshold)
        self._use_datasketch = _DSMinHashLSH is not None
        if self._use_datasketch:
            self._lsh = _DSMinHashLSH(threshold=threshold, num_perm=n_hash)
            self._store: dict[str, Set[str]] = {}
            self._counter = 0
        else:
            self.buckets = [defaultdict(list) for _ in range(self.bands)]
            self.items: List[Tuple[Tuple[int, ...], Set[str]]] = []

    def _band_keys(self, sig: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        return [tuple(sig[i * self.rows : (i + 1) * self.rows]) for i in range(self.bands)]

    def add(self, sig, shingles: Set[str]) -> None:
        if self._use_datasketch:
            key = f"mh-{self._counter}"
            self._counter += 1
            self._store[key] = shingles
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

    def query(self, sig, shingles: Set[str], jaccard_thresh: float) -> bool:
        thr = float(jaccard_thresh)
        if self._use_datasketch:
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
        eff_n_hash = int(n_hash) if _DSMinHash is not None else min(int(n_hash), 32)
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
        self.sim_sigs: List[int] = []
        self._exact_hashes: Set[Tuple[int, int]] = set()
        self._bk_tree = None
        self._faiss_sim = None
        if faiss is not None and self.sim_bits % 8 == 0:
            try:
                self._faiss_sim = faiss.IndexBinaryFlat(self.sim_bits)
            except Exception:
                self._faiss_sim = None
        if self._faiss_sim is None:
            self._bk_tree = _HammingBKTree(_hamm)
        self._faiss_dense = None
        if faiss is not None and self.vec_dim > 0:
            try:
                self._faiss_dense = faiss.IndexFlatIP(self.vec_dim)
            except Exception:
                self._faiss_dense = None
        self._annoy = None
        self._annoy_ids = 0
        self._annoy_built = False
        if _AnnoyIndex is not None and self.vec_dim > 0:
            try:
                self._annoy = _AnnoyIndex(self.vec_dim, "angular")
            except Exception:
                self._annoy = None
        self._vecs_mat: Optional[np.ndarray] = None
        self._vecs_cursor: int = 0
        self._vecs_filled: int = 0
        self._last_cosine_score = 0.0
        self._annoy_pending = 0
        self._annoy_dirty = False
        self._annoy_rebuild_every = max(1, int(annoy_rebuild_every or 1))
        self._annoy_n_trees = max(1, int(annoy_n_trees or 1))
        if self.vec_dim > 0 and self.max_vecs > 0:
            try:
                self._vecs_mat = np.zeros((self.max_vecs, self.vec_dim), dtype=np.float32)
            except Exception:
                self._vecs_mat = None
        if not hasattr(Deduper, "_ext_embed_fn"):
            Deduper._ext_embed_fn = None
            Deduper._ext_vecs = []
            Deduper._ext_cos = 0.90

    @classmethod
    def set_external_embedder(cls, fn, cos_thresh: float = 0.90):
        cls._ext_embed_fn = fn
        cls._ext_cos = float(cos_thresh) if cos_thresh else 0.90

    def check_digest(self, text: str) -> bool:
        """Fast exact-duplicate check on normalized text."""
        try:
            norm = self._normalizer(text)
        except Exception:
            norm = text or ""
        try:
            fingerprint = (len(norm), _md5_u128(norm))
        except Exception:
            return True
        return fingerprint not in self._exact_hashes

    def _embed(self, normalized: str) -> Optional[List[float]]:
        if self.vec_dim <= 0:
            return None
        txt = (normalized or "").lower()
        L = len(txt)
        if L == 0:
            return None
        if L >= 3:
            gram_count = L - 2
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

    def _cosine_near(self, vec: Optional[List[float]]) -> bool:
        self._last_cosine_score = 0.0
        if vec is None or self.vec_dim <= 0:
            return False
        best = 0.0
        if self._faiss_dense is not None and self._faiss_dense.ntotal > 0:
            arr = _dense_to_np(vec)
            k = min(32, self._faiss_dense.ntotal) or 1
            D, _ = self._faiss_dense.search(arr, k)
            if getattr(D, "size", 0):
                try:
                    best = max(best, float(np.max(D[0])))
                except Exception:
                    best = max(best, float(max((float(x) for x in D[0]), default=0.0)))
            for score in D[0]:
                if score >= self.cosine_thresh:
                    self._last_cosine_score = float(score)
                    return True
        if self._annoy is not None and self._annoy_ids > 0:
            min_build = max(32, self._annoy_rebuild_every)
            need_build = False
            if not self._annoy_built and self._annoy_ids >= min_build:
                need_build = True
            elif self._annoy_built and self._annoy_dirty and self._annoy_pending >= self._annoy_rebuild_every and self._annoy_ids >= min_build:
                need_build = True
            if need_build:
                try:
                    self._annoy.build(self._annoy_n_trees)
                    self._annoy_built = True
                    self._annoy_pending = 0
                    self._annoy_dirty = False
                except Exception:
                    self._annoy_built = False
                    self._annoy_pending = 0
                    self._annoy_dirty = False
            if self._annoy_built:
                try:
                    _, dists = self._annoy.get_nns_by_vector(vec, min(32, self._annoy_ids), include_distances=True)
                except Exception:
                    dists = ()
                for dist in dists:
                    cos = 1.0 - (dist * dist) / 2.0
                    if cos > best:
                        best = float(cos)
                    if cos >= self.cosine_thresh:
                        self._last_cosine_score = float(cos)
                        return True
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
        return float(getattr(self, "_last_cosine_score", 0.0))

    def _cosine_ext(self, vec: Optional[List[float]]) -> bool:
        if vec is None:
            return False
        try:
            ext_vecs = getattr(Deduper, "_ext_vecs", [])
            thr = float(getattr(Deduper, "_ext_cos", 0.90))
        except Exception:
            return False
        if not ext_vecs:
            return False
        for existing in ext_vecs:
            score = sum(a * b for a, b in zip(vec, existing))
            if score >= thr:
                return True
        return False

    def check_record(self, record: DedupeRecord) -> Tuple[bool, Optional[str]]:
        fingerprint = (record.exact_len, record.exact_hash)
        if fingerprint in self._exact_hashes:
            return False, "exact"

        if self._faiss_sim is not None and self._faiss_sim.ntotal > 0 and record.sig_bytes is not None:
            k = min(64, self._faiss_sim.ntotal) or 1
            D, _ = self._faiss_sim.search(record.sig_bytes, k)
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
                if isinstance(record.external_vector, list) and record.external_vector:
                    if self._cosine_ext(record.external_vector):
                        return False, "external"
            except Exception:
                pass
        return True, None

    def add_record(self, record: DedupeRecord) -> None:
        self._exact_hashes.add((record.exact_len, record.exact_hash))
        self.sim_sigs.append(record.simhash)
        if self._bk_tree is not None:
            self._bk_tree.add(record.simhash)
        self.index.add(record.minhash, record.shingles)
        if self.vec_dim > 0 and record.vector is not None:
            arr = np.asarray(record.vector, dtype=np.float32).reshape(-1)
            buf = self._vecs_mat
            if buf is not None and self.max_vecs > 0 and buf.shape[0] == self.max_vecs and arr.size == buf.shape[1]:
                idx = self._vecs_cursor
                try:
                    buf[idx, :] = arr
                    self._vecs_cursor = (self._vecs_cursor + 1) % buf.shape[0]
                    self._vecs_filled = min(self._vecs_filled + 1, buf.shape[0])
                except Exception:
                    pass
            elif arr.size == self.vec_dim:
                arr2 = arr.reshape(1, -1)
                if buf is None:
                    self._vecs_mat = arr2
                else:
                    try:
                        self._vecs_mat = np.vstack((buf, arr2))
                    except Exception:
                        return
                    if self.max_vecs > 0 and self._vecs_mat.shape[0] > self.max_vecs:
                        self._vecs_mat = self._vecs_mat[-self.max_vecs:]
                self._vecs_filled = self._vecs_mat.shape[0] if self._vecs_mat is not None else 0
                self._vecs_cursor = self._vecs_filled % max(1, self.max_vecs) if self.max_vecs > 0 else 0
        if self._faiss_sim is not None and record.sig_bytes is not None:
            try:
                self._faiss_sim.add(record.sig_bytes)
            except Exception:
                pass
        if self._faiss_dense is not None and record.vector is not None:
            try:
                self._faiss_dense.add(_dense_to_np(record.vector))
            except Exception:
                pass
        if self._annoy is not None and record.vector is not None:
            try:
                self._annoy.add_item(self._annoy_ids, record.vector)
                self._annoy_ids += 1
                self._annoy_pending += 1
                self._annoy_dirty = True
            except Exception:
                pass
        ext_fn = getattr(Deduper, "_ext_embed_fn", None)
        if ext_fn is not None and isinstance(record.external_vector, list) and record.external_vector:
            try:
                Deduper._ext_vecs.append(record.external_vector)
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


__all__ = ["Deduper", "DedupeRecord", "LSHMinhashIndex", "simhash_weighted_text", "_h64", "_simhash_weighted_np", "_simhash_tokens"]


