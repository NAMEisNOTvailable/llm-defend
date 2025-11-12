"""
Multiprocessing worker helpers for the compose pipeline.

The worker processes defer to serial composer callables that are registered at
runtime (see :func:`configure_worker_serializers`). This design avoids importing
the top-level ``v3`` module when the package is consumed programmatically.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from compose import state as compose_state
from compose.dedupe_helpers import _make_deduper
from compose.knobs import _resolve_oversample_multiplier
from compose.mismatch import MismatchTracker, set_mismatch_manager
from compose.quota import QuotaManager, set_quota_manager

__all__ = (
    "configure_worker_serializers",
    "_hard_neg_producer_init",
    "_hard_neg_producer_job",
    "_plain_neg_producer_init",
    "_plain_neg_producer_job",
    "_topic_neg_producer_init",
    "_topic_neg_producer_job",
)

SerialFn = Callable[..., Any]

_SERIAL_HARD_NEG: Optional[SerialFn] = None
_SERIAL_PLAIN_NEG: Optional[SerialFn] = None
_SERIAL_TOPIC_NEG: Optional[SerialFn] = None

_HARD_NEG_PRODUCER_CFG: Optional[Dict[str, Any]] = None
_HARD_NEG_PRODUCER_TARGETS: Optional[List[Dict[str, Any]]] = None

_PLAIN_NEG_PRODUCER_CFG: Optional[Dict[str, Any]] = None
_PLAIN_NEG_PRODUCER_TARGETS: Optional[List[Dict[str, Any]]] = None
_PLAIN_NEG_TARGET_SLICE: Optional[Tuple[Dict[str, Any], ...]] = None

_TOPIC_NEG_PRODUCER_CFG: Optional[Dict[str, Any]] = None
_TOPIC_NEG_PRODUCER_TARGETS: Optional[List[Dict[str, Any]]] = None
_TOPIC_NEG_TARGET_SLICE: Optional[Tuple[Dict[str, Any], ...]] = None


def configure_worker_serializers(
    hard_neg_fn: SerialFn,
    plain_neg_fn: SerialFn,
    topic_neg_fn: SerialFn,
) -> None:
    """Register the serial composer callbacks used by the worker jobs."""
    global _SERIAL_HARD_NEG, _SERIAL_PLAIN_NEG, _SERIAL_TOPIC_NEG
    _SERIAL_HARD_NEG = hard_neg_fn
    _SERIAL_PLAIN_NEG = plain_neg_fn
    _SERIAL_TOPIC_NEG = topic_neg_fn


def _require_callback(fn: Optional[SerialFn], name: str) -> SerialFn:
    if fn is None:
        raise RuntimeError(
            f"{name} is not configured; call compose.workers.configure_worker_serializers() first."
        )
    return fn


def _hard_neg_producer_init(
    target_pool: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    audit_q=None,
    quota_bundle: Optional[Dict[str, Any]] = None,
    mismatch_state: Optional[Dict[str, Any]] = None,
) -> None:
    global _HARD_NEG_PRODUCER_CFG, _HARD_NEG_PRODUCER_TARGETS
    compose_state.set_audit_queue(audit_q)
    _HARD_NEG_PRODUCER_CFG = cfg
    _HARD_NEG_PRODUCER_TARGETS = list(target_pool or [])
    if quota_bundle:
        try:
            cfg_q = quota_bundle.get("cfg")
            total_q = quota_bundle.get("N")
            shared_state = quota_bundle.get("state")
            if cfg_q is not None and shared_state is not None and total_q:
                set_quota_manager(QuotaManager(int(total_q), cfg_q, shared_state=shared_state))
        except Exception:
            set_quota_manager(None)
    else:
        set_quota_manager(None)

    if mismatch_state:
        try:
            set_mismatch_manager(MismatchTracker(shared_state=mismatch_state))
        except Exception:
            set_mismatch_manager(MismatchTracker())
    else:
        set_mismatch_manager(MismatchTracker())


def _hard_neg_producer_job(job: Tuple[int, int]) -> Dict[str, Any]:
    serial_fn = _require_callback(_SERIAL_HARD_NEG, "_compose_hard_negs_serial")
    seed, batch = job
    cfg = _HARD_NEG_PRODUCER_CFG or {}
    targets = _HARD_NEG_PRODUCER_TARGETS or []
    if not targets:
        return {"cands": []}

    deduper_params = cfg.get("deduper_params", {})
    local_deduper = _make_deduper(
        sim_bits=int(deduper_params.get("sim_bits", 64)),
        sim_thresh=int(deduper_params.get("sim_thresh", 2)),
        k=int(deduper_params.get("k", 5)),
        n_hash=0,
        bands=0,
        jaccard_thresh=1.01,
        vec_dim=0,
        cosine_thresh=1.0,
        max_vecs=0,
    )
    worker_cap = cfg.get("worker_family_cap")
    if worker_cap is None:
        eff_mult, _ = _resolve_oversample_multiplier(batch, cfg.get("oversample_mult", 1))
        worker_cap = int(math.ceil(int(batch) * max(1.0, eff_mult)))

    cands = serial_fn(
        target_pool=targets,
        k=int(batch),
        seed=int(seed),
        min_CN_share=float(cfg.get("min_CN_share", 0.60)),
        min_CN_share_auto=bool(cfg.get("min_CN_share_auto", False)),
        oversample_mult=cfg.get("oversample_mult", 1),
        deduper=local_deduper,
        family_cap=int(worker_cap),
        disc_rate=cfg.get("disc_rate"),
        art_match_rate=float(cfg.get("art_match_rate", 0.50)),
        struct_evidence_rate=float(cfg.get("struct_evidence_rate", 0.0)),
        mask_format_features_rate=float(cfg.get("mask_format_features_rate", 0.0)),
        neg_effect_guard=bool(cfg.get("neg_effect_guard", False)),
        mask_on=str(cfg.get("mask_on", "both")),
        dedupe_preserve_carrier=bool(cfg.get("dedupe_preserve_carrier", False)),
        dedupe_preserve_codeblocks=bool(cfg.get("dedupe_preserve_codeblocks", False)),
        use_end_nonce=bool(cfg.get("use_end_nonce", False)),
        use_model_eval=bool(cfg.get("use_model_eval", False)),
        strict_neg_diag_gate=bool(cfg.get("strict_neg_diag_gate", True)),
    )
    return {"cands": cands}


def _plain_neg_producer_init(
    target_pool: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    audit_q=None,
    quota_bundle: Optional[Dict[str, Any]] = None,
    mismatch_state: Optional[Dict[str, Any]] = None,
) -> None:
    global _PLAIN_NEG_PRODUCER_CFG, _PLAIN_NEG_PRODUCER_TARGETS, _PLAIN_NEG_TARGET_SLICE
    compose_state.set_audit_queue(audit_q)
    _PLAIN_NEG_PRODUCER_CFG = cfg
    _PLAIN_NEG_PRODUCER_TARGETS = list(target_pool or [])
    random.shuffle(_PLAIN_NEG_PRODUCER_TARGETS)
    _PLAIN_NEG_TARGET_SLICE = tuple(_PLAIN_NEG_PRODUCER_TARGETS)
    if quota_bundle:
        try:
            cfg_q = quota_bundle.get("cfg")
            total_q = quota_bundle.get("N")
            shared_state = quota_bundle.get("state")
            if cfg_q is not None and shared_state is not None and total_q:
                set_quota_manager(QuotaManager(int(total_q), cfg_q, shared_state=shared_state))
        except Exception:
            set_quota_manager(None)
    else:
        set_quota_manager(None)

    if mismatch_state:
        try:
            set_mismatch_manager(MismatchTracker(shared_state=mismatch_state))
        except Exception:
            set_mismatch_manager(MismatchTracker())
    else:
        set_mismatch_manager(MismatchTracker())


def _plain_neg_producer_job(job: Tuple[int, int]) -> Dict[str, Any]:
    serial_fn = _require_callback(_SERIAL_PLAIN_NEG, "_compose_plain_negatives_serial")
    seed, batch = job
    cfg = _PLAIN_NEG_PRODUCER_CFG or {}
    global _PLAIN_NEG_TARGET_SLICE
    targets = _PLAIN_NEG_TARGET_SLICE
    if not targets:
        base = _PLAIN_NEG_PRODUCER_TARGETS or []
        if not base:
            return {"cands": [], "stats": {}}
        targets = _PLAIN_NEG_TARGET_SLICE = tuple(base)

    deduper_params = cfg.get("deduper_params", {})
    local_deduper = _make_deduper(
        sim_bits=int(deduper_params.get("sim_bits", 64)),
        sim_thresh=int(deduper_params.get("sim_thresh", 2)),
        k=int(deduper_params.get("k", 5)),
        n_hash=0,
        bands=0,
        jaccard_thresh=1.01,
        vec_dim=0,
        cosine_thresh=1.0,
        max_vecs=0,
    )
    cands, stats = serial_fn(
        target_pool=targets,
        k=int(batch),
        seed=int(seed),
        deduper=local_deduper,
        disc_rate=cfg.get("disc_rate"),
        carrier_mix_prob=float(cfg.get("carrier_mix_prob", 0.25)),
        stack_prob=float(cfg.get("stack_prob", 0.15)),
        struct_evidence_rate=float(cfg.get("struct_evidence_rate", 0.0)),
        mask_format_features_rate=float(cfg.get("mask_format_features_rate", 0.0)),
        mask_on=str(cfg.get("mask_on", "both")),
        dedupe_preserve_carrier=bool(cfg.get("dedupe_preserve_carrier", False)),
        dedupe_preserve_codeblocks=bool(cfg.get("dedupe_preserve_codeblocks", False)),
        strict_neg_diag_gate=bool(cfg.get("strict_neg_diag_gate", True)),
        cfg=cfg,
    )
    return {"cands": cands, "stats": stats or {}}


def _topic_neg_producer_init(
    target_pool: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    audit_q=None,
    quota_bundle: Optional[Dict[str, Any]] = None,
    mismatch_state: Optional[Dict[str, Any]] = None,
) -> None:
    global _TOPIC_NEG_PRODUCER_CFG, _TOPIC_NEG_PRODUCER_TARGETS, _TOPIC_NEG_TARGET_SLICE
    compose_state.set_audit_queue(audit_q)
    _TOPIC_NEG_PRODUCER_CFG = cfg
    _TOPIC_NEG_PRODUCER_TARGETS = list(target_pool or [])
    random.shuffle(_TOPIC_NEG_PRODUCER_TARGETS)
    _TOPIC_NEG_TARGET_SLICE = tuple(_TOPIC_NEG_PRODUCER_TARGETS)
    if quota_bundle:
        try:
            cfg_q = quota_bundle.get("cfg")
            total_q = quota_bundle.get("N")
            shared_state = quota_bundle.get("state")
            if cfg_q is not None and shared_state is not None and total_q:
                set_quota_manager(QuotaManager(int(total_q), cfg_q, shared_state=shared_state))
        except Exception:
            set_quota_manager(None)
    else:
        set_quota_manager(None)

    if mismatch_state:
        try:
            set_mismatch_manager(MismatchTracker(shared_state=mismatch_state))
        except Exception:
            set_mismatch_manager(MismatchTracker())
    else:
        set_mismatch_manager(MismatchTracker())


def _topic_neg_producer_job(job: Tuple[int, int]) -> Dict[str, Any]:
    serial_fn = _require_callback(_SERIAL_TOPIC_NEG, "_compose_topic_shift_negatives_serial")
    seed, batch = job
    cfg = _TOPIC_NEG_PRODUCER_CFG or {}
    global _TOPIC_NEG_TARGET_SLICE
    targets = _TOPIC_NEG_TARGET_SLICE
    if not targets:
        base = _TOPIC_NEG_PRODUCER_TARGETS or []
        if not base:
            return {"cands": []}
        targets = _TOPIC_NEG_TARGET_SLICE = tuple(base)

    deduper_params = cfg.get("deduper_params", {})
    local_deduper = _make_deduper(
        sim_bits=int(deduper_params.get("sim_bits", 64)),
        sim_thresh=int(deduper_params.get("sim_thresh", 2)),
        k=int(deduper_params.get("k", 5)),
        n_hash=0,
        bands=0,
        jaccard_thresh=1.01,
        vec_dim=0,
        cosine_thresh=1.0,
        max_vecs=0,
    )
    cands = serial_fn(
        target_pool=targets,
        k=int(batch),
        seed=int(seed),
        deduper=local_deduper,
        mask_format_features_rate=float(cfg.get("mask_format_features_rate", 0.0)),
        mask_on=str(cfg.get("mask_on", "both")),
        dedupe_preserve_carrier=bool(cfg.get("dedupe_preserve_carrier", False)),
        dedupe_preserve_codeblocks=bool(cfg.get("dedupe_preserve_codeblocks", False)),
        disc_rate=cfg.get("disc_rate"),
        cfg=cfg,
    )
    return {"cands": cands}

