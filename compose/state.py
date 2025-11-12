"""
Runtime state container for the CN compose pipeline.

This module centralises mutable globals that were previously scattered
through the monolithic script so that other submodules can depend on a
stable interface without creating circular imports.
"""

from __future__ import annotations

import json
import multiprocessing as _mp
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from compose.weight_defaults import (
    CARRIER_FALLBACK_DECAY as _CARRIER_FALLBACK_DECAY,
    CARRIER_FALLBACK_PROFILE as _CARRIER_FALLBACK_PROFILE,
    carrier_weight_defaults,
    delivery_weight_defaults,
)

# --- Active configuration -------------------------------------------------

_ACTIVE_CFG: Any = None


def set_config(cfg: Any) -> None:
    """Set the process-wide configuration object."""
    global _ACTIVE_CFG
    _ACTIVE_CFG = cfg


def get_config() -> Any:
    """Return the currently active configuration (may be ``None``)."""
    return _ACTIVE_CFG


def cfg_get(name: str, default: Any = None, cfg_obj: Any = None) -> Any:
    """Get an attribute from the active config with a fallback."""
    cfg = cfg_obj if cfg_obj is not None else _ACTIVE_CFG
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


# --- Audit bookkeeping ----------------------------------------------------

_AUDIT_REASON_BASE: Dict[str, int] = {
    "semantic_gate": 0,
    "dedup": 0,
    "CN_ratio": 0,
    "family_cap": 0,
    "contract_violation": 0,
    "CN_ratio_low": 0,
    "gray_neg_keep": 0,
}

AUDIT_REJECTS: list[dict] = []
AUDIT_REASON_COUNT: Dict[str, int] = dict(_AUDIT_REASON_BASE)
MP_AUDIT_QUEUE: Optional[_mp.Queue] = None


def reset_audit_bookkeeping() -> None:
    """Reset audit counters and buffered rejection records."""
    AUDIT_REJECTS.clear()
    AUDIT_REASON_COUNT.clear()
    AUDIT_REASON_COUNT.update(_AUDIT_REASON_BASE)


def append_audit_record(record: dict) -> None:
    """Append a rejection/audit record and update counts."""
    AUDIT_REJECTS.append(record)
    reason = record.get("reason")
    if reason:
        AUDIT_REASON_COUNT[reason] = AUDIT_REASON_COUNT.get(reason, 0) + 1


def set_audit_queue(queue: Optional[_mp.Queue]) -> None:
    global MP_AUDIT_QUEUE
    MP_AUDIT_QUEUE = queue


def get_audit_queue() -> Optional[_mp.Queue]:
    return MP_AUDIT_QUEUE


# --- Weight tables & registries -------------------------------------------

GOAL_WEIGHTS: Dict[str, float] = {
    "policy_bypass": 0.28,
    "exfil_mem": 0.24,
    "tool_abuse": 0.20,
    "priv_escalate": 0.12,
    "rag_poison": 0.10,
    "sandbox_escape": 0.06,
}

# 载体权重（视觉/格式包裹，兼顾自然/技术域）// Carrier weights (visual/format wrappers)
CARRIER_WEIGHTS: Dict[str, float] = carrier_weight_defaults()
DELIVERY_WEIGHTS: Dict[str, float] = delivery_weight_defaults()

_CARRIER_FALLBACK_ADJUSTED = False


def _apply_carrier_fallback_profile(detail: str = "") -> None:
    global _CARRIER_FALLBACK_ADJUSTED
    if _CARRIER_FALLBACK_ADJUSTED:
        return
    weights = dict(CARRIER_WEIGHTS)
    for name in list(weights.keys()):
        if name in _CARRIER_FALLBACK_PROFILE:
            continue
        weights[name] = max(float(weights.get(name, 0.0)) * _CARRIER_FALLBACK_DECAY, 0.001)
    for name, value in _CARRIER_FALLBACK_PROFILE.items():
        weights[name] = float(value)
    total = sum(weights.values()) or 1.0
    for name in list(weights.keys()):
        weights[name] = round(weights[name] / total, 6)
    CARRIER_WEIGHTS.clear()
    CARRIER_WEIGHTS.update(weights)
    _CARRIER_FALLBACK_ADJUSTED = True
    profile_summary = {key: CARRIER_WEIGHTS[key] for key in sorted(_CARRIER_FALLBACK_PROFILE)}
    tag = detail or "fallback"
    try:  # pragma: no cover - optional diagnostics linkage
        from compose.capabilities import _register_capability as _cap_register  # type: ignore
    except Exception:
        pass
    else:
        detail_str = ", ".join(f"{k}={profile_summary[k]:.3f}" for k in profile_summary)
        _cap_register("carriers.weights_adjusted", True, f"{tag}:{detail_str}")
    print(f"[carrier][fallback] compiled carriers unavailable ({tag}); adjusted weights {profile_summary}")


try:  # pragma: no cover - import guards
    from compose.carriers import (
        HAS_COMPILED_CARRIERS as _HAS_COMPILED_CARRIERS,
        CARRIER_IMPL_DETAIL as _CARRIER_IMPL_DETAIL,
        CARRIER_TEMPLATES as _CARRIER_TEMPLATES,
    )
except Exception:  # pragma: no cover - defensive
    _HAS_COMPILED_CARRIERS = False
    _CARRIER_IMPL_DETAIL = "fallback:import_error"
    _CARRIER_TEMPLATES = {}
else:
    _HAS_COMPILED_CARRIERS = bool(_HAS_COMPILED_CARRIERS)

_FALLBACK_TEMPLATE_SENTINEL = {"none", "html_comment", "yaml_front_matter", "md_ref_link"}
try:
    _CARRIER_TEMPLATE_KEYS = set(_CARRIER_TEMPLATES.keys())
except Exception:  # pragma: no cover - guard
    _CARRIER_TEMPLATE_KEYS = set()

_minimal_template_set = bool(_CARRIER_TEMPLATE_KEYS) and _CARRIER_TEMPLATE_KEYS.issubset(_FALLBACK_TEMPLATE_SENTINEL)

if not _HAS_COMPILED_CARRIERS or _minimal_template_set:
    detail = _CARRIER_IMPL_DETAIL
    if _minimal_template_set and detail:
        detail = f"{detail};templates={sorted(_CARRIER_TEMPLATE_KEYS)}"
    _apply_carrier_fallback_profile(detail)


def apply_weight_overrides(
    carrier_updates: Optional[Mapping[str, float]] = None,
    delivery_updates: Optional[Mapping[str, float]] = None,
) -> None:
    """Apply persisted or CLI-provided overrides to weight tables."""
    if carrier_updates:
        _update_weights(CARRIER_WEIGHTS, carrier_updates)
    if delivery_updates:
        _update_weights(DELIVERY_WEIGHTS, delivery_updates)


def _update_weights(target: MutableMapping[str, float], updates: Mapping[str, float]) -> None:
    for key, value in updates.items():
        try:
            target[key] = float(value)
        except Exception:
            continue


def load_weights(path: str | Path, *, allow_missing: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Load carrier/delivery weight overrides from ``path`` and apply them.

    Returns (True, None) on success, (False, reason) on failure. When the file
    is missing and ``allow_missing`` is True, the reason is ``None``.
    """
    p = Path(path)
    if not p.exists():
        return (False, None) if allow_missing else (False, f"{p} does not exist")
    try:
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        return False, str(exc)

    if not isinstance(payload, Mapping):
        return False, "weights payload must be a JSON object"

    carriers_section: Mapping[str, Any] = {}
    deliveries_section: Mapping[str, Any] = {}
    maybe_carriers = payload.get("carriers")
    maybe_deliveries = payload.get("deliveries")
    if isinstance(maybe_carriers, Mapping) or isinstance(maybe_deliveries, Mapping):
        carriers_section = maybe_carriers or {}
        deliveries_section = maybe_deliveries or {}
    elif all(isinstance(v, (int, float)) for v in payload.values()):
        carriers_section = payload
    else:
        # Fallback for legacy schema where weights were nested arbitrarily.
        carriers_section = next(
            (v for k, v in payload.items() if isinstance(v, Mapping) and "plain_paragraph" in v),
            {},
        )
        deliveries_section = next(
            (v for k, v in payload.items() if isinstance(v, Mapping) and "multi_turn" in v),
            {},
        )

    carrier_updates: Dict[str, float] = {}
    delivery_updates: Dict[str, float] = {}

    for key, value in (carriers_section or {}).items():
        try:
            weight = float(value)
        except Exception:
            continue
        if key == "chat_dialog":
            delivery_updates["multi_turn"] = weight
        elif key in CARRIER_WEIGHTS:
            carrier_updates[key] = weight

    for key, value in (deliveries_section or {}).items():
        try:
            weight = float(value)
        except Exception:
            continue
        if key == "chat_dialog":
            delivery_updates["multi_turn"] = weight
        elif key in DELIVERY_WEIGHTS:
            delivery_updates[key] = weight

    if carrier_updates or delivery_updates:
        apply_weight_overrides(carrier_updates or None, delivery_updates or None)

    return True, None


def persist_weights(
    path: str | Path,
    *,
    stats_path: str | Path | None = None,
    extra_stats: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Persist carrier and delivery weights to ``path`` and (optionally) merge them
    into ``stats_path`` alongside ``extra_stats``.
    """
    p = Path(path)
    try:
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "carriers": CARRIER_WEIGHTS,
                    "deliveries": DELIVERY_WEIGHTS,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as exc:
        return False, str(exc)

    if stats_path is not None:
        sp = Path(stats_path)
        try:
            if sp.exists():
                with sp.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            else:
                data = {}
            if not isinstance(data, dict):
                data = {}
            data["carrier_weights"] = dict(CARRIER_WEIGHTS)
            data["delivery_weights"] = dict(DELIVERY_WEIGHTS)
            if extra_stats:
                data.update(extra_stats)
            with sp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            return False, str(exc)

    return True, None


# --- Quota / mismatch manager handles -------------------------------------

_QUOTA_MANAGER: Any = None
_QUOTA_SHARED_STATE: Optional[Any] = None

_MISMATCH_MANAGER: Any = None
_MISMATCH_SHARED_STATE: Optional[Any] = None
_MISMATCH_MANAGER_HANDLE: Any = None


def set_quota_manager(manager: Any) -> None:
    global _QUOTA_MANAGER
    _QUOTA_MANAGER = manager


def get_quota_manager() -> Any:
    return _QUOTA_MANAGER


def set_quota_shared_state(shared_state: Any) -> None:
    global _QUOTA_SHARED_STATE
    _QUOTA_SHARED_STATE = shared_state


def get_quota_shared_state() -> Any:
    return _QUOTA_SHARED_STATE


def set_mismatch_manager(manager: Any) -> None:
    global _MISMATCH_MANAGER
    _MISMATCH_MANAGER = manager


def get_mismatch_manager() -> Any:
    return _MISMATCH_MANAGER


def set_mismatch_shared_state(shared_state: Any) -> None:
    global _MISMATCH_SHARED_STATE
    _MISMATCH_SHARED_STATE = shared_state


def get_mismatch_shared_state() -> Any:
    return _MISMATCH_SHARED_STATE


def set_mismatch_manager_handle(handle: Any) -> None:
    global _MISMATCH_MANAGER_HANDLE
    _MISMATCH_MANAGER_HANDLE = handle


def get_mismatch_manager_handle() -> Any:
    return _MISMATCH_MANAGER_HANDLE


def clear_mismatch_manager_handle() -> None:
    global _MISMATCH_MANAGER_HANDLE
    _MISMATCH_MANAGER_HANDLE = None


# --- Helpers ---------------------------------------------------------------

def as_dict() -> Dict[str, Any]:
    """Return a snapshot of mutable state for diagnostics/testing."""
    return {
        "config": _ACTIVE_CFG,
        "goal_weights": dict(GOAL_WEIGHTS),
        "carrier_weights": dict(CARRIER_WEIGHTS),
        "delivery_weights": dict(DELIVERY_WEIGHTS),
        "audit_counts": dict(AUDIT_REASON_COUNT),
        "audit_queue": MP_AUDIT_QUEUE,
        "quota_manager": _QUOTA_MANAGER,
        "mismatch_manager": _MISMATCH_MANAGER,
    }


def _carrier_weight_lookup(name: str, default: float = 0.05) -> float:
    if name == "chat_dialog":
        return float(DELIVERY_WEIGHTS.get("multi_turn", default))
    return float(CARRIER_WEIGHTS.get(name, default))


def _set_carrier_weight(name: str, value: float) -> None:
    if name == "chat_dialog":
        DELIVERY_WEIGHTS["multi_turn"] = float(value)
    elif name in CARRIER_WEIGHTS:
        CARRIER_WEIGHTS[name] = float(value)


__all__ = [
    "AUDIT_REASON_COUNT",
    "AUDIT_REJECTS",
    "CARRIER_WEIGHTS",
    "DELIVERY_WEIGHTS",
    "GOAL_WEIGHTS",
    "MP_AUDIT_QUEUE",
    "append_audit_record",
    "apply_weight_overrides",
    "cfg_get",
    "clear_mismatch_manager_handle",
    "get_audit_queue",
    "get_config",
    "get_mismatch_manager",
    "get_mismatch_manager_handle",
    "get_mismatch_shared_state",
    "get_quota_manager",
    "get_quota_shared_state",
    "reset_audit_bookkeeping",
    "set_audit_queue",
    "set_config",
    "set_mismatch_manager",
    "set_mismatch_manager_handle",
    "set_mismatch_shared_state",
    "set_quota_manager",
    "set_quota_shared_state",
    "as_dict",
    "load_weights",
    "persist_weights",
    "_carrier_weight_lookup",
    "_set_carrier_weight",
]
