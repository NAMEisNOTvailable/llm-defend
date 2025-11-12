"""
Mismatch carrier management and shared quota tracking.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import random
from collections import Counter, defaultdict
from contextlib import nullcontext
from typing import Any, Dict, Optional

from compose import state as compose_state
from compose.state import cfg_get

RUNTIME_NEG_ACCEPTS_BY_CARRIER = defaultdict(int)

MISMATCH_CATALOG: Dict[str, set[str]] = {
    "class_label": {"yaml_front_matter", "md_ref_link", "http_header"},
    "summary": {"csv_formula", "plantuml", "git_diff"},
    "entities": {"bibtex", "mermaid", "svg"},
    "kv": {"table_md", "jinja"},
    "outline": {"robots", "api_log", "email_chain"},
}

MISMATCH_QUOTA: Dict[str, float] = {
    "class_label": 0.15,
    "summary": 0.12,
    "entities": 0.10,
    "kv": 0.10,
    "outline": 0.08,
}


def should_fill_quota_for(carrier: str, cfg: Any = None) -> bool:
    base = (carrier or "none").split("+")[0]
    target = int(cfg_get("coverage_min_per_combo", 0, cfg) or 0)
    if target <= 0:
        return False
    return RUNTIME_NEG_ACCEPTS_BY_CARRIER.get(base, 0) < target


class MismatchTracker:
    """
    Tracks how often mismatch carriers are sampled so quota hints can be honoured.
    """

    def __init__(self, shared_state: Optional[Dict[str, Any]] = None):
        self._used: Any = Counter()
        self._total: Any = Counter()
        self._lock = None
        if shared_state is not None:
            used = shared_state.get("used")
            total = shared_state.get("total")
            if used is not None:
                self._used = used
            if total is not None:
                self._total = total
            self._lock = shared_state.get("lock")

    @staticmethod
    def _get(mapping: Any, key: str) -> int:
        try:
            return int(mapping.get(key, 0))
        except AttributeError:
            try:
                return int(mapping[key])
            except Exception:
                return 0
        except Exception:
            return 0

    @staticmethod
    def _set(mapping: Any, key: str, value: int) -> None:
        try:
            mapping[key] = int(value)
        except Exception:
            pass

    def need(self, key: str, quota: float, target_min: int) -> bool:
        quota = max(0.0, float(quota))
        if quota <= 0.0:
            return False
        used = self._get(self._used, key)
        total = self._get(self._total, key)
        if total < target_min:
            return True
        return (used / max(1, total)) < quota

    def mark(self, key: str, accepted: bool) -> None:
        if self._lock:
            with self._lock:
                self._mark_locked(key, accepted)
        else:
            self._mark_locked(key, accepted)

    def _mark_locked(self, key: str, accepted: bool) -> None:
        self._set(self._total, key, self._get(self._total, key) + 1)
        if accepted:
            self._set(self._used, key, self._get(self._used, key) + 1)

    def choose(
        self,
        contract_mode: str,
        pool: list[str],
        chooser,
        rand_fn,
        quota: float,
        target_min: int,
    ) -> Optional[str]:
        if not pool:
            return None
        quota = max(0.0, float(quota))
        if quota <= 0.0:
            return None
        name = chooser(pool)
        if not name:
            return None
        with (self._lock or nullcontext()):
            favoured = self.need(contract_mode, quota, target_min)
            accept = bool(favoured or rand_fn() < quota)
            self._mark_locked(contract_mode, accept)
        return name if accept else None


def set_mismatch_manager(manager: Optional[MismatchTracker]) -> None:
    compose_state.set_mismatch_manager(manager or MismatchTracker())


def get_mismatch_manager() -> MismatchTracker:
    manager = compose_state.get_mismatch_manager()
    if manager is None:
        manager = MismatchTracker()
        compose_state.set_mismatch_manager(manager)
    return manager


def ensure_shared_mismatch_state() -> Dict[str, Any]:
    state = compose_state.get_mismatch_shared_state()
    if state is None:
        mgr = mp.Manager()
        state = {"used": mgr.dict(), "total": mgr.dict(), "lock": mgr.Lock()}
        compose_state.set_mismatch_manager_handle(mgr)
        compose_state.set_mismatch_shared_state(state)
    set_mismatch_manager(MismatchTracker(shared_state=state))
    return state


def _shutdown_mismatch_state() -> None:
    mgr = compose_state.get_mismatch_manager_handle()
    if mgr is not None:
        try:
            mgr.shutdown()
        except Exception:
            pass
        compose_state.clear_mismatch_manager_handle()


atexit.register(_shutdown_mismatch_state)


def choose_mismatch_carrier(contract_mode: str, rng: random.Random) -> str:
    pool = list(MISMATCH_CATALOG.get(contract_mode, []))
    if not pool:
        return "none"
    chooser = getattr(rng, "choice", None) or random.choice
    rand_fn = getattr(rng, "random", None) or random.random
    quota = float(MISMATCH_QUOTA.get(contract_mode, 0.0) or 0.0)
    target = int(cfg_get("coverage_min_per_combo", 3) or 0)
    tracker = get_mismatch_manager()
    selection = tracker.choose(contract_mode, pool, chooser, rand_fn, quota, max(1, target))
    return selection if selection else "none"


__all__ = [
    "MISMATCH_CATALOG",
    "MISMATCH_QUOTA",
    "MismatchTracker",
    "RUNTIME_NEG_ACCEPTS_BY_CARRIER",
    "choose_mismatch_carrier",
    "ensure_shared_mismatch_state",
    "get_mismatch_manager",
    "set_mismatch_manager",
    "should_fill_quota_for",
]
