"""
Quota enforcement helpers for the compose pipeline.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from compose import state as compose_state
@dataclass
class ArtifactQuotaCfg:
    zero_width: float = 0.15
    bidi_ctrl: float = 0.06
    payload_enc: Dict[str, float] = field(
        default_factory=lambda: {
            "base64": 0.10,
            "url": 0.08,
            "html": 0.05,
            "hex": 0.05,
            "rot13": 0.03,
            "fullwidth": 0.05,
            "zwsp": 0.05,
        }
    )
    double_wrap: float = 0.10
    code_switch: float = 0.18


class QuotaManager:
    """
    Tracks feature/encoding usage to honour sampling quotas across workers.
    """

    def __init__(self, total: int, cfg: ArtifactQuotaCfg, shared_state: Optional[Dict[str, Any]] = None):
        self.N = max(1, int(total))
        self.cfg = cfg
        self.tgt = {
            "zw": int(max(0.0, cfg.zero_width) * self.N),
            "bidi": int(max(0.0, cfg.bidi_ctrl) * self.N),
            "dw": int(max(0.0, cfg.double_wrap) * self.N),
            "cs": int(max(0.0, cfg.code_switch) * self.N),
        }
        payload_cfg = cfg.payload_enc or {}
        self.tgt_enc = {k: int(max(0.0, v) * self.N) for k, v in payload_cfg.items()}
        self._lock = None
        if shared_state:
            self.used = shared_state.get("used") or {}
            self.used_enc = shared_state.get("used_enc") or {}
            self._lock = shared_state.get("lock")
            for key in self.tgt:
                self.used.setdefault(key, 0)
            for key in self.tgt_enc:
                self.used_enc.setdefault(key, 0)
        else:
            self.used = {key: 0 for key in self.tgt}
            self.used_enc = {key: 0 for key in self.tgt_enc}

    @staticmethod
    def _get(mapping: Any, key: str) -> int:
        try:
            return int(mapping.get(key, 0))
        except Exception:
            try:
                return int(mapping[key])
            except Exception:
                return 0

    def need(self, key: str) -> bool:
        tgt = self.tgt.get(key, 0)
        if tgt <= 0:
            return False
        return self._get(self.used, key) < tgt

    def take(self, key: str) -> None:
        if key not in self.tgt:
            return
        if self._lock:
            with self._lock:
                self.used[key] = self._get(self.used, key) + 1
        else:
            self.used[key] = self._get(self.used, key) + 1

    def need_enc(self, enc_key: str) -> bool:
        tgt = self.tgt_enc.get(enc_key, 0)
        if tgt <= 0:
            return False
        return self._get(self.used_enc, enc_key) < tgt

    def take_enc(self, enc_key: str) -> None:
        if enc_key not in self.tgt_enc:
            return
        if self._lock:
            with self._lock:
                self.used_enc[enc_key] = self._get(self.used_enc, enc_key) + 1
        else:
            self.used_enc[enc_key] = self._get(self.used_enc, enc_key) + 1

    def choose_encoding(self, rng: random.Random) -> Optional[str]:
        candidates: List[Tuple[str, int]] = []
        for key, tgt in self.tgt_enc.items():
            remaining = tgt - self._get(self.used_enc, key)
            if remaining > 0:
                candidates.append((key, remaining))
        if not candidates:
            return None
        max_rem = max(rem for _, rem in candidates)
        top = [key for key, rem in candidates if rem == max_rem]
        return rng.choice(top) if top else None


def _init_shared_quota_state(cfg: ArtifactQuotaCfg) -> tuple[Any, Dict[str, Any]]:
    manager = mp.Manager()
    used = manager.dict({key: 0 for key in ("zw", "bidi", "dw", "cs")})
    payload_cfg = cfg.payload_enc or {}
    used_enc = manager.dict({k: 0 for k in payload_cfg.keys()})
    lock = manager.Lock()
    return manager, {"used": used, "used_enc": used_enc, "lock": lock}


class quota_scope:
    """
    Context manager that installs a QuotaManager into global state, sharing
    counters across workers when requested.
    """

    def __init__(self, total: int, cfg: ArtifactQuotaCfg, shared: bool = False):
        self.total = max(1, int(total))
        self.cfg = cfg
        self.shared = shared
        self.prev: Optional[QuotaManager] = None
        self.prev_shared: Any = None
        self.manager = None
        self.state = None

    def __enter__(self) -> Optional[Dict[str, Any]]:
        self.prev = compose_state.get_quota_manager()
        self.prev_shared = compose_state.get_quota_shared_state()
        if self.shared:
            self.manager, self.state = _init_shared_quota_state(self.cfg)
            set_quota_manager(QuotaManager(self.total, self.cfg, shared_state=self.state))
            compose_state.set_quota_shared_state(self.state)
            return {"cfg": self.cfg, "N": self.total, "state": self.state}
        set_quota_manager(QuotaManager(self.total, self.cfg))
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        set_quota_manager(self.prev)
        compose_state.set_quota_shared_state(self.prev_shared)
        if self.manager:
            try:
                self.manager.shutdown()
            except Exception:
                pass
        self.manager = None
        self.state = None


def set_quota_manager(manager: Optional[QuotaManager]) -> None:
    compose_state.set_quota_manager(manager)
    legacy = sys.modules.get("make_malicious_prompts_cn_compose_v2")
    if legacy is not None:
        try:
            setattr(legacy, "_QUOTA_MANAGER", manager)
        except Exception:
            pass


def should_apply_feature(key: str, base_prob: float, rng: random.Random) -> bool:
    qm = compose_state.get_quota_manager()
    if qm:
        if qm.need(key):
            qm.take(key)
            return True
        return False
    return rng.random() < base_prob


__all__ = [
    "ArtifactQuotaCfg",
    "QuotaManager",
    "quota_scope",
    "set_quota_manager",
    "should_apply_feature",
]
