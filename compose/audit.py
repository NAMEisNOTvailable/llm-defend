
"""
Audit helpers for the compose pipeline.
"""

from __future__ import annotations

import sys
import time
import re
import math
import numpy as np  # type: ignore
import multiprocessing as _mp
from queue import Empty
from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from compose import state as compose_state


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    pct = min(max(float(pct), 0.0), 100.0)
    return float(np.percentile(values, pct))


def audit_semantic_neighbors(rows_all, topk: int = 5):
    """Compute high-percentile nearest-neighbor cosine similarity on text fields.
    Returns {'semantic_neighbor_max_sim_p95', 'semantic_neighbor_max_sim_p99'} or {'skipped': reason}.
    """
    try:
        texts = [str((r or {}).get('text', '')) for r in rows_all]
        if not texts:
            return {'skipped': 'empty'}
        vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2)
        X = vec.fit_transform(texts)
        if X.shape[0] < 2:
            return {'skipped': 'not_enough_samples'}
        sims = cosine_similarity(X)
        max_sim = []
        for idx in range(len(texts)):
            row = sims[idx]
            row[idx] = 0.0
            max_sim.append(float(row.max()))
        if not max_sim:
            return {'skipped': 'empty'}
        return {
            'semantic_neighbor_max_sim_p95': _percentile(max_sim, 95),
            'semantic_neighbor_max_sim_p99': _percentile(max_sim, 99),
        }
    except Exception as e:
        return {'skipped': str(e)}


def _drain_audit_queue(q) -> None:
    if q is None:
        return
    while True:
        try:
            rec = q.get_nowait()
        except Empty:
            break
        except Exception:
            break
        if not isinstance(rec, dict):
            continue
        compose_state.append_audit_record(rec)


def _finalize_audit_queue(q) -> None:
    if q is None:
        return
    try:
        _drain_audit_queue(q)
    finally:
        close = getattr(q, 'close', None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        join_thread = getattr(q, 'join_thread', None)
        if callable(join_thread):
            try:
                join_thread()
            except Exception:
                pass


def audit_reject(reason: str, ctx: dict):
    """
    统一的拒绝/告警出口，将阶段/原因/片段/结构写入 AUDIT_*，便于 _stats.json 进一步分析
    """
    try:
        rec = {
            "ts": round(time.time(), 3),
            "reason": reason,
        }
        for k in ("phase", "task", "family", "intent", "pair_id", "combo", "sig", "policy"):
            if k in ctx and ctx[k] is not None:
                rec[k] = ctx[k]
        if "err" in ctx and ctx["err"]:
            rec["err"] = str(ctx["err"])
        if "text" in ctx and ctx["text"]:
            snip = re.sub(r"\s+", " ", str(ctx["text"]).replace("\n", " "))[:160]
            rec["text_snippet"] = snip
        compose_state.append_audit_record(rec)
        q = compose_state.get_audit_queue()
        if q is not None:
            try:
                if _mp.current_process().name != "MainProcess":
                    q.put(rec, block=False)
            except Exception:
                pass
    except Exception as e:
        sys.stderr.write(f"[audit_reject-fail] {reason}: {e}\n")


def audit_soft(reason: str, exc: Exception, ctx: dict | None = None):
    payload = dict(ctx or {})
    try:
        payload["err"] = str(exc)
    except Exception:
        payload["err"] = repr(exc)
    audit_reject(reason, payload)


def get_reject_records() -> list[dict]:
    """Return accumulated audit rejection records."""
    return list(compose_state.AUDIT_REJECTS)


def get_reason_counts() -> Dict[str, int]:
    """Return counts of rejections per reason."""
    return dict(compose_state.AUDIT_REASON_COUNT)


def get_reason_count(reason: str) -> int:
    """Convenience accessor for a single reason count."""
    return int(compose_state.AUDIT_REASON_COUNT.get(reason, 0))


__all__ = [
    'audit_semantic_neighbors',
    '_drain_audit_queue',
    '_finalize_audit_queue',
    'audit_reject',
    'audit_soft',
    'get_reject_records',
    'get_reason_counts',
    'get_reason_count',
]
