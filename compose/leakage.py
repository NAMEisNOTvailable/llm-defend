"""
Leakage diagnostics utilities extracted from the legacy compose script.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore
from compat_regex import regex as _re

from compose.dedupe_helpers import placeholderless_mirror
from compose.surface_noise import add_struct_evidence_benign
from compose.utils import CN_latin_ratio, byte_len, feature_probe_clean

__all__ = ("leakage_pressure_test", "mitigate_leakage")


def _char_ngrams(s: str, n: int = 3, cap: int = 2048) -> List[str]:
    s = _re.sub(r"\s+", " ", s or "").strip()
    s = s[:cap]
    return [s[i : i + n] for i in range(max(0, len(s) - n + 1))]


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    q = min(max(float(q), 0.0), 1.0)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def leakage_pressure_test(
    rows: List[Dict[str, Any]],
    n: int = 3,
    vocab_size: int = 300,
    threshold: float = 0.85,
    return_details: bool = False,
) -> Optional[Dict[str, Any]]:
    texts = [r.get("text", "") for r in rows]
    labels = [int(r.get("label", 0)) for r in rows]
    if not any(labels) or all(labels):
        print("[leakage] skipped: single-class")
        empty = {
            "auc": 0.0,
            "scores": [0.0 for _ in rows],
            "labels": labels,
            "coefs": [],
            "vocab": [],
            "threshold": threshold,
        }
        return empty if return_details else None

    freq = Counter()
    for text in texts:
        freq.update(_char_ngrams(text, n=n))
    vocab = [gram for gram, _ in freq.most_common(vocab_size)]
    idx = {gram: i for i, gram in enumerate(vocab)}

    pos_c = np.ones(len(vocab))
    neg_c = np.ones(len(vocab))
    pos_tot = 1.0
    neg_tot = 1.0
    for text, label in zip(texts, labels):
        seen = set(_char_ngrams(text, n=n))
        for gram in seen:
            j = idx.get(gram)
            if j is None:
                continue
            if label == 1:
                pos_c[j] += 1  # type: ignore[operator]
                pos_tot += 1
            else:
                neg_c[j] += 1  # type: ignore[operator]
                neg_tot += 1

    coefs_arr = np.log(pos_c / pos_tot) - np.log(neg_c / neg_tot)
    coefs = coefs_arr.tolist()
    scores: List[float] = []
    for text in texts:
        grams = set(_char_ngrams(text, n=n))
        score = 0.0
        for gram in grams:
            j = idx.get(gram)
            if j is not None:
                score += float(coefs[j])
        scores.append(score)

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = {i: r + 1 for r, i in enumerate(order)}
    pos_idx = [i for i, label in enumerate(labels) if label == 1]
    neg_idx = [i for i, label in enumerate(labels) if label == 0]
    if not pos_idx or not neg_idx:
        print("[leakage] skipped: single-class after split")
        empty = {
            "auc": 0.0,
            "scores": scores,
            "labels": labels,
            "coefs": [float(x) for x in coefs],
            "vocab": vocab,
            "threshold": threshold,
        }
        return empty if return_details else None

    sum_ranks_pos = sum(ranks[i] for i in pos_idx)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-9)
    print(f"[leakage] char-{n}-gram linear AUC={auc:.3f} (vocab={len(vocab)})")
    if auc >= threshold:
        order = list(np.argsort(-np.abs(coefs)))[:10]
        tops = [vocab[i] for i in order]
        print("[leakage][warn] high AUC; top n-grams:", tops)

    details = {
        "auc": float(auc),
        "scores": scores,
        "labels": labels,
        "coefs": [float(x) for x in coefs],
        "vocab": vocab,
        "threshold": float(threshold),
        "order": int(n),
    }
    return details if return_details else None


def mitigate_leakage(
    rows: List[Dict[str, Any]],
    leak_info: Dict[str, Any],
    iteration: int = 0,
    pos_quantile: float = 0.85,
    neg_quantile: float = 0.20,
) -> int:
    scores = leak_info.get("scores") or []
    labels = leak_info.get("labels") or []
    if not scores or len(scores) != len(rows):
        return 0

    acted = 0
    pos_idx = [i for i, label in enumerate(labels) if label == 1]
    if pos_idx:
        pos_scores = [float(scores[i]) for i in pos_idx]
        cut = _quantile(pos_scores, min(max(pos_quantile, 0.0), 1.0))
        if cut is None:
            cut = max(pos_scores)
        for i in pos_idx:
            if scores[i] >= cut:
                new_text = placeholderless_mirror(rows[i]["text"])
                if new_text and new_text.strip() and new_text.strip() != rows[i]["text"].strip():
                    rows[i]["text"] = new_text
                    rows[i].setdefault("leakage_adjustments", []).append(
                        {"iteration": int(iteration), "action": "placeholderless_mirror"}
                    )
                    rows[i]["len"] = byte_len(new_text)
                    rows[i]["feats"] = feature_probe_clean(
                        new_text,
                        carrier=rows[i].get("carrier"),
                        payload_encoding=rows[i].get("payload_encoding"),
                    )
                    rows[i]["latin_over_CN"] = CN_latin_ratio(new_text)
                    acted += 1

    neg_idx = [i for i, label in enumerate(labels) if label == 0]
    if neg_idx:
        neg_scores = [float(scores[i]) for i in neg_idx]
        cut_neg = _quantile(neg_scores, min(max(neg_quantile, 0.0), 1.0))
        if cut_neg is None:
            cut_neg = min(neg_scores)
        for i in neg_idx:
            if scores[i] <= cut_neg:
                new_text = add_struct_evidence_benign(rows[i]["text"], p=1.0)
                if new_text != rows[i]["text"]:
                    rows[i]["text"] = new_text
                    rows[i].setdefault("leakage_adjustments", []).append(
                        {"iteration": int(iteration), "action": "add_struct_evidence"}
                    )
                    rows[i]["len"] = byte_len(new_text)
                    rows[i]["feats"] = feature_probe_clean(
                        new_text,
                        carrier=rows[i].get("carrier"),
                        payload_encoding=rows[i].get("payload_encoding"),
                    )
                    rows[i]["latin_over_CN"] = CN_latin_ratio(new_text)
                    acted += 1

    high_take = max(1, len(scores) // 10 or 1)
    for i in sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:high_take]:
        rows[i]["weight"] = float(max(0.25, float(rows[i].get("weight", 1.0)) * 0.7))
        rows[i].setdefault("leakage_adjustments", []).append(
            {"iteration": int(iteration), "action": "downweight"}
        )
    return acted
