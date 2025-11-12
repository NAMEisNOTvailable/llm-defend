"""Generation pipeline, invariant adjudication, and coverage budgeting."""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from dedupe_core import Deduper, DedupeRecord

from .anchors import (
    Anchors,
    APPEND_RE,
    PATH_RE,
    ROLE_RE,
    TOOLCALL_RE,
    UP_RE,
    strip_anchors,
)
from .invariants import EVIDENCE_CHECK, invariant_loop_state_drift, invariant_upstream
from .renderers import CARRIER_RENDER
from .soft import (
    FIELD_ALIAS_CARRIERS,
    LANG_CARRIERS,
    STYLE_ADAPTER_CARRIERS,
    _SOFT_RX,
    apply_delivery,
    apply_style,
    evidence_families,
    inject_soft_hints,
    mechanism_truth_record,
    soft_evidence_kinds,
    soft_semantic_hit,
    _soft_kind_weight,
)
from .spec import AttackSpec, MECHANISM_KEYS, sample_spec
from .textops import randomize_field_aliases, sinicize_surface
from .utils import (
    CN_share,
    bucket_hash,
    _looks_like_email,
    _looks_like_http_header,
    _looks_like_yaml,
    _probe_audit,
    _resolve_structural_ratio,
)

__all__ = [
    "ANCHOR_OVERRIDE_FALLBACK_RX",
    "Deduper",
    "DedupeRecord",
    "EVD_HINT_CODECMT",
    "MARKDOWN_TABLE_ROW_RX",
    "MULTITURN_ASSIST_RESULT_RX",
    "MULTITURN_JSON_TURNS_RX",
    "MULTITURN_ROLE_RX",
    "MULTITURN_USER_RESULT_RX",
    "adjudicate_invariants",
    "count_by",
    "generate_batch",
    "infer_contract_type",
    "semantic_signature",
]

MARKDOWN_TABLE_ROW_RX = re.compile(r"(?m)^\s*\|[^|\n]+\|[^|\n]+\|")
MULTITURN_ROLE_RX = re.compile(r"^(?:\[?(?:用户|助手|User|Assistant|我|你|问|答|甲|乙)\]?)[\s：:>\-|]*", re.M)
MULTITURN_JSON_TURNS_RX = re.compile(r'"turns"\s*:\s*\[', re.I)
MULTITURN_USER_RESULT_RX = re.compile(r"(?m)^用户\s*:.*(结果槽|RESULT_MAIN)")
MULTITURN_ASSIST_RESULT_RX = re.compile(r"(?m)^助手\s*:.*(结果槽|RESULT_MAIN)")
ANCHOR_OVERRIDE_FALLBACK_RX = re.compile(r"(覆盖|忽略|override|替换上文|重设|优先采用)", re.I)
EVD_HINT_CODECMT = re.compile(r"(?:NOTE|Hint|提示|说明|仅按本段|仅输出|最后一行)")

def adjudicate_invariants(text: str, spec: AttackSpec) -> Tuple[bool, List[str]]:
    """
    Validate carrier shape and structural/soft evidence invariants.
    校验载体形状与结构/软证据不变量。

    - English: Enforces carrier-specific form, checks evidence hits via
      structural patterns (and soft hints under anchor-free), and requires
      minimal CJK share for language-carriers.
    - 中文：按载体形状约束、结构模式命中（无锚时允许软证据），并对语言载体要求
      最低中文占比。
    """
    fails: List[str] = []
    soft_hits_weight = 0.0
    evidence_items = list(spec.evidence or [])
    constraints = getattr(spec, "constraints", {}) or {}
    required_soft_weight = max(2.0, float(constraints.get("anchor_free_soft_min_weight", 2.0)))
    anchor_free_flag = bool(constraints.get("anchor_free"))
    carrier = spec.carrier
    text_stripped = text.strip()

    # 载体形态检查（例：jsonc 必须有 ```json fence）
    if carrier == "jsonc" and not text_stripped.startswith("```json"):
        fails.append("carrier_jsonc_missing_fence")
    if carrier == "markdown_table":
        rows = MARKDOWN_TABLE_ROW_RX.findall(text)
        if len(rows) < 3:  # 至少一行表头 + 两行数据
            fails.append("carrier_table_structure_weak")

    # 证据不变量
    for ev in evidence_items:
        check_fn = EVIDENCE_CHECK.get(ev)
        ok_ev = bool(check_fn(text)) if check_fn else True
        if (not ok_ev) and anchor_free_flag and soft_semantic_hit(ev, text):
            soft_hits_weight += _soft_kind_weight(ev)
            ok_ev = True
        if not ok_ev:
            fails.append(f"evidence_missing:{ev}")

    # anchor_free 下对“上游/检索”证据施加半结构刚性：必须出现显式字段或头
    if anchor_free_flag and "upstream" in evidence_items:
        if not invariant_upstream(text):  # 未命中上面的结构检查
            fails.append("evidence_semicontract_missing:upstream")

    if anchor_free_flag:
        kinds = soft_evidence_kinds(text)
        min_hits = int(constraints.get("anchor_free_soft_min_hits", 3))
        weight_score = sum(_soft_kind_weight(kind) for kind in kinds)
        required_weight_current = max(required_soft_weight, 2.0, float(min_hits) * 0.75)
        soft_hits_weight = max(soft_hits_weight, weight_score)
        if weight_score < required_weight_current:
            fails.append("soft_evidence_insufficient")

    # 中文占比：仅对“自然语/文档类载体”强制，代码/协议类放宽
    if carrier in LANG_CARRIERS and CN_share(text) < spec.min_CN_share:
        fails.append("cjk_share_below_threshold")
        fails.append("fix:cjk_alias")
    if carrier == "yaml" and not _looks_like_yaml(text):
        fails.append("carrier:yaml")
    if carrier == "code_comment" and "/*" not in text and "//" not in text:
        fails.append("carrier:code_comment")
    if carrier == "email" and not _looks_like_email(text):
        fails.append("carrier:email")
    if carrier == "http_header" and not _looks_like_http_header(text):
        fails.append("carrier:http_header")
    if carrier == "csv_formula" and "=" not in text:
        fails.append("carrier:csv_formula")
    if carrier == "git_diff" and "```diff" not in text:
        fails.append("carrier:git_diff")
    if carrier == "latex" and "\\begin{comment}" not in text:
        fails.append("carrier:latex")
    if carrier == "mermaid" and "```mermaid" not in text:
        fails.append("carrier:mermaid")
    if carrier == "front_matter" and not text_stripped.startswith("---"):
        fails.append("carrier:front_matter")

    # 交付模式：multi_turn 基本不变量
    if spec.delivery == "multi_turn" and not fails:
        roles = MULTITURN_ROLE_RX.findall(text)
        json_turns = bool(MULTITURN_JSON_TURNS_RX.search(text))
        if not (json_turns or len(roles) >= 3):
            fails.append("delivery:multi_turn_role_format_weak")
        if not (invariant_loop_state_drift(text) or _SOFT_RX["loop_state_drift"].search(text)):
            fails.append("delivery_missing:multi_turn_state")
        if not anchor_free_flag and "result_slot" in evidence_items:
            if MULTITURN_USER_RESULT_RX.search(text):
                fails.append("delivery:multi_turn_user_contains_result")
            if not MULTITURN_ASSIST_RESULT_RX.findall(text):
                fails.append("delivery:multi_turn_missing_result_in_assistant")
        if anchor_free_flag:
            _contract_conflict = bool(
                _SOFT_RX["role"].search(text)
                or _SOFT_RX["contract_override"].search(text)
                or ANCHOR_OVERRIDE_FALLBACK_RX.search(text)
            )
            soft_min = int(constraints.get("anchor_free_soft_min_hits", 3))
            required_weight_mt = max(required_soft_weight, 2.0, float(soft_min) * 0.75)
            if not (_contract_conflict and soft_hits_weight >= required_weight_mt):
                fails.append("anchor_free_soft_evidence_too_weak")
        if carrier == "code_comment":
            if ("/*" not in text) and ("//" not in text):
                fails.append("carrier:code_comment")
            elif anchor_free_flag and not EVD_HINT_CODECMT.search(text or ""):
                fails.append("carrier:code_comment_soft_hint_missing")

    return (len(fails) == 0), fails

# ---------- 8) 语义签名与覆盖报告 ----------
def semantic_signature(spec: AttackSpec, text: str) -> str:
    """
    去锚点签名：以“机制/契约类型”为主，不把 openpi_* 计为独立机制；
    纳入 anchor_free 与证据家族，避免不同锚点态/证据族被过度聚合。
    """
    evh = bucket_hash(spec.evidence)
    anchor_flag = int(bool(getattr(spec, "constraints", {}) and spec.constraints.get("anchor_free")))
    fam = "|".join(sorted(evidence_families(spec.evidence)))
    mech = infer_contract_type(spec)
    appearance = ("openpi" if str(getattr(spec, 'strategy', ''))[:7] == 'openpi_' else 'normal')
    keys = [f"mech={mech}", spec.channel, spec.carrier, spec.delivery,
            f"anchor_free={anchor_flag}", f"appearance={appearance}", f"evfam={fam}", f"evh={evh}",
            f"speech={getattr(spec,'speech_family','formal')}"]
    return "::".join(keys)

@lru_cache(maxsize=4096)
def _infer_contract_type_cached(ev_key: Tuple[str, ...], strategy: str) -> str:
    ev = set(ev_key)
    if "tool" in ev:              return "tool_override"
    if "role" in ev:              return "role_takeover"
    if "result_slot" in ev:       return "slot_override"
    if "append" in ev:            return "append_override"
    if "path" in ev:              return "path_hijack"
    if "upstream" in ev:          return "upstream_reveal"
    if "memory_write" in ev:      return "state_poison"
    if "loop_state_drift" in ev:  return "loop_drift"
    if "reward_field" in ev or "reward_channel" in ev:
                                  return "reward_hacking"
    return strategy


def infer_contract_type(spec: AttackSpec) -> str:
    strategy = str(getattr(spec, "strategy", "format_anchor") or "format_anchor")
    ev_key = tuple(sorted(spec.evidence or []))
    return _infer_contract_type_cached(ev_key, strategy)


# ---------- Mechanism coverage helpers ----------
def _infer_trigger_path(delivery: str, carrier: str) -> str:
    c = (carrier or "").lower()
    dm = (delivery or "").lower()
    if c in {"jsonc", "api_log", "http_header", "csv_formula"}:
        return "format_carrier"
    if c in {"yaml_front_matter", "front_matter", "yaml", "md_ref_link", "table_md"}:
        return "rag_doc"
    if c in {"git_diff", "jinja", "latex", "mermaid", "code_comment"}:
        return "in_context"
    if dm in {"multi_turn"}:
        return "multi_turn"
    if dm in {"long_web", "indirect"}:
        return "rag_doc"
    if dm in {"direct", "format"}:
        return "in_context"
    return "unknown"

def _prepare_mechanism_payload(text: str, spec: 'AttackSpec', mechanism_signal_fn: Optional[Callable[[str, 'AttackSpec'], Any]]) -> Tuple[Dict[str, bool], Dict[str, Any], List[str]]:
    flags = {k: False for k in MECHANISM_KEYS}
    diagnostics: Dict[str, Any] = {}
    events: List[str] = []
    raw: Any = {}
    if mechanism_signal_fn is not None:
        try:
            raw = mechanism_signal_fn(text, spec) or {}
        except Exception as exc:
            _probe_audit("mechanism_probe_error", {"err": str(exc), "strategy": getattr(spec, "strategy", None)})
            raw = {}
    if isinstance(raw, tuple):
        if len(raw) >= 1 and isinstance(raw[0], dict):
            for k in MECHANISM_KEYS:
                flags[k] = bool(raw[0].get(k, False))
        if len(raw) >= 2 and isinstance(raw[1], dict):
            diagnostics = raw[1]
        if len(raw) >= 3 and isinstance(raw[2], (list, tuple)):
            events = [str(x) for x in raw[2]]
    elif isinstance(raw, dict):
        if isinstance(raw.get("flags"), dict):
            for k in MECHANISM_KEYS:
                flags[k] = bool(raw["flags"].get(k, False))
        else:
            for k in MECHANISM_KEYS:
                if k in raw:
                    flags[k] = bool(raw.get(k))
        diag_src = raw.get("diagnostics") or raw.get("signals")
        if isinstance(diag_src, dict):
            diagnostics = diag_src
        events_src = raw.get("events")
        if isinstance(events_src, (list, tuple)):
            events = [str(x) for x in events_src]
    if diagnostics is None:
        diagnostics = {}
    evidence = set(getattr(spec, "evidence", []) or [])
    if "role" in evidence:
        flags["role"] = True
    if "tool" in evidence or getattr(spec, "channel", "") == "tool":
        flags["tool"] = True
    if evidence & {"upstream", "path"}:
        flags["retriever"] = True
    if "memory_write" in evidence:
        flags["memory"] = True
    if evidence & {"reward_channel", "reward_field"}:
        flags["reward"] = True
    if getattr(spec, "delivery", "") == "multi_turn":
        flags["role"] = True
    if not events:
        events = [f"EV:{ev}" for ev in sorted(evidence)] if evidence else []
    return flags, diagnostics, events

def _flatten_diag_tokens(data: Dict[str, Any]) -> Set[str]:
    tokens: Set[str] = set()
    if not isinstance(data, dict):
        return tokens
    for key, value in data.items():
        name = f"sig:{key}"
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, bool) and sub_val:
                    tokens.add(f"{name}:{sub_key}")
                elif isinstance(sub_val, (int, float)) and float(sub_val):
                    tokens.add(f"{name}:{sub_key}")
                elif isinstance(sub_val, str) and sub_val:
                    tokens.add(f"{name}:{sub_key}:{sub_val[:24]}")
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                if isinstance(item, str) and item:
                    tokens.add(f"{name}:{item[:24]}")
                elif isinstance(item, (int, float)) and float(item):
                    tokens.add(f"{name}:{idx}")
                elif isinstance(item, dict):
                    for sub_key, sub_val in item.items():
                        if isinstance(sub_val, bool) and sub_val:
                            tokens.add(f"{name}:{sub_key}")
        else:
            if isinstance(value, bool):
                if value:
                    tokens.add(name)
            elif isinstance(value, (int, float)):
                if float(value):
                    tokens.add(name)
            elif isinstance(value, str) and value:
                tokens.add(f"{name}:{value[:24]}")
    return tokens

def _mechanism_signature_tokens(events: List[str], diagnostics: Dict[str, Any], mechanisms: List[str], evidence: Iterable[str]) -> Set[str]:
    tokens: Set[str] = set()
    seq = [str(e) for e in events or []]
    for e in seq:
        tokens.add(f"ev:{e}")
    for i in range(len(seq) - 1):
        tokens.add(f"ev2:{seq[i]}>{seq[i+1]}")
    for i in range(len(seq) - 2):
        tokens.add(f"ev3:{seq[i]}>{seq[i+1]}>{seq[i+2]}")
    tokens |= _flatten_diag_tokens(diagnostics)
    sorted_mechs = sorted(mechanisms)
    for m in sorted_mechs:
        tokens.add(f"mech:{m}")
    if sorted_mechs:
        tokens.add(f"mech_combo:{'+'.join(sorted_mechs)}")
    for ev in sorted(evidence or []):
        tokens.add(f"evidence:{ev}")
    return tokens

def _mechanism_near_dup(existing: List[frozenset[str]], candidate: Set[str], thresh: float) -> bool:
    if not candidate:
        return False
    cand = set(candidate)
    for prev in existing:
        union = len(prev | cand)
        if union == 0:
            continue
        inter = len(prev & cand)
        if inter / union >= thresh:
            return True
    return False

def _format_mech_combo_key(key: Tuple[Tuple[str, ...], str]) -> str:
    mech, path = key
    mech_part = "+".join(mech) if mech else "_none"
    return f"{mech_part}@{path}"

def _choose_combo_rr(
    by_combo: Dict[Tuple[Any, ...], int],
    want_axes: List[str],
    min_per_combo: int,
    rng: random.Random,
) -> Tuple[Optional[Tuple[Any, ...]], int]:
    """Select the combo with the largest deficit (round-robin with jitter) to keep coverage balanced."""
    deficits: List[Tuple[Tuple[float, float], Tuple[Any, ...]]] = []
    for key, count in by_combo.items():
        deficit = max(0, min_per_combo - count)
        score = (deficit, rng.random())
        deficits.append((score, key))
    _, pick = max(deficits) if deficits else (None, None)
    if pick is None:
        return None, 0
    return pick, max(0, min_per_combo - by_combo.get(pick, 0))

# ---------- 9) 生成一批样本（含覆盖预算/重试） ----------
def generate_batch(
     n: int,
     seed: Optional[int] = None,
     pin: Optional[Dict] = None,
     coverage_axes: Tuple[str, ...] = ("contract_type", "channel", "carrier", "delivery", "anchor_free", "ev_bucket", "ev_family", "appearance"),
     min_per_combo: int = 0,
     min_per_cfam: int = 0,
     audit_cb=None, cjk_ratio_fn=None,
     mechanism_signal_fn: Optional[Callable[[str, 'AttackSpec'], Any]] = None
):
    """
    Generate a batch of positive samples with coverage budgeting and dedupe.
    生成一批正样本，并进行覆盖预算与语义去重。

    - English: Iteratively sample specs, render carriers, apply delivery,
      obfuscate anchors (if anchor‑free), validate invariants, and enforce
      coverage quotas and near‑duplicate filters.
    - 中文：迭代采样规格、渲染载体、封装交付、在无锚模式下去锚，校验不变量，并
      执行覆盖配额与近重复过滤。
    """
    rnd = random.Random(seed)
    out, coverage = [], {}
    seen_signatures: Set[str] = set()
    by_combo = defaultdict(int)
    by_mech = defaultdict(int)
    by_cfam = defaultdict(int)
    by_mech_combo = defaultdict(int)
    mech_attempts = defaultdict(int)
    mech_success = defaultdict(int)
    mech_fail = defaultdict(lambda: defaultdict(int))
    mech_blocked: Set[Tuple[Tuple[str, ...], str]] = set()
    mech_sig_cache: Dict[Tuple[Tuple[str, ...], str], List[frozenset[str]]] = defaultdict(list)
    fail_by_combo = defaultdict(lambda: defaultdict(int))
    signature_cache = {}
    combo_texts = defaultdict(list)
    combo_ngrams = defaultdict(list)
    dedupe_combo: Dict[Tuple, Deduper] = {}
    dedupe_mech: Dict[Tuple, Deduper] = {}
    dedupe_style: Dict[Tuple, Deduper] = {}
    reject_cjk_by_carrier = defaultdict(int)
    attempts_by_carrier = defaultdict(int)
    alias_hits_by_carrier = defaultdict(int)
    sinicize_hits_by_carrier = defaultdict(int)
    struct_cn_hits_by_carrier = defaultdict(int)

    sim_bits = 64
    near_dup_thr = float((pin or {}).get("near_dup_thr", 0.92))
    near_thr_local = min(near_dup_thr, 0.88)
    vec_thr = float((pin or {}).get("near_dup_vec_thr", 0.93))
    jacc_thr = float((pin or {}).get("near_dup_ngram_thr", 0.88))
    global_sim_thr = float((pin or {}).get("near_dup_global_sim_thr", 0.95))
    global_jacc_thr = float((pin or {}).get("near_dup_global_ngram_thr", 0.90))
    mechanism_jacc_thr = float((pin or {}).get("mechanism_jaccard_thresh", 0.78))

    def _dedupe_normalize(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower()).strip()

    def _simhash_dist_for(similarity: float) -> int:
        similarity = max(0.0, min(1.0, similarity))
        return max(0, int(math.floor((1.0 - similarity) * sim_bits)))

    local_sim_thresh = _simhash_dist_for(near_thr_local)
    global_sim_thresh = _simhash_dist_for(global_sim_thr)
    vec_dim_local = int((pin or {}).get("vec_dim_override", (1024 if vec_thr < 0.999 else 0)))

    dedupe_template = Deduper(
        sim_bits=sim_bits,
        sim_thresh=local_sim_thresh,
        k=3,
        n_hash=64,
        bands=16,
        jaccard_thresh=jacc_thr,
        vec_dim=vec_dim_local,
        cosine_thresh=vec_thr,
        normalizer=_dedupe_normalize,
    )

    def _make_local_deduper() -> Deduper:
        return Deduper(
            sim_bits=sim_bits,
            sim_thresh=local_sim_thresh,
            k=3,
            n_hash=64,
            bands=16,
            jaccard_thresh=jacc_thr,
            vec_dim=vec_dim_local,
            cosine_thresh=vec_thr,
            normalizer=_dedupe_normalize,
        )

    dedupe_global = Deduper(
        sim_bits=sim_bits,
        sim_thresh=global_sim_thresh,
        k=3,
        n_hash=64,
        bands=16,
        jaccard_thresh=global_jacc_thr,
        vec_dim=0,
        cosine_thresh=1.0,
        normalizer=_dedupe_normalize,
    )

    reason_map_local = {
        "simhash": "semantic_near_dup",
        "jaccard": "semantic_ngram_near_dup",
        "cosine": "semantic_vec_near_dup",
        "external": "semantic_vec_near_dup",
    }
    reason_map_global = {
        "simhash": "semantic_global_near_dup",
        "jaccard": "semantic_global_ngram_near_dup",
        "cosine": "semantic_global_vec_near_dup",
        "external": "semantic_global_vec_near_dup",
    }

    def _snapshot_record(rec: DedupeRecord) -> Dict[str, object]:
        mh = rec.minhash
        if hasattr(mh, "hashvalues"):
            try:
                minhash_snapshot = tuple(int(x) for x in mh.hashvalues)  # type: ignore[attr-defined]
            except Exception:
                minhash_snapshot = ()
        else:
            try:
                minhash_snapshot = tuple(int(x) for x in mh)
            except TypeError:
                try:
                    minhash_snapshot = tuple(int(x) for x in list(mh))  # type: ignore[arg-type]
                except Exception:
                    minhash_snapshot = ()
        return {
            "normalized": rec.normalized,
            "simhash": rec.simhash,
            "minhash": minhash_snapshot,
            "shingles": frozenset(rec.shingles),
            "vector": tuple(rec.vector) if rec.vector else None,
        }
    audit_cb = audit_cb or (lambda *a, **k: None)
    cjk_ratio_fn = cjk_ratio_fn or CN_share
    soft_counts_hist = defaultdict(int)          # {命中种数: 样本数}
    soft_kinds_totals = defaultdict(int)         # {软证据类别: 频次}
    soft_used_by_ev = defaultdict(int)           # {证据ev: 使用软证据通过的次数}
    max_nohit_attempts = max(1, int((pin or {}).get("max_nohit_attempts", 20)))
    default_block_threshold = max(1, max_nohit_attempts // 2)
    block_fail_threshold = int((pin or {}).get("combo_block_threshold", default_block_threshold))
    block_fail_threshold = max(1, min(block_fail_threshold, max_nohit_attempts))
    attempts_per_combo = defaultdict(int)
    success_per_combo  = defaultdict(int)
    blocked_combos = set()
    combo_cache: Dict[Tuple, Tuple] = {}
    ev_family_cache: Dict[Tuple[str, ...], str] = {}
    contract_cache: Dict[Tuple[Tuple[str, ...], str], str] = {}
    def combo_key(meta: dict, axes: Optional[Tuple[str, ...]] = None) -> Tuple:
        axis = axes or coverage_axes
        key = tuple(meta.get(a, "_") for a in axis)
        cached = combo_cache.get(key)
        if cached is None:
            combo_cache[key] = key
            return key
        return cached

    tries, max_tries = 0, max(n * 80, 2000)
    want_axes = list(coverage_axes)
    while len(out) < n and tries < max_tries:
        tries += 1
        # 先选欠额最大的组合
        pick, deficit = _choose_combo_rr(by_combo, want_axes, min_per_combo, rnd)

        pin_dyn = {}
        if pick:
            pin_dyn = {a: v for a, v in zip(want_axes, pick) if a in ("strategy","channel","carrier","delivery")}
            # 让 ev_bucket 也成为硬约束
            if "ev_bucket" in want_axes:
                idx = want_axes.index("ev_bucket")
                pin_dyn["ev_bucket"] = pick[idx]
                # Anchor-free status can be forced per combo
                if "anchor_free" in want_axes:
                    idx = want_axes.index("anchor_free")
                    pin_dyn["anchor_free_forced"] = int(pick[idx])
        

        pin_payload = {**(pin or {}), **pin_dyn}
        target_family = None
        if pick and "ev_family" in want_axes:
            try:
                target_family = pick[want_axes.index("ev_family")]
            except Exception:
                target_family = None
        max_family_attempts = 6 if target_family else 1
        spec = None
        ev_bucket = None
        ev_key = ()
        fam_sig = None
        for _ in range(max_family_attempts):
            spec = sample_spec(seed=rnd.randint(0, 1<<30), pin=pin_payload)
            ev_bucket = bucket_hash(spec.evidence)
            if "ev_bucket" in pin_dyn and pin_dyn["ev_bucket"] != ev_bucket:
                continue
            ev_key = tuple(sorted(spec.evidence or []))
            fam_sig = ev_family_cache.get(ev_key)
            if fam_sig is None:
                fam_sig = "|".join(sorted(evidence_families(spec.evidence)))
                ev_family_cache[ev_key] = fam_sig
            if target_family and fam_sig != target_family:
                continue
            break
        else:
            continue
        if "ev_family" in pin_dyn and fam_sig is not None and pin_dyn["ev_family"] != fam_sig:
            continue
        alias_p_cn = float((pin or {}).get("alias_p_cn", 0.70))
        spec._alias_p_cn = alias_p_cn
        anchors = Anchors(rnd)
        spec._rnd = rnd
        structural_ratio = _resolve_structural_ratio(pin_payload)
        structural = rnd.random() < structural_ratio
        _struct_evs = {"result_slot","tool","role","path","upstream"}
        any_struct_ev = any(e in _struct_evs for e in (spec.evidence or []))
        base_p = float((pin or {}).get("anchor_free_p", 0.85))
        p_if_struct = float((pin or {}).get("anchor_free_p_if_struct", min(0.60, base_p)))
        eff_p = (p_if_struct if any_struct_ev else base_p)
        forced = (pin or {}).get("anchor_free_forced", None)
        if forced in (0,1,"0","1"):
            anchor_free = bool(int(forced))
        else:
            anchor_free = (rnd.random() < eff_p)
        if spec.strategy.startswith("openpi_"):
            anchor_free = True
            spec.constraints["anchor_free_soft_min_hits"] = max(1, int(spec.constraints.get("anchor_free_soft_min_hits", 3)))        
        soft_min = int((pin or {}).get("anchor_free_soft_min_hits", 3))
        spec.constraints = dict(getattr(spec, "constraints", {}), anchor_free=bool(anchor_free), anchor_free_soft_min_hits=soft_min)
        renderer = CARRIER_RENDER.get(spec.carrier)
        strategy_key = str(getattr(spec, "strategy", "format_anchor") or "format_anchor")
        contract_type = contract_cache.get((ev_key, strategy_key))
        if contract_type is None:
            contract_type = _infer_contract_type_cached(ev_key, strategy_key)
            contract_cache[(ev_key, strategy_key)] = contract_type
        meta_tmp = {
            "strategy": spec.strategy,
            "channel":  spec.channel,
            "carrier":  spec.carrier,
            "delivery": spec.delivery,
            "anchor_free": int(anchor_free),
            "ev_family": fam_sig,
            "ev_bucket": ev_bucket,
            "appearance": ("openpi" if str(spec.strategy).startswith('openpi_') else 'normal'),
            "contract_type": contract_type,
        }
        key_tmp_axes = tuple(a for a in coverage_axes if a in meta_tmp)
        key_tmp = combo_key(meta_tmp, key_tmp_axes)
        #—— 网格可达性短路 —— 
        if key_tmp in blocked_combos:
            fail_by_combo[key_tmp]["blocked"] += 1
            attempts_per_combo[key_tmp] += 1
            if attempts_per_combo[key_tmp] >= block_fail_threshold and success_per_combo.get(key_tmp, 0) == 0:
                blocked_combos.add(key_tmp)
            continue
        if not renderer:
            # 稳健处理：记录缺口，回退重采，不做随机替代以避免分布漂移
            fail_by_combo[key_tmp]["missing_renderer"] += 1
            attempts_per_combo[key_tmp] += 1
            if attempts_per_combo[key_tmp] >= block_fail_threshold and success_per_combo.get(key_tmp, 0) == 0:
                blocked_combos.add(key_tmp)
            continue
        core = renderer(spec, anchors, structural=structural, anchor_free=anchor_free)
        # 先落地交付形态
        txt = apply_delivery(core, spec)
        # 风格适配：仅在自然语载体上做语气扰动，避免破坏协议结构
        if spec.carrier in STYLE_ADAPTER_CARRIERS:
            try:
                txt = apply_style(txt, spec, rnd)
            except Exception:
                pass
        # 轻度 Unicode 扰动：零宽/同形/双向（默认低概率）
        def _obfuscate(s: str, rng) -> str:
            s = re.sub(r"([。；，、：])", lambda m: m.group(1)+"\u200b", s)  # 零宽
            if rng.random() < 0.3:
                s = "\u202d" + s + "\u202c"  # LRO/PDF
            return s
        if rnd.random() < float((pin or {}).get("obfuscation_p", 0.15)):
            txt = _obfuscate(txt, rnd)        
        # 记录是否含锚点（用于审计）；anchor_free 时再去锚
        _had_anchors = bool(APPEND_RE.search(txt) or PATH_RE.search(txt) or UP_RE.search(txt)
                            or ROLE_RE.search(txt) or TOOLCALL_RE.search(txt) or ("{{RESULT_MAIN}}" in txt))
        if anchor_free:
            txt = strip_anchors(txt)
        _applied_sinicize = False
        _applied_struct_cn = False
        _applied_alias = False
        attempts_by_carrier[spec.carrier] += 1
        ok, _fails = adjudicate_invariants(txt, spec)
        # —— 无锚且软证据不足：最小注入软提示后重判 —— 
        if not ok and bool(getattr(spec, "constraints", {}).get("anchor_free", False)):
            if any(f.startswith("soft_") for f in _fails):
                txt2 = inject_soft_hints(
                    txt,
                    need = int(getattr(spec, "constraints", {}).get("anchor_free_soft_min_hits", 3)),
                    must_include_contract = False,
                    rnd = rnd
                )
                ok2, _fails2 = adjudicate_invariants(txt2, spec)
                if ok2:
                    txt, ok, _fails = txt2, ok2, _fails2
                else:
                    for r in _fails2: fail_by_combo[key_tmp][r] += 1
                    continue

        if not ok:
            # 只因中文占比失败：先语义释义化 → 再字段别名 → 最后低概率用结构模板
            if set(_fails) <= {"cjk_share_below_threshold", "fix:cjk_alias"}:
                can_sinicize = spec.carrier in STYLE_ADAPTER_CARRIERS
                can_alias = spec.carrier in FIELD_ALIAS_CARRIERS
                alias_ctx_txt = txt
                alias_ctx_fails = _fails
                if can_sinicize:
                    txt2 = sinicize_surface(txt)
                    ok2, _fails2 = adjudicate_invariants(txt2, spec)
                    if ok2:
                        txt, ok, _fails = txt2, ok2, _fails2
                        _applied_sinicize = True
                        sinicize_hits_by_carrier[spec.carrier] += 1
                    else:
                        alias_ctx_txt = txt2
                        alias_ctx_fails = _fails2
                if not ok and can_alias:
                    alias_p_cn = float((pin or {}).get("alias_p_cn", 0.70))
                    txt3 = randomize_field_aliases(alias_ctx_txt, p_cn=alias_p_cn, rng=rnd)
                    ok3, _fails3 = adjudicate_invariants(txt3, spec)
                    if ok3:
                        txt, ok, _fails = txt3, ok3, _fails3
                        _applied_alias = True
                        alias_hits_by_carrier[spec.carrier] += 1
                    else:
                        alias_ctx_txt = txt3
                        alias_ctx_fails = _fails3
                if not ok:
                    # 3) 仅在很低概率下用 structural=True 的中文模板救火，降低“外壳伪迹”
                    rescue_p = float((pin or {}).get("struct_cn_rescue_p", 0.10))
                    fails_for_reject = alias_ctx_fails
                    if rnd.random() < rescue_p:
                        try:
                            core2 = renderer(spec, anchors, structural=True)
                            txt4 = apply_delivery(core2, spec)
                            ok4, _fails4 = adjudicate_invariants(txt4, spec)
                        except Exception:
                            ok4, _fails4, txt4 = False, alias_ctx_fails, alias_ctx_txt
                        if ok4:
                            txt, ok, _fails = txt4, ok4, _fails4
                            _applied_struct_cn = True
                            struct_cn_hits_by_carrier[spec.carrier] += 1
                        else:
                            fails_for_reject = _fails4
                    if not ok:
                        if any(r == "cjk_share_below_threshold" for r in fails_for_reject):
                            reject_cjk_by_carrier[spec.carrier] += 1
                        for r in fails_for_reject:
                            fail_by_combo[key_tmp][r] += 1
                        attempts_per_combo[key_tmp] += 1
                        if attempts_per_combo[key_tmp] >= block_fail_threshold and success_per_combo.get(key_tmp, 0) == 0:
                            blocked_combos.add(key_tmp)
                        continue
        if not _applied_alias and spec.carrier in FIELD_ALIAS_CARRIERS:
            try:
                base_p = float((pin or {}).get("alias_p_cn", 0.70))
                alias_p_cn = 0.40 if _applied_struct_cn else base_p
                txt_after_alias = randomize_field_aliases(txt, p_cn=alias_p_cn, rng=rnd)
                ok_alias, _fails_alias = adjudicate_invariants(txt_after_alias, spec)
                if ok_alias:
                    txt = txt_after_alias
                    _applied_alias = True
                    alias_hits_by_carrier[spec.carrier] += 1
                # 不通过则回退，不做强行替换
            except Exception:
                pass
        # --- 机制级真值摘要（在去锚后的最终文本上评估） ---
        mech = mechanism_truth_record(strip_anchors(txt), spec)
        soft_hits_local = int(mech.get("soft_hits", 0))
        soft_bucket = ('0' if soft_hits_local==0 else ('1' if soft_hits_local==1 else '2+')) if bool(getattr(spec,'constraints',{}).get('anchor_free', False)) else 'n/a'
        meta = {
            "strategy": spec.strategy,
            "channel":  spec.channel,
            "carrier":  spec.carrier,
            "delivery": spec.delivery,
            "evidence": sorted(spec.evidence or []),
            "contains_anchors": int(_had_anchors),
            "rewrite": {"sinicize": int(_applied_sinicize), "struct_cn": int(_applied_struct_cn), "alias": int(_applied_alias)},
            "anchor_free": int(anchor_free),
            "ev_family": fam_sig,
            "ev_bucket": ev_bucket,
            "alias_p_cn": float(getattr(spec, "_alias_p_cn", float((pin or {}).get("alias_p_cn", 0.70)))),  # 审计用
            "contract_type": contract_type,
            "appearance": ("openpi" if str(spec.strategy).startswith('openpi_') else 'normal'),
            # Style axes for audit
            "speech_family": getattr(spec, "speech_family", "formal"),
            "region": getattr(spec, "region", "cn_mainland"),
            "register": getattr(spec, "register", "regulatory"),
            "industry": getattr(spec, "industry", "it_ops"),
            "persona": getattr(spec, "persona", "qa_reviewer"),
            "soft_bucket": soft_bucket,
            "soft_evidence": {
                "kinds": mech.get("soft_kinds", []),
                "count": int(mech.get("soft_hits", 0)),
                "min_required": int(mech.get("soft_min", soft_min)),
                "used_soft_for": mech.get("used_soft_for", []),
                "by_evidence": mech.get("by_evidence", {}),
            },
        }
        preview_txt = strip_anchors(txt) if not anchor_free else txt
        mech_flags, mech_diag, mech_events = _prepare_mechanism_payload(preview_txt, spec, mechanism_signal_fn)
        mechanisms = sorted([k for k, v in mech_flags.items() if v])
        mech_tuple = tuple(mechanisms) if mechanisms else ("_none",)
        trigger_path = _infer_trigger_path(spec.delivery, spec.carrier)
        mech_combo_key = (mech_tuple, trigger_path)
        mech_label = "+".join(mechanisms) if mechanisms else "_none"
        meta.update({
            "mechanism_flags": mech_flags,
            "mechanisms": mechanisms,
            "mechanism_combo": mech_label,
            "trigger_path": trigger_path,
            "mechanism_path_key": "{}@{}".format(mech_label, trigger_path),
        })
        if mech_diag:
            meta["mechanism_signals"] = mech_diag
        if mech_events:
            meta["mechanism_events"] = mech_events
        mechanism_signature = _mechanism_signature_tokens(mech_events, mech_diag, mechanisms, spec.evidence or [])
        meta["mechanism_signature_size"] = len(mechanism_signature)
        def _note_mech_failure(reason_key: str) -> None:
            mech_fail[mech_combo_key][reason_key] += 1
            mech_attempts[mech_combo_key] += 1
            if mech_attempts[mech_combo_key] >= block_fail_threshold and mech_success.get(mech_combo_key, 0) == 0:
                mech_blocked.add(mech_combo_key)

        key = combo_key(meta)
        if mech_combo_key in mech_blocked:
            _note_mech_failure("blocked")
            fail_by_combo[key]["mechanism_blocked"] += 1
            attempts_per_combo[key_tmp] += 1
            if attempts_per_combo[key_tmp] >= block_fail_threshold and success_per_combo.get(key_tmp, 0) == 0:
                blocked_combos.add(key_tmp)
            continue
        sig = semantic_signature(spec, txt)
        mech_key = (meta.get("contract_type"), meta.get("ev_bucket"), meta.get("channel"), meta.get("delivery"))

        def _register_dup(reason_key: str, mech_reason: Optional[str] = None) -> None:
            audit_cb(reason_key, {"combo": str(key)})
            fail_by_combo[key][reason_key] += 1
            attempts_per_combo[key_tmp] += 1
            if attempts_per_combo[key_tmp] >= block_fail_threshold and success_per_combo.get(key_tmp, 0) == 0:
                blocked_combos.add(key_tmp)
            _note_mech_failure(mech_reason or reason_key)

        # --- mechanism saturation and long-tail activation ---
        mech_cap = int((pin or {}).get("mech_cap_per_bucket", 18))
        long_tail_period = max(1, int((pin or {}).get("long_tail_period", 200)))
        long_tail_boost = float((pin or {}).get("long_tail_tail_mix_p", 0.25))
        if by_mech_combo.get(mech_combo_key, 0) >= mech_cap:
            _register_dup("mechanism_bucket_saturated", mech_reason="mechanism_bucket_saturated")
            continue
        if tries % long_tail_period == 0:
            pin_dyn.setdefault("tail_mix_p", long_tail_boost)

        if sig in seen_signatures and by_mech_combo[mech_combo_key] >= min_per_combo:
            audit_cb("dup_signature", {"combo": str(key), "sig": sig})
            fail_by_combo[key]["dup_signature"] += 1
            attempts_per_combo[key_tmp] += 1
            if attempts_per_combo[key_tmp] >= block_fail_threshold and success_per_combo.get(key_tmp, 0) == 0:
                blocked_combos.add(key_tmp)
            continue

        style_key = (
            meta.get("contract_type"),
            meta.get("ev_family"),
            meta.get("speech_family", getattr(spec, "speech_family", "formal")),
        )
        primary_record: DedupeRecord = dedupe_template.prepare(txt)
        dedupe_pending: List[Tuple[Deduper, DedupeRecord]] = []

        def _check_with(deduper_obj: Deduper, reason_map: Dict[str, str]) -> bool:
            ok, reason = deduper_obj.check_record(primary_record)
            if not ok:
                reason_name = reason_map.get(reason or "simhash", next(iter(reason_map.values())))
                _register_dup(reason_name)
                return False
            dedupe_pending.append((deduper_obj, primary_record))
            return True

        existing_mech_sigs = mech_sig_cache.get(mech_combo_key, [])
        if mechanism_jacc_thr < 1.0 and mechanism_signature:
            if _mechanism_near_dup(existing_mech_sigs, mechanism_signature, mechanism_jacc_thr):
                _register_dup("mechanism_near_dup", mech_reason="mechanism_near_dup")
                continue

        combo_ded = dedupe_combo.get(key)
        if combo_ded is None:
            combo_ded = _make_local_deduper()
            dedupe_combo[key] = combo_ded
        if not _check_with(combo_ded, reason_map_local):
            continue

        mech_ded = dedupe_mech.get(mech_key)
        if mech_ded is None:
            mech_ded = _make_local_deduper()
            dedupe_mech[mech_key] = mech_ded
        if not _check_with(mech_ded, reason_map_local):
            continue

        style_ded = dedupe_style.get(style_key)
        if style_ded is None:
            style_ded = _make_local_deduper()
            dedupe_style[style_key] = style_ded
        if not _check_with(style_ded, reason_map_local):
            continue

        if not _check_with(dedupe_global, reason_map_global):
            continue

        # 优先填满每个组合的最低覆盖
        need_mech = (min_per_combo > 0 and by_mech_combo[mech_combo_key] < min_per_combo)
        can_accept = need_mech or (min_per_combo == 0)
        # 机制级去重上限（防止不同壳但同机制簇的堆叠）
        max_per_mech = int((pin or {}).get("max_per_mech", 999999))
        # 机制优先：若当前机制已达上限且无机制缺口，则暂缓
        if not need_mech and by_mech.get(mech_key, 0) >= max_per_mech:
            can_accept = False
            fail_by_combo[key]["mech_quota_hold"] += 1

        # 如果还没填满所有组合的最低覆盖，不要让少数组合过度占用配额
        # 在任何格式轴放行前，优先满足机制维度的 min_per_combo
        if min_per_combo > 0 and not need_mech:
            if any(v < min_per_combo for v in by_combo.values()):
                can_accept = False
                fail_by_combo[key]["quota_hold"] += 1
        if not can_accept:
            attempts_per_combo[key_tmp] += 1
            if attempts_per_combo[key_tmp] >= block_fail_threshold and success_per_combo.get(key_tmp, 0) == 0:
                blocked_combos.add(key_tmp)
            continue
        final_txt = txt  # Always emit the adjudicated text so anchors survive when required
        seen_signatures.add(sig)
        success_per_combo[key] += 1
        success_per_combo[key_tmp] += 1
        mech_success[mech_combo_key] += 1
        by_mech_combo[mech_combo_key] += 1
        by_mech[mech_key] += 1
        if mechanism_signature:
            mech_sig_cache[mech_combo_key].append(frozenset(mechanism_signature))
        out.append({"text": final_txt, "label": 1, "meta": meta})
        soft_counts_hist[len(mech.get("soft_kinds", []))] += 1
        for k in mech.get("soft_kinds", []): soft_kinds_totals[k] += 1
        for ev in mech.get("used_soft_for", []): soft_used_by_ev[ev] += 1

        for deduper_obj, record_obj in dedupe_pending:
            deduper_obj.add_record(record_obj)

        combo_texts[key].append(final_txt)
        combo_ngrams[key].append(frozenset(primary_record.shingles))
        signature_cache[final_txt] = _snapshot_record(primary_record)
        mech_cov = coverage.setdefault("mechanism_matrix", {})
        mech_key_str = meta.get("mechanism_path_key", _format_mech_combo_key(mech_combo_key))
        mech_cov[mech_key_str] = mech_cov.get(mech_key_str, 0) + 1

        by_combo[key] += 1
        for fam in (evidence_families(spec.evidence) or ["_none"]):
            by_cfam[(spec.carrier, fam)] += 1
        # style & softbucket counts (for audit)
        style_counts = coverage.get('by_style')
        if not isinstance(style_counts, dict):
            style_counts = {}
            coverage['by_style'] = style_counts
        key_style = meta.get('speech_family', 'formal')
        style_counts[key_style] = style_counts.get(key_style, 0) + 1


    total_attempts = max(1, sum(attempts_by_carrier.values()))
    total_near_simhash = sum(v.get("semantic_near_dup", 0) for v in fail_by_combo.values())
    total_near_ngram  = sum(v.get("semantic_ngram_near_dup", 0) for v in fail_by_combo.values())
    total_near_vec    = sum(v.get("semantic_vec_near_dup", 0) for v in fail_by_combo.values())

    # Renderer missing rate (audit)
    missing_total = sum(v.get("missing_renderer", 0) for v in fail_by_combo.values())
    report = {
        "num": len(out),
        "coverage_signatures": len(seen_signatures),
        "by_strategy": count_by(out, lambda x: x["meta"]["strategy"]),
        "by_channel":  count_by(out, lambda x: x["meta"]["channel"]),
        "by_carrier":  count_by(out, lambda x: x["meta"]["carrier"]),
        "by_delivery": count_by(out, lambda x: x["meta"]["delivery"]),
        "by_combo_axes": list(coverage_axes),
        "by_combo_counts": {str(k): v for k, v in by_combo.items()},
        "by_cfam": {f"{c}|{fam}": n for (c, fam), n in by_cfam.items()},
         "by_evidence": count_by(out, lambda x: tuple(x["meta"].get("evidence", []))),
        "by_evidence_combo": count_by(out, lambda x: "|".join(x["meta"].get("evidence", []))),
        "by_mechanism": count_by(out, lambda x: "|".join([
            x["meta"].get("contract_type","_"), x["meta"].get("ev_bucket","_"),
            x["meta"].get("channel","_"), x["meta"].get("delivery","_")
        ])),
        "by_mechanism_combo": {_format_mech_combo_key(k): v for k, v in by_mech_combo.items()},
        "mechanism_attempts": {_format_mech_combo_key(k): mech_attempts.get(k, 0) for k in mech_attempts},
        "mechanism_success": {_format_mech_combo_key(k): mech_success.get(k, 0) for k in mech_success},
        "mechanism_failures": {_format_mech_combo_key(k): dict(v) for k, v in mech_fail.items()},
        "by_speech_family": count_by(out, lambda x: x["meta"].get("speech_family", "formal")),
        "by_region": count_by(out, lambda x: x["meta"].get("region", "cn_mainland")),
        "by_register": count_by(out, lambda x: x["meta"].get("register", "regulatory")),
        "by_industry": count_by(out, lambda x: x["meta"].get("industry", "it_ops")),
        "by_persona": count_by(out, lambda x: x["meta"].get("persona", "qa_reviewer")),
        "fail_by_combo": {str(k): dict(v) for k, v in fail_by_combo.items()},
        "fail_totals":   {r: sum(v.get(r,0) for v in fail_by_combo.values())
                          for r in {rr for v in fail_by_combo.values() for rr in v.keys()}},
        "reject_cjk_by_carrier": dict(reject_cjk_by_carrier),
        "attempts_by_carrier":   dict(attempts_by_carrier),
        "rewrite_hits_by_carrier": {
            "sinicize": dict(sinicize_hits_by_carrier),
            "struct_cn": dict(struct_cn_hits_by_carrier),
            "alias": dict(alias_hits_by_carrier),
        },
        "near_dup_rates": {
            "simhash": total_near_simhash / total_attempts,
            "ngram":   total_near_ngram  / total_attempts,
            "vec":     total_near_vec    / total_attempts,
        },
        "soft_evidence_hist": dict(soft_counts_hist),
        "soft_evidence_kinds_totals": dict(soft_kinds_totals),
        "soft_evidence_used_by_evidence": dict(soft_used_by_ev),
        "blocked_combos": sorted(list(blocked_combos)),
        "attempts_per_combo": {str(k): v for k, v in attempts_per_combo.items()},
        "success_per_combo":  {str(k): v for k, v in success_per_combo.items()},
        "renderer_missing_count": int(missing_total),
        "renderer_missing_rate": (missing_total / max(1, tries)),
        "dedupe_cache_stats": {
            "combo_groups": len(combo_texts),
            "signature_cache_entries": len(signature_cache),
        },
        "mechanism_matrix": coverage.get("mechanism_matrix", {}),
    }
    return out, report

def count_by(items, key_fn):
    d = {}
    for item in items:
        key = key_fn(item)
        d[key] = d.get(key, 0) + 1
    return d
