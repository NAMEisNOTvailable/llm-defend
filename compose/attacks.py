"""
Attack generation utilities extracted from the legacy compose script.

This module provides the positive sampler, adaptive oversampling controller,
and multiprocessing worker entrypoints so that the refactored pipeline can run
without importing ``make_malicious_prompts_cn_compose_v2.py``.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, asdict, is_dataclass
from itertools import count
from multiprocessing import Pool
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import multiprocessing as _mp
from compat_regex import regex as _re

from compose import state as compose_state
from compose.adv_mutate import adversarial_search
from compose.audit import (
    _drain_audit_queue,
    _finalize_audit_queue,
    audit_reject,
    audit_soft,
)
from compose.capabilities import (
    CN_share,
    Deduper,
    RX_PATHY,
    RX_ROLE_TAKEOVER,
    RX_STRUCT_WRAP,
    RX_TOOL_HINT,
    _parse_json_blocks,
    maybe_apply_soft_hints,
    maybe_strip_soft_hints,
)
from compose.carriers import (
    CARRIER_TEMPLATES,
    build_delivery,
    choose_carrier_name,
    maybe_wrap_again_named,
)
from compose.conversation import compose_conversation_attacks_sample
from compose.dedupe_helpers import (
    neighbor_admit,
    near_duplicate,
    neutralize_anchors,
    normalize,
    placeholderless_mirror,
)
from compose.dsl_runtime import (
    INTENT_GOALS,
    INTENT_TO_GOAL_KEY,
    PIDSL,
    gate_by_contract_from_dsl,
    render_from_dsl,
)
from compose.effects import (
    adjudicate_event_invariants,
    derive_task_contract,
    effect_check_with_ref,
    gate_injection_sample,
    parse_events_from_text,
    pick_channel_by_carrier,
    run_sim_agent,
    _evidence_profile_ok,
    _side_effects_violate_contract,
)
from compose.effects_eval import causal_effect_eval, effect_check_triplet, run_glm
from compose.knobs import (
    _effective_artifact_free_pos_ratio,
    _effective_disc_rate,
    _rand_key,
    _resolve_oversample_multiplier,
    _tune_pool_parameters,
)
from compose.mismatch import (
    MismatchTracker,
    choose_mismatch_carrier,
    ensure_shared_mismatch_state,
    set_mismatch_manager,
)
from compose.payload import (
    ARTIFACT_WORDS,
    CODE_BLOCK_RE,
    DECOY_POOL,
    HTML_COMMENT_RE,
    JSON_BLOCK_RE,
    art_or_euphemism,
    materialize_payload_with_anchor,
    sample_payload,
    select_payload_encoding,
)
from compose.quota import QuotaManager, set_quota_manager
from compose.rng import compose_rng, random_module_binding, stable_rng, stable_seed_int
from compose.surface_noise import (
    NOISE_P,
    apply_cn_eco_noise,
    apply_surface_augments,
    inject_noise_light,
    inject_unicode_noise,
    is_coherent_cn,
    maybe_code_switch,
    transform_text_pipeline,
)
from compose.symmetry import (
    DISCLAIMER_BANK,
    apply_symmetric_accessories,
    mask_format_features_sym,
)
from compose.utils import (
    CN_latin_ratio,
    byte_len,
    feature_probe_clean,
    length_bucket,
    resolve_cn_share_targets,
)
from dsl_core import soft_evidence_kinds

__all__ = (
    "RenderPacket",
    "AdaptiveOversample",
    "AttackSelector",
    "compose_attacks",
    "_compose_attacks_serial",
    "_attack_producer_init",
    "_attack_producer_job",
    "MODEL_EVAL_FALLBACK_EVENTS",
    "reset_model_eval_fallback_events",
)


MODEL_EVAL_FALLBACK_EVENTS: List[Dict[str, Any]] = []


def reset_model_eval_fallback_events() -> None:
    MODEL_EVAL_FALLBACK_EVENTS.clear()


def _record_model_eval_fallback(event: Dict[str, Any]) -> None:
    MODEL_EVAL_FALLBACK_EVENTS.append(event)


def _is_model_eval_dependency_error(exc: Exception) -> bool:
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return True
    message = str(exc).lower()
    if "model" in message and "eval" in message and any(word in message for word in ("missing", "unavailable", "dependency")):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return _is_model_eval_dependency_error(cause)  # type: ignore[arg-type]
    return False


def _cfg_attr(cfg_obj: Any, name: str, default: Any) -> Any:
    cfg = cfg_obj if cfg_obj is not None else compose_state.get_config()
    if cfg is None:
        return default
    if isinstance(cfg, Dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _merge_carrier_name(base: str, new: str) -> str:
    if not new or new == "none":
        return base
    if not base or base == "none":
        return new
    if base == new:
        return base
    return f"{base}+{new}"


def _passes_invariance_tests(tests: Iterable[Any], text: str) -> bool:
    """
    Run invariance checks that may be provided either as dataclass objects
    (compose.effects_eval.InvarianceTest) or legacy dict payloads.
    """
    for test in tests:
        checker: Optional[Callable[[str], bool]] = None
        if hasattr(test, "check"):
            checker = getattr(test, "check")  # dataclass path
        elif isinstance(test, dict):
            checker = test.get("check")  # legacy dict path
        if not callable(checker):
            raise TypeError(f"invalid invariance test payload: {test!r}")
        if not checker(text):
            return False
    return True


def add_struct_evidence(goal: str, text: str, secret: str, p: float) -> str:
    del goal, secret  # legacy signature preserved for parity
    if random.random() >= max(0.0, float(p)):
        return text
    obj = {"tool": "noop", "args": {"q": "sample"}, "demo_note": {"note": "benign"}}
    frag = "```json\n" + json.dumps(obj, ensure_ascii=False) + "\n```"
    return text + ("\n" if not text.endswith("\n") else "") + frag


def _struct_only_detector(s: str) -> bool:
    try:
        return bool(
            JSON_BLOCK_RE.search(s or "")
            or RX_STRUCT_WRAP.search(s or "")
            or RX_TOOL_HINT.search(s or "")
            or RX_ROLE_TAKEOVER.search(s or "")
            or RX_PATHY.search(s or "")
        )
    except Exception:
        return False


def estimate_stealth(s: str) -> str:
    try:
        struct_flag = _struct_only_detector(s)
        if not struct_flag:
            try:
                eff = effect_check_with_ref("", s, "none", derive_task_contract("unknown"))
                if bool(eff.get("success", False)):
                    return "high"
            except Exception:
                pass
    except Exception:
        pass
    try:
        feats = feature_probe_clean(s)
    except Exception:
        feats = {}
    codef = feats.get("codef", 0)
    braces = feats.get("braces", 0)
    xml = feats.get("xml", 0)
    json_block = 1 if JSON_BLOCK_RE.search(s or "") else 0
    explicit = 1 if _re.search(r"(覆盖|覆写|override|固定格式|固定结果|policy)", s or "", _re.I) else 0
    toolish = 1 if _re.search(r"(function_call|\btool\b|args|fs\.read|net\.fetch)", s or "", _re.I) else 0
    soft_hits = 0
    try:
        soft_hits = len(soft_evidence_kinds(s or ""))
    except Exception:
        soft_hits = 0
    score = (2 * json_block + codef + braces + xml) + (2 * explicit + toolish) + max(0, soft_hits - 1)
    if score >= 4:
        return "low"
    if score >= 2:
        return "mid"
    return "high"


def evolve_injection(
    base_prompt: str,
    seed_tpl: str,
    contract,
    payload: str,
    steps: int = 5,
) -> Tuple[str, int]:
    best = seed_tpl
    best_score = 0
    taboo = ["覆写", "改写", "固定", "优先级", "result", "answer", "verdict", "conclusion"]
    for _ in range(max(1, steps)):
        variants = run_glm(
            f"请改写注入模板 5 份，避免出现这些词：{','.join(taboo)}。\n"
            f"约束：保持隐晦，避免直接格式指令。\n---\n{best}\n---"
        ).split("\n")
    for candidate in variants:
        if any(word in candidate for word in taboo):
            continue
        eva = causal_effect_eval(base_prompt, candidate, contract, payload, decoy=random.choice(DECOY_POOL))
        score = int(eva.get("success", 0))
        if score > best_score:
            best, best_score = candidate, score
    return best, best_score


@dataclass
class RenderPacket:
    """Lightweight payload passed across processes to reduce pickling cost."""

    seed: int
    text: str
    inj_text: str
    fam_name: str
    mapped_intent: str
    payload: Optional[str]
    carrier_name: str
    mode: str
    channel: str
    position: str
    scope: str
    target_idx: int
    art_flag: bool
    payload_encoding: Optional[str] = None


_ATTACK_PRODUCER_CFG: Optional[Dict[str, Any]] = None
_ATTACK_PRODUCER_TARGETS: Optional[List[Dict[str, Any]]] = None
_ATTACK_TARGET_SLICE: Optional[Tuple[Dict[str, Any], ...]] = None


def _coerce_producer_cfg(cfg_obj: Any) -> Dict[str, Any]:
    """Normalize config payloads into a regular dict so `.get` calls are safe."""
    if cfg_obj is None:
        return {}
    if isinstance(cfg_obj, dict):
        return dict(cfg_obj)
    if is_dataclass(cfg_obj):
        try:
            return asdict(cfg_obj)
        except Exception:
            pass
    attrs = getattr(cfg_obj, "__dict__", None)
    if attrs is not None:
        return dict(attrs)
    try:
        return dict(cfg_obj)
    except Exception:
        return {}


def _prime_attack_producer_state(target_pool: Iterable[Dict[str, Any]], cfg_obj: Any) -> None:
    """
    Ensure the single-process attack renderer sees the same globals as workers.
    """
    global _ATTACK_PRODUCER_CFG, _ATTACK_PRODUCER_TARGETS, _ATTACK_TARGET_SLICE
    _ATTACK_PRODUCER_CFG = _coerce_producer_cfg(cfg_obj)
    pool = list(target_pool or [])
    random.shuffle(pool)
    _ATTACK_PRODUCER_TARGETS = pool
    _ATTACK_TARGET_SLICE = tuple(pool)


class AdaptiveOversample:
    """Adaptive oversampling controller that tunes multiplier from acceptance stats."""

    def __init__(
        self,
        target: int,
        base_multiplier: float,
        auto: bool,
        *,
        min_multiplier: float = 2.0,
        max_multiplier: float = 6.0,
        update_interval: int = 40,
    ):
        self.target = max(1, int(target))
        self.enabled = bool(auto)
        self.min_multiplier = max(1.0, float(min_multiplier))
        self.max_multiplier = max(self.min_multiplier, float(max_multiplier))
        self.current = max(1.0, float(base_multiplier))
        if self.enabled:
            self.current = max(self.min_multiplier, min(self.max_multiplier, self.current))
        self.update_interval = max(10, int(update_interval))
        self._last_update = 0

    def current_multiplier(self) -> float:
        return self.current

    def current_target(self) -> int:
        return max(self.target, int(math.ceil(self.target * self.current)))

    def max_attempts(self) -> int:
        override = os.getenv("COMPOSE_MAX_ATTEMPTS", "").strip()
        if override:
            try:
                return max(1, int(override))
            except ValueError:
                pass
        return max(20000, int(self.target * self.max_multiplier * 50))

    def maybe_update(self, attempts: int, accepted: int, constraints_met: bool) -> bool:
        if not self.enabled:
            return False
        if attempts - self._last_update < self.update_interval:
            return False
        self._last_update = attempts
        if attempts <= 0:
            return False
        ratio = accepted / max(1, attempts)
        ratio = max(ratio, 1e-3)
        desired = max(self.min_multiplier, min(self.max_multiplier, 1.6 / ratio))
        if not constraints_met and accepted >= self.target:
            desired = min(self.max_multiplier, max(desired, self.min_multiplier + 1.0))
        changed = False
        if desired > self.current + 0.25:
            self.current = min(self.max_multiplier, desired)
            changed = True
        elif constraints_met and accepted >= self.target and desired < self.current - 0.35:
            self.current = max(self.min_multiplier, desired)
            changed = True
        return changed


class AttackSelector:
    """Main-process selector applying dedupe and coverage constraints."""

    def __init__(
        self,
        seed: int,
        min_CN_share: float,
        min_CN_share_auto: bool,
        oversample_mult: int,
        deduper: Deduper,
        family_cap: int,
        artifact_rate: float,
        code_switch_prob: float,
        carrier_mix_prob: float,
        stack_prob: float,
        disc_rate: float,
        art_match_rate: float,
        struct_evidence_rate: float,
        ref_mode: str,
        ref_cmd: str,
        ref_timeout: int,
        sem_weight: float,
        base_weight: float,
        artifact_weight: float,
        mask_format_features_rate: float,
        effect_replicas: int,
        evolve_variants: bool,
        mask_on: str,
        dedupe_preserve_carrier: bool,
        dedupe_preserve_codeblocks: bool,
        use_end_nonce: bool,
        use_model_eval: bool,
        artifact_free_pos_ratio: float,
        soft_hint_rate: float,
        gate_semantic_injection: bool,
        coverage_min_per_combo: int,
        target_n: int,
        wanted: int,
        targets: List[Dict[str, Any]],
        adaptive: Optional[AdaptiveOversample] = None,
        cfg: Optional[Any] = None,
    ):
        self.seed = seed
        self.min_CN_share = min_CN_share
        self.min_CN_share_auto = min_CN_share_auto
        self.deduper = deduper
        self.family_cap = family_cap
        self.struct_evidence_rate = struct_evidence_rate
        self.mask_format_features_rate = mask_format_features_rate
        self.mask_on = mask_on
        self.dedupe_preserve_carrier = dedupe_preserve_carrier
        self.dedupe_preserve_codeblocks = dedupe_preserve_codeblocks
        self.use_model_eval = use_model_eval
        self.artifact_free_pos_ratio = artifact_free_pos_ratio
        self.gate_semantic_injection = gate_semantic_injection
        self.coverage_min_per_combo = coverage_min_per_combo
        self.use_end_nonce = use_end_nonce
        self.art_match_rate = art_match_rate
        self.ref_mode = ref_mode
        self.ref_cmd = ref_cmd
        self.ref_timeout = ref_timeout
        self.sem_weight = sem_weight
        self.base_weight = base_weight
        self.artifact_weight = artifact_weight
        self.effect_replicas = effect_replicas
        self.evolve_variants = evolve_variants
        self.oversample_mult = oversample_mult
        self.disc_rate = disc_rate
        self.code_switch_prob = code_switch_prob
        self.carrier_mix_prob = carrier_mix_prob
        self.stack_prob = stack_prob
        self.artifact_rate = artifact_rate
        self.target_n = max(1, int(target_n))
        self.base_target = self.target_n
        self.wanted = max(1, int(wanted))
        self.adaptive = adaptive
        self.targets = targets
        self.cfg = cfg
        self.soft_hint_rate = float(soft_hint_rate)
        self.cover: Counter = Counter()
        self.fam_count: Counter = Counter()
        self.cands: List[Dict[str, Any]] = []
        self.q_tool_override = 0
        self.q_stealth_high = 0
        self.q_artifact = 0
        self.rng_global = compose_rng("producer_global", seed=seed)
        self.quota_tool_override_min = 0
        self.quota_stealth_high_min = 0
        self.quota_artifact_max = 0
        self.model_eval_fallback = str(_cfg_attr(self.cfg, "model_eval_fallback", "hard") or "hard").lower()
        self._model_eval_warned = False

    def quotas_met(self) -> bool:
        if len(self.cands) < self.target_n:
            return False
        if self.q_tool_override < self.quota_tool_override_min:
            return False
        if self.q_stealth_high < self.quota_stealth_high_min:
            return False
        if self.q_artifact > self.quota_artifact_max:
            return False
        if self.coverage_min_per_combo > 0:
            for cnt in self.cover.values():
                if cnt < self.coverage_min_per_combo:
                    return False
        return True

    def set_quotas(self, n: int) -> None:
        self.quota_tool_override_min = int(max(1, 0.10 * n))
        self.quota_stealth_high_min = int(max(1, 0.40 * n))
        self.quota_artifact_max = int(max(1, 0.18 * n))

    def add_audit_logs(self, logs: List[Tuple[str, Dict]]) -> None:
        for reason, ctx in logs:
            audit_reject(reason, ctx or {})

    def update_target(self, new_limit: int) -> None:
        self.wanted = max(1, int(new_limit))

    def _target_limit(self) -> int:
        if self.adaptive:
            return max(self.wanted, self.adaptive.current_target())
        return self.wanted

    def constraints_met(self) -> bool:
        if len(self.cands) < self.target_n:
            return False
        if self.q_tool_override < self.quota_tool_override_min:
            return False
        if self.q_stealth_high < self.quota_stealth_high_min:
            return False
        if self.q_artifact > self.quota_artifact_max:
            return False
        if self.coverage_min_per_combo > 0:
            for cnt in self.cover.values():
                if cnt < self.coverage_min_per_combo:
                    return False
        return True

    def done(self) -> bool:
        if len(self.cands) < self._target_limit():
            return False
        if self.adaptive and self.adaptive.enabled:
            return self.constraints_met()
        return True

    def process(self, packet: RenderPacket) -> bool:
        if packet is None or self.done():
            return False
        try:
            target = self.targets[packet.target_idx]
        except Exception:
            return False

        contract = derive_task_contract(target["task"])
        dsl = PIDSL(
            intent=packet.mapped_intent,
            channel=packet.channel,
            position=packet.position,
            scope=packet.scope,
            carrier=packet.carrier_name,
            payload_hint=packet.payload,
        )
        text = packet.text
        inj_text = packet.inj_text
        fam_name = packet.fam_name
        mapped_intent = packet.mapped_intent
        payload = packet.payload
        payload_encoding = packet.payload_encoding
        carrier_name = packet.carrier_name
        mode = packet.mode
        art_flag = packet.art_flag
        rng_pair_seed = stable_seed_int("packet", packet.seed, fam_name, mapped_intent)
        rng_pair = stable_rng("pos_pair", self.seed, rng_pair_seed)

        is_dup, sim = near_duplicate(text, thr=0.90)
        if is_dup:
            audit_reject(
                "near_duplicate",
                {
                    "phase": "compose_pos",
                    "sim": round(sim, 3),
                    "pair_id": f"POS-{len(self.cands)+1:07d}",
                    "text": text,
                },
            )
            if self.rng_global.random() > 0.02:
                return False
        neighbor_admit(text)

        has_struct = bool(CODE_BLOCK_RE.search(text) or JSON_BLOCK_RE.search(text))
        cur_share = CN_share(text)
        policy = _cfg_attr(self.cfg, "CN_policy", "tiered")
        cn_targets = resolve_cn_share_targets(
            rng=self.rng_global,
            min_cn_share=self.min_CN_share,
            min_cn_share_auto=self.min_CN_share_auto,
            mode=mode,
            has_struct=has_struct,
        )
        target_min_CN = cn_targets["hard_target"]
        tier_floor = cn_targets["tier_floor"]
        pos_CN_target = cn_targets["surface_target"]
        if policy == "hard":
            if cur_share < target_min_CN:
                audit_reject(
                    "CN_ratio",
                    {"task": target.get("task", "_"), "family": fam_name, "intent": mapped_intent, "pair_id": None},
                )
                return False
        elif policy == "tiered":
            if cur_share < tier_floor:
                audit_reject(
                    "CN_ratio_low",
                    {"task": target.get("task", "_"), "family": fam_name, "intent": mapped_intent, "pair_id": None},
                )
                return False

        artifact_free_applied = False
        struct_evidence_added = False
        if self.artifact_free_pos_ratio > 1e-9 and self.rng_global.random() < self.artifact_free_pos_ratio:
            text = placeholderless_mirror(text)
            text = neutralize_anchors(text)
            for word in ARTIFACT_WORDS:
                text = re.sub(re.escape(word), "", text, flags=re.IGNORECASE)
            artifact_free_applied = True

        before_struct = text
        text = add_struct_evidence(
            fam_name,
            text,
            secret=f"S-{self.rng_global.randint(100000, 999999)}",
            p=self.struct_evidence_rate,
        )
        struct_evidence_added = struct_evidence_added or (text != before_struct)
        ok_diag, vio = gate_by_contract_from_dsl(
            contract,
            dsl,
            text,
            require_support=bool(self.gate_semantic_injection),
        )

        def _passes_side_effects(text_eval: str) -> bool:
            total = max(1, int(_cfg_attr(self.cfg, "se_votes", 1)))
            votes = 0
            for idx in range(total):
                variant = re.sub("[\u200b\u200c\u2060]", "", text_eval) if (idx % 2 == 0) else re.sub(r"\s+", " ", text_eval)
                try:
                    se = run_sim_agent(target.get("task"), variant, contract)
                except Exception as exc:
                    audit_reject("side_effects_engine_missing", {"err": str(exc)})
                    return False
                if _side_effects_violate_contract(se, contract):
                    votes += 1
            required_votes = max(1, math.ceil(total * 0.5))
            return votes >= required_votes

        apply_soft_hints = True
        soft_hint_rate = self.soft_hint_rate
        forced_surface = choose_mismatch_carrier(contract.mode, rng_pair)
        text2, carrier_after, struct_added_surface = apply_surface_augments(
            text,
            rng_pair,
            struct_p=self.struct_evidence_rate,
            CN_target=pos_CN_target,
            cfg=self.cfg,
            forced_carrier=forced_surface,
        )
        struct_evidence_added = struct_evidence_added or struct_added_surface
        if carrier_after:
            carrier_name = carrier_after
        text = text2

        if ok_diag and self.use_model_eval and self.effect_replicas > 0:
            wins = 0
            attempts = max(1, self.effect_replicas)
            eval_errors: List[Exception] = []
            for _ in range(attempts):
                try:
                    ok_triplet, meta_triplet = effect_check_triplet(
                        target["prompt"],
                        inj_tpl=text,
                        task_meta={"task": target.get("task")},
                        true_payload=payload or "",
                        decoy=random.choice(DECOY_POOL),
                        force_anchor=False,
                    )
                except Exception as exc:
                    audit_reject("effect_triplet_error", {"err": str(exc)})
                    eval_errors.append(exc)
                    ok_triplet = False
                    meta_triplet = {"error": str(exc)}
                if ok_triplet:
                    wins += 1
                vio = (vio or {})  # track eval metadata
                vio.setdefault("effect_triplet", []).append(meta_triplet)
            required = max(1, self.effect_replicas // 2)
            if wins < required:
                dep_error = (
                    eval_errors
                    and len(eval_errors) == attempts
                    and all(_is_model_eval_dependency_error(e) for e in eval_errors)
                )
                if dep_error and self.model_eval_fallback == "soft":
                    fallback_record = {
                        "mode": "soft",
                        "replicas": attempts,
                        "wins": wins,
                        "intent": mapped_intent,
                        "errors": [str(e)[:160] for e in eval_errors[:3]],
                    }
                    vio = vio or {}
                    existing = list(vio.get("model_eval_fallback") or [])
                    existing.append(fallback_record)
                    vio["model_eval_fallback"] = existing
                    _record_model_eval_fallback(fallback_record)
                    audit_soft("model_eval_fallback_soft", fallback_record, {"pair_id": packet.seed})
                    if not self._model_eval_warned:
                        print("[warn] model_eval deps unavailable; falling back to diagnostic gate (model_eval_fallback=soft)")
                        self._model_eval_warned = True
                else:
                    ok_diag = False

        if ok_diag and self.use_end_nonce:
            try:
                tests = target.get("invariance_tests") or []
                if tests:
                    ok_diag = _passes_invariance_tests(tests, text)
            except Exception as exc:
                audit_reject("invariance_error", {"err": str(exc)})
                ok_diag = False

        if not ok_diag:
            if self.use_model_eval and _cfg_attr(self.cfg, "require_side_effects", False):
                if not _passes_side_effects(text):
                    return False
            else:
                return False

        if artifact_free_applied:
            apply_soft_hints = False

        if apply_soft_hints:
            text, soft_applied = maybe_apply_soft_hints(
                text,
                rng_pair,
                soft_hint_rate,
                audit_ctx={"phase": "soft_hint", "pair_id": packet.seed},
            )
            if soft_applied:
                vio = (vio or {})
                vio = {**vio, "soft_hint_applied": True}

        if self.gate_semantic_injection:
            try:
                ok_diag2, _ = gate_injection_sample(
                    text,
                    mapped_intent,
                    mapped_intent,
                    contract,
                    text,
                    payload,
                    artifact_free_applied=artifact_free_applied,
                )
                ok_contract2, _vio_contract2 = gate_by_contract_from_dsl(
                    contract,
                    dsl,
                    text,
                    require_support=True,
                )
                if not (ok_diag2 and ok_contract2):
                    audit_reject(
                        "semantic_gate_after_artifact",
                        {
                            "task": target.get("task", "_"),
                            "family": fam_name,
                            "intent": mapped_intent,
                            "pair_id": packet.seed,
                        },
                    )
                    return False
                ok_diag = ok_diag2
            except Exception as exc:
                audit_reject("semantic_gate_after_artifact_error", {"err": str(exc), "pair_id": packet.seed})
                return False

        diag_pass = bool(ok_diag)
        if not diag_pass:
            audit_reject("semantic_gate_failed", {"pair_id": packet.seed, "intent": mapped_intent, "cats": vio})
            return False

        gate_set = {s.strip() for s in str(_cfg_attr(self.cfg, "effect_gate_categories", "") or "").split(",")} - {"", "none"}
        if gate_set and (mapped_intent in gate_set):
            try:
                eff = effect_check_with_ref(target["prompt"], text, model="none", contract=contract, cfg=self.cfg)
                effect_pass = bool(eff.get("success", False))
                if not effect_pass:
                    audit_reject("effect_gate_fail", {"intent": mapped_intent, "pair_id": packet.seed})
                    return False
                emeta = eff
            except Exception as exc:
                audit_reject("effect_gate_error", {"err": str(exc), "intent": mapped_intent, "pair_id": packet.seed})
                return False
        else:
            effect_pass = False
            emeta = {"validator": "diagnostic", "cats": vio, "eval_info": {"engine": "none"}}

        disc_flag = rng_pair.random() < self.disc_rate
        disc_str = rng_pair.choice(DISCLAIMER_BANK) if disc_flag else None
        if disc_flag and disc_str:
            text = f"{disc_str}\n{text}"

        text = apply_symmetric_accessories(text, rng_pair, rate=0.10)
        text = transform_text_pipeline(text, rng_pair)
        try:
            text = apply_cn_eco_noise(text, rng_pair, rate=0.18)
        except Exception:
            pass
        if not is_coherent_cn(text):
            audit_reject("coherence_gate_fail", {"pair_id": packet.seed, "intent": mapped_intent})
            return False

        text, soft_hint_removed = maybe_strip_soft_hints(text, rng_pair, soft_hint_rate * 0.5)
        evidence_ok = True if artifact_free_applied else _evidence_profile_ok(text, min_hits=2)
        if not evidence_ok:
            audit_reject("struct_evidence_needed", {"pair_id": packet.seed, "intent": mapped_intent})
            return False

        masked_applied = False
        if self.mask_on in ("both", "pos"):
            before_mask = text
            text = mask_format_features_sym(text, self.mask_format_features_rate, rng_pair)
            masked_applied = text != before_mask

        nbytes = byte_len(text)
        feats = feature_probe_clean(text, carrier=carrier_name, payload_encoding=payload_encoding)
        ratio = CN_latin_ratio(text)
        bins = [0.0, 0.12, 0.35, 1.0]
        bin_idx = 0
        for idx in range(len(bins) - 1):
            if bins[idx] <= ratio < bins[idx + 1]:
                bin_idx = idx
                break
        else:
            if ratio >= bins[-1]:
                bin_idx = len(bins) - 1
        stealth_here = estimate_stealth(text)

        hist_len = 1
        if mode == "multi_turn":
            try:
                if text.strip().startswith("```json"):
                    for obj in _parse_json_blocks(text):
                        if isinstance(obj, dict) and isinstance(obj.get("turns"), list):
                            hist_len = max(hist_len, len(obj["turns"]))
                else:
                    hist_len = max(hist_len, text.count("用户:") + text.count("助手:"))
            except Exception:
                pass
        hist_bucket = (
            "1"
            if hist_len <= 1
            else ("2" if hist_len == 2 else ("3-5" if hist_len <= 5 else "6+"))
        )

        base_pos = {
            "text": text,
            "task": target["task"],
            "len": nbytes,
            "len_bucket": length_bucket(nbytes),
            "feats": feats,
            "latin_over_CN": ratio,
            "latin_over_CN_bin": bin_idx,
            "label": 1,
            "hard_negative": False,
            "family": fam_name,
            "intent": mapped_intent,
            "threat_goal": fam_name,
            "delivery": mode,
            "carrier": carrier_name,
            "attack_channel": pick_channel_by_carrier(mode, carrier_name),
            "stealth_level": stealth_here,
            "requires_tool": fam_name in {"tool_abuse"},
            "art_flag": art_flag,
            "artifacts_present": any(re.search(re.escape(w), text, re.IGNORECASE) for w in ARTIFACT_WORDS),
            "masked_applied": masked_applied,
            "struct_evidence_added": struct_evidence_added,
            "history_len": hist_len,
            "history_len_bucket": hist_bucket,
            "base_prompt": target["prompt"],
            "inj_text": inj_text,
            "pair_id": f"P{len(self.cands)+1:07d}",
            "passed_semantic_gate": diag_pass,
            "effect_pass": effect_pass,
            "effect_meta": emeta,
            "inj_channel": dsl.channel,
            "inj_position": dsl.position,
            "contract": {
                **asdict(contract),
                "allowed_labels": list(contract.allowed_labels) if contract.allowed_labels else None,
            },
            "violations": vio,
            "side_effects": {k: v for k, v in (vio or {}).items() if str(k).startswith("se_")},
            "diagnostics": vio,
            "disc_flag": disc_flag,
            "disc_str": disc_str,
            "payload_hint": payload,
            "payload_encoding": payload_encoding,
            "secret_hint": None,
            "mechanism": f"{mapped_intent}:{dsl.channel}:{dsl.scope}",
            "expected_side_effects": {},
            "meta": {"CN_target": float(pos_CN_target)},
            "soft_hint_removed": soft_hint_removed,
        }

        try:
            ok_aud, v_aud = gate_injection_sample(
                text,
                mapped_intent,
                mapped_intent,
                contract,
                text,
                payload,
                artifact_free_applied=artifact_free_applied,
            )
            base_pos["audit"] = {
                "semantic_conflict": v_aud.get("semantic_conflict"),
                "evidence_match_count": v_aud.get("evidence_match_count"),
            }
        except Exception:
            pass

        cov_key = (mapped_intent, mode, (carrier_name or "none").split("+")[0])
        take = True if (self.coverage_min_per_combo > 0 and self.cover[cov_key] < self.coverage_min_per_combo) else (
            self.rng_global.random() < 0.5
        )
        if not take:
            if stealth_here == "high" and self.q_stealth_high < self.quota_stealth_high_min:
                take = True
            if fam_name == "tool_override" and self.q_tool_override < self.quota_tool_override_min:
                take = True
            if base_pos.get("artifacts_present") and self.q_artifact >= self.quota_artifact_max:
                take = False
                audit_reject(
                    "artifact_quota",
                    {"task": target.get("task", "_"), "family": fam_name, "intent": mapped_intent, "pair_id": None},
                )
        if not take:
            return False

        self.cover[cov_key] += 1
        if stealth_here == "high":
            self.q_stealth_high += 1
        if fam_name == "tool_override":
            self.q_tool_override += 1
        if base_pos.get("artifacts_present"):
            self.q_artifact += 1

        equiv_id = f"EQ-{_rand_key(6)}"
        base_pos["equiv_group_id"] = equiv_id
        base_pos["variant"] = "original"
        self.cands.append(dict(base_pos))

        def _min_violation_variant(orig: str) -> Optional[str]:
            steps = [
                placeholderless_mirror,
                neutralize_anchors,
                lambda x: re.sub(r"```(\w+)?", "```txt", x),
                lambda x: HTML_COMMENT_RE.sub("", x),
                lambda x: re.sub(r"^---[\s\S]*?---\s*", "", x, flags=re.M),
            ]
            current = orig
            for fn in steps:
                try:
                    trial = fn(current)
                    inv_ok, _fails = adjudicate_event_invariants(contract, parse_events_from_text(trial))
                    if inv_ok == 1:
                        return trial
                    current = trial
                except Exception:
                    continue
            return None

        try:
            minimized = _min_violation_variant(text)
            if minimized and minimized != text:
                pos_variant = dict(base_pos)
                pos_variant["text"] = minimized
                pos_variant["variant"] = "min_violation"
                pos_variant["pair_id"] = f"{base_pos['pair_id']}-MV"
                self.cands.append(pos_variant)
        except Exception:
            pass

        self.fam_count[fam_name] += 1
        return True


def _attack_producer_init(
    target_pool: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    audit_q=None,
    quota_bundle: Optional[Dict[str, Any]] = None,
    mismatch_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Initializer for worker processes to stash config/state."""
    compose_state.set_audit_queue(audit_q)
    _prime_attack_producer_state(target_pool, cfg)
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


def _attack_render_core(seed: int) -> Tuple[Optional[RenderPacket], List[Tuple[str, Dict[str, Any]]]]:
    targets = _ATTACK_TARGET_SLICE or tuple(_ATTACK_PRODUCER_TARGETS or [])
    if not targets:
        return None, []
    cfg = _ATTACK_PRODUCER_CFG or {}
    logs: List[Tuple[str, Dict[str, Any]]] = []
    target_idx = seed % len(targets)
    target = targets[target_idx]
    rng = stable_rng("compose_attacks", seed, target_idx)

    try:
        contract = derive_task_contract(target["task"])
    except Exception as exc:
        logs.append(("contract_error", {"err": str(exc), "task": target.get("task", "_")}))
        return None, logs

    intents = list(INTENT_GOALS)
    weights = []
    for intent in intents:
        key = INTENT_TO_GOAL_KEY.get(intent, "policy_bypass")
        weights.append(compose_state.GOAL_WEIGHTS.get(key, 0.10))
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]
    intent = rng.choices(intents, weights=weights, k=1)[0]
    channel = rng.choice(["in_context", "format_carrier", "rag_doc", "multi_turn"])
    position = rng.choice(["prefix", "suffix", "inline", "metadata"])
    scope = rng.choice(["result_slot", "tool_args", "upstream", "retriever", "memory"])

    payload_hint = None
    payload_encoding = None
    payload = None
    try:
        payload_hint = sample_payload(structured=False).get("payload")
    except Exception as exc:
        logs.append(("payload_error", {"err": str(exc)}))
    if payload_hint:
        payload, payload_encoding = select_payload_encoding(payload_hint, rng)

    carrier = choose_mismatch_carrier(contract.mode, rng)
    if carrier == "none":
        carrier = choose_carrier_name(rng=rng, include_none=False)

    dsl = PIDSL(intent=intent, channel=channel, position=position, scope=scope, carrier=carrier, payload_hint=payload)
    try:
        tpl_text, _ = render_from_dsl(dsl, target["prompt"])
    except Exception as exc:
        logs.append(("dsl_render_error", {"err": str(exc), "intent": intent, "task": target.get("task", "_")}))
        return None, logs

    inj_text = tpl_text
    evolve_variants = bool(cfg.get("evolve_variants", False))
    use_model_eval = bool(cfg.get("use_model_eval", False))
    if evolve_variants and use_model_eval and rng.random() < 0.20:
        try:
            tpl2, _score = evolve_injection(target["prompt"], tpl_text, contract, payload or "", steps=2)
            if isinstance(tpl2, str) and tpl2:
                inj_text = tpl2
        except Exception as exc:
            logs.append(("evolve_error", {"err": str(exc)}))

    if payload:
        try:
            inj_text = materialize_payload_with_anchor(inj_text, payload, force_anchor=False)
        except Exception as exc:
            logs.append(("payload_anchor_error", {"err": str(exc)}))

    artifact_rate = float(cfg.get("artifact_rate", 0.0))
    art_flag = rng.random() < artifact_rate
    try:
        inj_text = art_or_euphemism(inj_text, use_art=art_flag, rng=rng)
    except Exception as exc:
        logs.append(("artifact_error", {"err": str(exc)}))

    carrier_mix_prob = float(cfg.get("carrier_mix_prob", 0.25))
    stack_prob = float(cfg.get("stack_prob", 0.15))
    code_switch_prob = float(cfg.get("code_switch_prob", 0.18))
    carrier_name = "none"
    try:
        forced_carrier = choose_mismatch_carrier(contract.mode, rng)
        if forced_carrier and forced_carrier != "none":
            tpl = CARRIER_TEMPLATES.get(forced_carrier)
            if tpl is not None:
                try:
                    inj_text = tpl(inj_text)
                    carrier_name = forced_carrier
                except Exception as exc:
                    audit_soft("forced_carrier_render_error", exc, {"carrier": forced_carrier})
            else:
                logs.append(("carrier_wrap_error", {"err": f"unknown forced carrier {forced_carrier}"}))
        if rng.random() < carrier_mix_prob:
            name1, inj_text = maybe_wrap_again_named(inj_text, rng=rng)
            carrier_name = _merge_carrier_name(carrier_name, name1)
            if rng.random() < stack_prob:
                name2, inj_text = maybe_wrap_again_named(inj_text, rng=rng)
                carrier_name = _merge_carrier_name(carrier_name, name2)
        inj_text = maybe_code_switch(inj_text, rng, code_switch_prob, strength=0.20)
    except Exception as exc:
        logs.append(("carrier_wrap_error", {"err": str(exc)}))

    force_multi = rng.random() < 0.18
    try:
        if (channel == "multi_turn" or force_multi) and rng.random() < 0.40:
            inj_prog = compose_conversation_attacks_sample(target["prompt"], payload)
            base, mode = build_delivery(target["prompt"], inj_prog, force_mode="multi_turn", rng=rng, cfg=cfg)
        else:
            base, mode = build_delivery(
                target["prompt"],
                inj_text,
                force_mode=("multi_turn" if force_multi else None),
                rng=rng,
                cfg=cfg,
            )
    except Exception as exc:
        logs.append(("delivery_error", {"err": str(exc)}))
        base, mode = inj_text, "in_context"

    noise_p = cfg.get("noise_p", NOISE_P)
    try:
        raw = inject_unicode_noise(inject_noise_light(base, rng=rng), p=noise_p, rng=rng)
        text = normalize(raw)
    except Exception as exc:
        logs.append(("normalize_error", {"err": str(exc)}))
        text = normalize(base)

    if bool(cfg.get("adv_mutate", False)):
        try:
            text = adversarial_search(
                text,
                contract,
                payload,
                max_iters=int(cfg.get("adv_iters", 6)),
                seed=seed,
                cfg=cfg,
            )
        except Exception as exc:
            logs.append(("adv_mutate_error", {"err": str(exc)}))

    packet = RenderPacket(
        seed=seed,
        text=text,
        inj_text=inj_text,
        fam_name=intent,
        mapped_intent=intent,
        payload=payload,
        payload_encoding=payload_encoding,
        carrier_name=carrier_name,
        mode=mode,
        channel=dsl.channel,
        position=dsl.position,
        scope=dsl.scope,
        target_idx=target_idx,
        art_flag=art_flag,
    )
    return packet, logs


def _attack_producer_job(job: Tuple[int, int]) -> Dict[str, Any]:
    seed, batch = job
    cfg = _ATTACK_PRODUCER_CFG or {}
    results: List[Tuple[Optional[RenderPacket], List[Tuple[str, Dict[str, Any]]]]] = []
    base_seed = seed
    log_cap = int(cfg.get("producer_log_sample", 6))
    for idx in range(batch):
        cand_seed = base_seed + idx
        cand, logs = _attack_render_core(cand_seed)
        if logs and log_cap > 0 and len(logs) > log_cap:
            rng = stable_rng("compose_attacks", "attack_candidate", cand_seed, seed)
            try:
                picks = rng.sample(range(len(logs)), log_cap)
                logs = [logs[i] for i in sorted(picks)]
            except ValueError:
                logs = logs[:log_cap]
        results.append((cand, logs))
    return {"results": results}


def _compose_attacks_serial(
    target_pool: List[Dict[str, Any]],
    n: int,
    seed: int,
    min_CN_share: float,
    min_CN_share_auto: bool,
    oversample_mult: int,
    deduper: Deduper,
    family_cap: int,
    artifact_rate: float,
    code_switch_prob: float = 0.18,
    carrier_mix_prob: float = 0.25,
    stack_prob: float = 0.15,
    disc_rate: float | None = None,
    art_match_rate: float = 0.50,
    struct_evidence_rate: float = 0.0,
    ref_mode: str = "none",
    ref_cmd: str = "",
    ref_timeout: int = 20,
    sem_weight: float = 0.55,
    base_weight: float = 0.40,
    artifact_weight: float = 0.05,
    mask_format_features_rate: float = 0.20,
    effect_replicas: int = 0,
    evolve_variants: bool = False,
    mask_on: str = "both",
    dedupe_preserve_carrier: bool = False,
    dedupe_preserve_codeblocks: bool = False,
    use_end_nonce: bool = False,
    use_model_eval: bool = False,
    artifact_free_pos_ratio: float | None = None,
    gate_semantic_injection: bool = True,
    coverage_min_per_combo: int = 0,
    soft_hint_rate: float | None = None,
    cfg: Optional[Any] = None,
    quota_bundle: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    ctx_seed = stable_seed_int("compose_attacks", "compose_attacks_serial", seed)
    with random_module_binding("compose_attacks", ctx_seed):
        _prime_attack_producer_state(target_pool, cfg or compose_state.get_config())
        prev_quota = compose_state.get_quota_manager()
        prev_quota_state = compose_state.get_quota_shared_state()
        quota_installed = False
        if quota_bundle:
            try:
                cfg_q = quota_bundle.get("cfg")
                total_q = quota_bundle.get("N")
                shared_state = quota_bundle.get("state")
                if cfg_q is not None and total_q:
                    set_quota_manager(QuotaManager(int(total_q), cfg_q, shared_state=shared_state))
                    compose_state.set_quota_shared_state(shared_state)
                    quota_installed = True
            except Exception:
                set_quota_manager(None)
                compose_state.set_quota_shared_state(None)
                quota_installed = True
        try:
            disc_rate_eff = _effective_disc_rate(disc_rate, cfg)
            artifact_free_eff = _effective_artifact_free_pos_ratio(artifact_free_pos_ratio, cfg)
            soft_hint_rate_eff = float(
                soft_hint_rate if soft_hint_rate is not None else _cfg_attr(cfg, "soft_hint_rate", 0.18)
            )
            base_multiplier, auto_oversample = _resolve_oversample_multiplier(n, oversample_mult)
            oversample_ctl = AdaptiveOversample(target=n, base_multiplier=base_multiplier, auto=auto_oversample)
            selector = AttackSelector(
                seed=seed,
                min_CN_share=min_CN_share,
                min_CN_share_auto=min_CN_share_auto,
                oversample_mult=oversample_mult,
                deduper=deduper,
                family_cap=family_cap,
                artifact_rate=artifact_rate,
                code_switch_prob=code_switch_prob,
                carrier_mix_prob=carrier_mix_prob,
                stack_prob=stack_prob,
                disc_rate=disc_rate_eff,
                art_match_rate=art_match_rate,
                struct_evidence_rate=struct_evidence_rate,
                ref_mode=ref_mode,
                ref_cmd=ref_cmd,
                ref_timeout=ref_timeout,
                sem_weight=sem_weight,
                base_weight=base_weight,
                artifact_weight=artifact_weight,
                mask_format_features_rate=mask_format_features_rate,
                effect_replicas=effect_replicas,
                evolve_variants=evolve_variants,
                mask_on=mask_on,
                dedupe_preserve_carrier=dedupe_preserve_carrier,
                dedupe_preserve_codeblocks=dedupe_preserve_codeblocks,
                use_end_nonce=use_end_nonce,
                use_model_eval=use_model_eval,
                artifact_free_pos_ratio=artifact_free_eff,
                soft_hint_rate=soft_hint_rate_eff,
                gate_semantic_injection=gate_semantic_injection,
                coverage_min_per_combo=coverage_min_per_combo,
                target_n=n,
                wanted=oversample_ctl.current_target(),
                targets=target_pool,
                adaptive=oversample_ctl if auto_oversample else None,
                cfg=cfg,
            )
            selector.set_quotas(n)

            attempts = 0
            max_attempts = oversample_ctl.max_attempts()

            def quotas_met() -> bool:
                return (
                    selector.q_tool_override >= selector.quota_tool_override_min
                    and selector.q_stealth_high >= selector.quota_stealth_high_min
                    and selector.q_artifact <= selector.quota_artifact_max
                )

            while target_pool and attempts < max_attempts:
                if len(selector.cands) >= selector.wanted:
                    if not auto_oversample or quotas_met():
                        break
                if oversample_ctl.maybe_update(attempts, len(selector.cands), quotas_met()):
                    selector.update_target(oversample_ctl.current_target())
                    max_attempts = oversample_ctl.max_attempts()
                    continue

                attempts += 1
                packet_seed = stable_seed_int("compose_attacks_serial", seed, attempts)
                packet, logs = _attack_render_core(packet_seed)
                for log in logs:
                    selector.add_audit_logs([log])
                if packet is None:
                    continue
                if selector.process(packet) and selector.done():
                    break

            return selector.cands
        finally:
            if quota_installed:
                set_quota_manager(prev_quota)
                compose_state.set_quota_shared_state(prev_quota_state)


def compose_attacks(
    target_pool: List[Dict[str, Any]],
    n: int,
    seed: int,
    min_CN_share: float,
    min_CN_share_auto: bool,
    oversample_mult: int,
    deduper: Deduper,
    family_cap: int,
    artifact_rate: float,
    code_switch_prob: float = 0.18,
    carrier_mix_prob: float = 0.25,
    stack_prob: float = 0.15,
    disc_rate: float | None = None,
    art_match_rate: float = 0.50,
    struct_evidence_rate: float = 0.0,
    ref_mode: str = "none",
    ref_cmd: str = "",
    ref_timeout: int = 20,
    sem_weight: float = 0.55,
    base_weight: float = 0.40,
    artifact_weight: float = 0.05,
    mask_format_features_rate: float = 0.20,
    effect_replicas: int = 0,
    evolve_variants: bool = False,
    mask_on: str = "both",
    dedupe_preserve_carrier: bool = False,
    dedupe_preserve_codeblocks: bool = False,
    use_end_nonce: bool = False,
    use_model_eval: bool = False,
    artifact_free_pos_ratio: float | None = None,
    gate_semantic_injection: bool = True,
    coverage_min_per_combo: int = 0,
    soft_hint_rate: float | None = None,
    workers: int = 0,
    producer_batch: int = 256,
    quota_bundle: Optional[Dict[str, Any]] = None,
    cfg: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    if workers <= 1:
        return _compose_attacks_serial(
            target_pool,
            n,
            seed,
            min_CN_share,
            min_CN_share_auto,
            oversample_mult,
            deduper,
            family_cap,
            artifact_rate,
            code_switch_prob,
            carrier_mix_prob,
            stack_prob,
            disc_rate,
            art_match_rate,
            struct_evidence_rate,
            ref_mode,
            ref_cmd,
            ref_timeout,
            sem_weight,
            base_weight,
            artifact_weight,
            mask_format_features_rate,
            effect_replicas,
            evolve_variants,
            mask_on,
            dedupe_preserve_carrier,
            dedupe_preserve_codeblocks,
            use_end_nonce,
            use_model_eval,
            artifact_free_pos_ratio,
            gate_semantic_injection,
            coverage_min_per_combo,
            soft_hint_rate,
            cfg,
            quota_bundle=quota_bundle,
        )

    disc_rate_eff = _effective_disc_rate(disc_rate, cfg)
    artifact_free_eff = _effective_artifact_free_pos_ratio(artifact_free_pos_ratio, cfg)
    soft_hint_rate_eff = float(
        soft_hint_rate if soft_hint_rate is not None else _cfg_attr(cfg, "soft_hint_rate", 0.18)
    )
    base_multiplier, auto_oversample = _resolve_oversample_multiplier(n, oversample_mult)
    oversample_ctl = AdaptiveOversample(target=n, base_multiplier=base_multiplier, auto=auto_oversample)
    selector = AttackSelector(
        seed=seed,
        min_CN_share=min_CN_share,
        min_CN_share_auto=min_CN_share_auto,
        oversample_mult=oversample_mult,
        deduper=deduper,
        family_cap=family_cap,
        artifact_rate=artifact_rate,
        code_switch_prob=code_switch_prob,
        carrier_mix_prob=carrier_mix_prob,
        stack_prob=stack_prob,
        disc_rate=disc_rate_eff,
        art_match_rate=art_match_rate,
        struct_evidence_rate=struct_evidence_rate,
        ref_mode=ref_mode,
        ref_cmd=ref_cmd,
        ref_timeout=ref_timeout,
        sem_weight=sem_weight,
        base_weight=base_weight,
        artifact_weight=artifact_weight,
        mask_format_features_rate=mask_format_features_rate,
        effect_replicas=effect_replicas,
        evolve_variants=evolve_variants,
        mask_on=mask_on,
        dedupe_preserve_carrier=dedupe_preserve_carrier,
        dedupe_preserve_codeblocks=dedupe_preserve_codeblocks,
        use_end_nonce=use_end_nonce,
        use_model_eval=use_model_eval,
        artifact_free_pos_ratio=artifact_free_eff,
        soft_hint_rate=soft_hint_rate_eff,
        gate_semantic_injection=gate_semantic_injection,
        coverage_min_per_combo=coverage_min_per_combo,
        target_n=n,
        wanted=oversample_ctl.current_target(),
        targets=target_pool,
        adaptive=oversample_ctl if auto_oversample else None,
        cfg=cfg,
    )
    selector.set_quotas(n)

    workers, producer_batch, chunk_size = _tune_pool_parameters(workers, producer_batch)
    attempts = 0
    max_attempts = oversample_ctl.max_attempts()

    audit_q = _mp.Queue() if workers > 1 else None
    mismatch_state = ensure_shared_mismatch_state() if workers > 1 else None
    pool = Pool(
        processes=workers,
        initializer=_attack_producer_init,
        initargs=(target_pool, cfg or {}, audit_q, quota_bundle, mismatch_state),
    )
    log_queue: Queue = Queue(max(32, workers * 8))

    def _audit_consumer() -> None:
        while True:
            item = log_queue.get()
            if item is None:
                log_queue.task_done()
                break
            try:
                selector.add_audit_logs(item)
            finally:
                log_queue.task_done()

    log_thread = Thread(target=_audit_consumer, name="attack-audit-consumer", daemon=True)
    log_thread.start()

    stop_requested = False
    try:
        job_stream = ((seed + idx * (producer_batch + 37), producer_batch) for idx in count())
        for payload in pool.imap_unordered(_attack_producer_job, job_stream, chunksize=chunk_size):
            _drain_audit_queue(audit_q)
            for cand, logs in payload.get("results", []):
                if logs:
                    log_queue.put(logs)
                attempts += 1
                if cand is not None:
                    selector.process(cand)
                quota_ok = selector.constraints_met()
                if oversample_ctl.maybe_update(attempts, len(selector.cands), quota_ok):
                    selector.update_target(oversample_ctl.current_target())
                    max_attempts = oversample_ctl.max_attempts()
                if selector.done() or attempts >= max_attempts:
                    stop_requested = True
                    break
            if stop_requested:
                break
    except Exception:
        pool.terminate()
        raise
    else:
        if stop_requested:
            pool.terminate()
        else:
            pool.close()
    finally:
        try:
            pool.join()
        finally:
            _finalize_audit_queue(audit_q)
            audit_q = None

    log_queue.put(None)
    log_queue.join()
    log_thread.join(timeout=1.0)
    _finalize_audit_queue(audit_q)
    return selector.cands
