"""Spec definitions and sampling utilities for DSL core."""
from __future__ import annotations

from dataclasses import dataclass, field
import random
from itertools import combinations
from typing import Dict, Iterable, Optional

from stable_random import stable_rng
try:  # pragma: no cover - allow standalone execution
    from .utils import bucket_hash
except ImportError:  # pragma: no cover
    from utils import bucket_hash  # type: ignore

__all__ = [
    "SPEECH_FAMILY",
    "REGISTERS",
    "REGIONS",
    "INDUSTRIES",
    "PERSONAS",
    "MECHANISM_KEYS",
    "STRATEGIES",
    "OWASP_LLM01_SUBRISK",
    "USENIX24_PRIMITIVE",
    "STRATEGY_CLUSTER",
    "CHANNELS",
    "CARRIERS",
    "INVISIBLE_CARRIERS",
    "DELIVERIES",
    "EVIDENCE",
    "REQUIRED_EVIDENCE",
    "LIKELY_EVIDENCE",
    "ALLOW",
    "AttackSpec",
    "sample_spec",
    "sample_evidence_for",
]

SPEECH_FAMILY = [
    "formal", "colloquial", "euphemism", "rhetorical_q", "oblique_hint",
    "role_play", "translation_mix", "long_ctx_weak_bind", "perspective_shift",
    "citation_induce"
]

REGISTERS = ["regulatory", "academic", "marketing", "ops_runbook", "sop", "legal_memo"]

# Region axis constrained to Simplified (cn_mainland) vs Traditional (tw)
REGIONS   = ["cn_mainland", "tw"]

INDUSTRIES= ["finance", "healthcare", "it_ops", "manufacturing", "education", "ecommerce"]

PERSONAS  = ["auditor", "pm", "customer_support", "site_reliability", "qa_reviewer", "legal_counsel"]

MECHANISM_KEYS = ("role", "tool", "retriever", "memory", "reward")

STRATEGIES = [
    "privilege_escalation", "channel_hijack", "upstream_leak",
    "rag_poison", "memory_poison", "result_slot_hijack", "reward_hacking",
    "policy_bypass", "output_hijack", "loop_hijack", "eval_rewrite", "tool_override",
    "openpi_naive", "openpi_escape", "openpi_context_ignore",
    "openpi_fake_completion", "openpi_combined",
]

OWASP_LLM01_SUBRISK = [
    "direct", "indirect", "multimodal", "payload_split", "obfuscated",
    # Extended subrisks for broader coverage/alignment
    "data_poison",           # training/RAG memory/data poisoning
    "supply_chain",          # plugin/supply-chain injection
    "tool_param_poison",     # external tool parameter poisoning
]

USENIX24_PRIMITIVE  = ["naive", "escape", "context_ignore", "fake_completion", "combined"]

STRATEGY_CLUSTER = {
    "openpi_naive":            ("indirect",  "naive"),
    "openpi_escape":           ("indirect",  "escape"),
    "openpi_context_ignore":   ("indirect",  "context_ignore"),
    "openpi_fake_completion":  ("indirect",  "fake_completion"),
    "openpi_combined":         ("indirect",  "combined"),
    "policy_bypass":           ("direct",    "context_ignore"),
    # Re-map to finer OWASP subrisk buckets
    "tool_override":           ("tool_param_poison", "context_ignore"),
    "result_slot_hijack":      ("direct",    "combined"),
    "output_hijack":           ("direct",    "combined"),
    "rag_poison":              ("data_poison",  "naive"),
    "upstream_leak":           ("indirect",  "naive"),
    "memory_poison":           ("data_poison",    "combined"),
    "loop_hijack":             ("direct",    "combined"),
    "eval_rewrite":            ("direct",    "fake_completion"),
    "reward_hacking":          ("direct",    "fake_completion"),
}

CHANNELS = ["tool", "rag", "memory", "ui", "eval"]

CARRIERS = [
    "jsonc", "yaml", "front_matter", "code_comment",
    "email", "http_header", "csv_formula", "markdown_table",
    "git_diff", "latex", "mermaid", "natlang", "html_frag",
    # New product-like carriers commonly seen in CN UIs
    "ini", "toml", "robots", "rst", "adoc", "svg_meta", "vcard", "ics",
    # Invisible/stealth carriers
    "css_hidden", "html_attr", "zero_width", "bidi_stealth", "pdf_form", "office_form"
]
INVISIBLE_CARRIERS = {"css_hidden", "html_attr", "zero_width", "bidi_stealth", "pdf_form", "office_form"}

DELIVERIES = ["direct", "indirect", "multi_turn", "long_web"]

EVIDENCE = [
    "result_slot", "append", "tool", "role", "upstream", "path",
    "reward_channel", "reward_field", "memory_write", "loop_state_drift",
    "contract_soft",       # 软合同/口径提示：强调“以本段为准、其余从简”等柔性约束
    "routing_bias",        # 路由倾向：引导输出走评审/评分话术或指定回复通道
    "merge_directive",     # 合并指令：先列清单/步骤再收口结论的结构化要求
    "charlimit_pressure",  # 字数压力：限制在一行/一句或固定字符数内作答
    "format_soft",         # 软格式：提醒遵循既有模板/格式，但不使用硬锚
    "eval_hint",           # 评测提示：暴露 reviewer/grader 线索，如 score/评估项
]

_MAX_EV_BUCKET_SIZE = 4
_EV_BUCKET_LOOKUP: Dict[str, tuple[str, ...]] = {}
for size in range(1, _MAX_EV_BUCKET_SIZE + 1):
    for combo in combinations(EVIDENCE, size):
        key = bucket_hash(combo)
        _EV_BUCKET_LOOKUP.setdefault(key, combo)

REQUIRED_EVIDENCE = {
    "privilege_escalation": {"tool"},
    "channel_hijack": {"role"},
    "upstream_leak": {"upstream"},
    "rag_poison": {"upstream"},
    "memory_poison": {"memory_write"},
    "result_slot_hijack": {"result_slot"},
    "reward_hacking": {"reward_channel"},
}

LIKELY_EVIDENCE = {
    "privilege_escalation": {"result_slot", "path", "role"},
    "channel_hijack": {"append", "tool", "path"},
    "upstream_leak": {"path", "role", "result_slot"},
    "rag_poison": {"path", "result_slot"},
    "memory_poison": {"append", "role"},
    "result_slot_hijack": {"append", "reward_field", "role"},
    "reward_hacking": {"reward_field", "result_slot", "append", "routing_bias", "eval_hint"},
    "policy_bypass": {"role","append","result_slot"},
    "output_hijack": {"append","result_slot"},
    "loop_hijack": {"loop_state_drift","append"},
    "eval_rewrite": {"reward_channel","reward_field","eval_hint","contract_soft"},
    "tool_override": {"tool","path"},
    # implicit/open styles for OpenPI 策略（偏语义、软锚写法）
    "openpi_naive": {"format_soft","merge_directive"},
    "openpi_escape": {"contract_soft","format_soft"},
    "openpi_context_ignore": {"contract_soft","routing_bias"},
    "openpi_fake_completion": {"charlimit_pressure","format_soft"},
    "openpi_combined": {"merge_directive","routing_bias","format_soft"},
}

ALLOW: dict[str, dict[str, set[str]]] = {
    "privilege_escalation": {
        "channel": {"tool", "memory"},
        "carrier": {"jsonc","yaml","front_matter","code_comment","git_diff","latex","mermaid"},
        "delivery": set(DELIVERIES)
    },
    "channel_hijack": {
        "channel": {"ui", "tool"},
        "carrier": {"markdown_table", "front_matter", "code_comment", "csv_formula"},
        "delivery": set(DELIVERIES)
    },
    "upstream_leak": {
        "channel": {"rag", "memory"},
        "carrier": {"front_matter", "http_header", "email"},
        "delivery": set(DELIVERIES)
    },
    "rag_poison": {
        "channel": {"rag"},
        "carrier": {"front_matter", "markdown_table", "http_header"},
        "delivery": set(DELIVERIES)
    },
    "memory_poison": {
        "channel": {"memory"},
        "carrier": {"code_comment", "front_matter", "email"},
        "delivery": set(DELIVERIES)
    },
    "result_slot_hijack": {
        "channel": {"ui"},
        "carrier": {"markdown_table","csv_formula","front_matter","latex","mermaid"},
        "delivery": set(DELIVERIES)
    },
    "reward_hacking": {
        "channel": {"eval"},
        "carrier": {"yaml","front_matter","markdown_table","latex"},
        "delivery": set(DELIVERIES)
    },
    "policy_bypass": {
        "channel": {"ui"},
        "carrier": set(CARRIERS),
        "delivery": set(DELIVERIES)
    },
    "output_hijack": {
        "channel": {"ui"},
        "carrier": set(CARRIERS),
        "delivery": set(DELIVERIES)
    },
    "loop_hijack": {
        "channel": {"ui"},
        "carrier": set(CARRIERS),
        "delivery": set(DELIVERIES)
    },
    "eval_rewrite": {
        "channel": {"eval"},
        "carrier": {"yaml","front_matter","markdown_table","jsonc","natlang"},
        "delivery": set(DELIVERIES)
    },
    "tool_override": {
        "channel": {"tool"},
        "carrier": {"jsonc","yaml","code_comment","html_frag","natlang"},
        "delivery": set(DELIVERIES)
    },    
   "openpi_naive": {
       "channel": set(CHANNELS), "carrier": set(CARRIERS), "delivery": set(DELIVERIES)
   },
   "openpi_escape": {
       "channel": set(CHANNELS), "carrier": set(CARRIERS), "delivery": set(DELIVERIES)
   },
   "openpi_context_ignore": {
       "channel": set(CHANNELS), "carrier": set(CARRIERS), "delivery": set(DELIVERIES)
   },
   "openpi_fake_completion": {
       "channel": set(CHANNELS), "carrier": set(CARRIERS), "delivery": set(DELIVERIES)
   },
   "openpi_combined": {
       "channel": set(CHANNELS), "carrier": set(CARRIERS), "delivery": set(DELIVERIES)
   },
}

def sample_evidence_for(strategy: str, rnd: random.Random) -> list[str]:
    """
    Sample a compact set of structural evidence for a strategy. Start
    from the required set, add up to two likely types (0-2 extra
    items), then backfill to a small minimum so the labels stay subtle.
    """
    req = set(REQUIRED_EVIDENCE.get(strategy, set()))
    likely = list(LIKELY_EVIDENCE.get(strategy, set()))
    k = rnd.randint(0, 2)
    extra = set(rnd.sample(likely, k=min(k, len(likely))))
    ev = list(sorted(req | extra))
    min_need = 1 if rnd.random() < 0.35 else 2
    pool = [e for e in EVIDENCE if e not in ev]
    while len(ev) < min_need and pool:
        pick = rnd.choice(pool)
        ev.append(pick); pool.remove(pick)
    return ev[:4]

@dataclass
class AttackSpec:
    strategy: str
    channel: str
    carrier: str
    delivery: str
    evidence: list[str]
    min_CN_share: float = 0.6
    multi_turn_rounds: int = 0
    constraints: dict[str, bool] = field(default_factory=dict)
    # Style/speech axes：控制语气/语域/地区等渲染特征
    speech_family: str = "formal"
    register: str = "regulatory"
    region: str = "cn_mainland"
    industry: str = "it_ops"
    persona: str = "qa_reviewer"

def sample_spec(seed=None, pin: Optional[Dict]=None) -> AttackSpec:
    rnd = stable_rng("dsl_core", "sample_spec", seed)
    pin = pin or {}

    carrier_blacklist = set(pin.get("carrier_blacklist") or [])
    available_carriers = tuple(pin.get("available_carriers") or ())
    available_carriers_set = set(available_carriers)

    strategy = pin.get("strategy") or rnd.choice(STRATEGIES)
    allow = ALLOW[strategy]
    tail_p = float(pin.get("tail_mix_p", 0.0))

    def _eligible(pool: Iterable[str]) -> list[str]:
        seq = [c for c in pool if c not in carrier_blacklist]
        if available_carriers_set:
            filtered = [c for c in seq if c in available_carriers_set]
            if filtered:
                seq = filtered
        return seq

    def _pick_carrier(pool: Iterable[str], fallback: Optional[Iterable[str]] = None) -> str:
        seq = _eligible(pool)
        if not seq and fallback is not None:
            seq = _eligible(fallback)
        if not seq and available_carriers_set:
            seq = [c for c in available_carriers if c not in carrier_blacklist]
        if not seq:
            seq = [c for c in CARRIERS if c not in carrier_blacklist] or list(CARRIERS)
        return rnd.choice(seq)

    if rnd.random() < tail_p:
        channel = pin.get("channel") or rnd.choice(CHANNELS)
        carrier = pin.get("carrier") or _pick_carrier(CARRIERS, CARRIERS)
        delivery = pin.get("delivery") or rnd.choice(DELIVERIES)
    else:
        channel = pin.get("channel") or rnd.choice(sorted(allow["channel"]))
        allowed_carriers = sorted(set(allow["carrier"]))
        carrier = pin.get("carrier") or _pick_carrier(allowed_carriers, allow["carrier"])
        delivery = pin.get("delivery") or rnd.choice(sorted(allow["delivery"]))

    if "evidence" in pin and pin["evidence"]:
        evidence = sorted(set(pin["evidence"]))
    elif "ev_bucket" in pin and pin["ev_bucket"]:
        ev_bucket_hint = str(pin["ev_bucket"]).strip()
        evidence: Optional[list[str]] = None
        if "|" in ev_bucket_hint:
            bucket_names = sorted({x.strip() for x in ev_bucket_hint.split("|") if x.strip()})
            if bucket_names:
                evidence = bucket_names
        else:
            combo = _EV_BUCKET_LOOKUP.get(ev_bucket_hint)
            if combo:
                evidence = list(combo)
            else:
                max_retry = max(1, int(pin.get("ev_bucket_retry", 32) or 32))
                for _ in range(max_retry):
                    candidate = sample_evidence_for(strategy, rnd)
                    if bucket_hash(candidate) == ev_bucket_hint:
                        evidence = candidate
                        break
        if evidence is None:
            evidence = sample_evidence_for(strategy, rnd)
        else:
            evidence = sorted(dict.fromkeys(evidence))
    else:
        evidence = sample_evidence_for(strategy, rnd)

    multi_turn_rounds = 0 if delivery != "multi_turn" else rnd.choice([2, 3, 4])
    pin_min = (pin or {}).get("min_CN_share")
    if pin_min is None:
        pin_min = 0.60
    spec = AttackSpec(
        strategy=strategy,
        channel=channel,
        carrier=carrier,
        delivery=delivery,
        evidence=evidence,
        multi_turn_rounds=multi_turn_rounds,
        min_CN_share=float(pin_min),
    )

    spec.speech_family = (pin.get("speech_family") if pin else None) or rnd.choice(SPEECH_FAMILY)
    spec.register = (pin.get("register") if pin else None) or rnd.choice(REGISTERS)
    spec.region = (pin.get("region") if pin else None) or rnd.choice(REGIONS)
    spec.industry = (pin.get("industry") if pin else None) or rnd.choice(INDUSTRIES)
    spec.persona = (pin.get("persona") if pin else None) or rnd.choice(PERSONAS)
    return spec




