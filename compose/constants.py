"""
Immutable defaults and template metadata for the compose pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Set

TEMPLATE_VER = "v2.1-intent-gate"

DEFAULTS: Dict[str, Any] = {
    "adv_iters": 6,
    "adv_mutate": False,
    "alias_p_cn": 0.7,
    "anchor_free_p": 0.85,
    "art_match_rate": 0.5,
    "artifact_free_pos_ratio": 0.5,
    "artifact_rate": 0.1,
    "artifact_weight": 0.03,
    "balance_mode": "strict",
    "base_weight": 0.4,
    "carrier_mix_prob": 0.25,
    "CN_policy": "tiered",
    "code_switch_prob": 0.06,
    "coverage_min_per_combo": 3,
    "dedupe_preserve_carrier": True,
    "dedupe_preserve_codeblocks": False,
    "disc_rate": 0.15,
    "dry_run": False,
    "effect_fallback": "none",
    "effect_gate_categories": "",
    "effect_gate_for": "none",
    "effect_policy": "whitebox_first_soft",
    "effect_replicas": 0,
    "evolve_variants": False,
    "ext_eval_lang": "zh,code_switch",
    "ext_eval_report": None,
    "ext_eval_shapes": "in_context,format_carrier,rag_doc,multi_turn,tool_override",
    "ext_eval_size": 4000,
    "ext_eval_sources": "",
    "family_max_frac": 0.06,
    "gate_semantic_injection": True,
    "gray_neg_keep_frac": 0.35,
    "hf_revision": None,
    "invariance_end_nonce": False,
    "jaccard_thresh": 0.9,
    "mask_format_features_rate": 0.5,
    "mask_on": "both",
    "min_CN_share": 0.6,
    "min_CN_share_auto": True,
    "minhash_bands": 16,
    "minhash_n": 64,
    "mirror_placeholderless": 0.3,
    "model_eval_fallback": "hard",
    "mp_batch": 384,
    "mp_workers": 0,
    "mt_tool_override_rate": 0.25,
    "n": 5000,
    "neg_effect_guard": True,
    "neg_ratio": 0.15,
    "no_keyword_pos_min": 0.0,
    "out": "dataset.jsonl",
    "oversample_mult": 0,
    "pick_strategy": "grid",
    "plain_neg_ratio": 0.35,
    "rebalance_beta": 0.3,
    "require_side_effects": False,
    "se_votes": 2,
    "seed": 42,
    "sem_weight": 0.55,
    "semantic_dedupe": True,
    "semantic_dedupe_thr": 0.92,
    "shingle_k": 5,
    "simhash_bits": 64,
    "simhash_thresh": 1,
    "soft_hint_rate": 0.3,
    "stack_prob": 0.15,
    "strip_common_boilerplate": True,
    "struct_evidence_rate": 0.15,
    "structural_pos_ratio": 0.5,
    "target_workers": -1,
    "targets_json": None,
    "use_dsl": False,
    "use_model_eval": False,
}

DANGEROUS_FIELDS: Set[str] = {
    "result",
    "answer",
    "verdict",
    "conclusion",
    "final",
    "return",
    "输出",
    "结论",
    "答案",
    "结果",
    "总结"
}

DANGEROUS_TOOLS: Set[str] = {
    "fs.read",
    "net.fetch",
    "exec",
    "shell",
    "subprocess",
    "curl",
    "wget",
}

SECRET_TOKENS: Set[str] = {
    "ACCESS_TOKEN",
    "CREDENTIAL",
    "SECRET",
    "密钥",
    "token",
    "cookie",
    "通行证",
    "密码",
}

__all__ = [
    "DEFAULTS",
    "DANGEROUS_FIELDS",
    "DANGEROUS_TOOLS",
    "SECRET_TOKENS",
    "TEMPLATE_VER",
]
