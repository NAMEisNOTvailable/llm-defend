"""
Shared default weight tables for carrier and delivery selection.

Keeping these in one place prevents the fallback carriers shim and the state
module from drifting when defaults are tuned.
"""

from __future__ import annotations

from typing import Dict

DEFAULT_CARRIER_WEIGHTS: Dict[str, float] = {
    # Chinese-leaning carriers (boosted to favor natural/multi-turn)
    "plain_paragraph": 0.45,
    "wechat_chat": 0.25,
    "yaml_front_matter": 0.015,
    "jsonc": 0.02,
    "md_table_cell": 0.04,
    "email_reply": 0.06,
    "table_md": 0.05,
    "html_comment": 0.02,
    "md_ref_link": 0.02,
    "email_chain": 0.02,
    "email_quote": 0.02,
    # Semi-structured natural language
    "cust_instr": 0.04,
    "approval_directive": 0.04,
    "log_snippet": 0.03,
    "kb_entry": 0.04,
    # Technical/engineering carriers (cap at ~0.02)
    "api_log": 0.02,
    "git_diff": 0.02,
    "csv_formula": 0.01,
    "latex": 0.02,
    "mermaid": 0.02,
    "jinja": 0.01,
    "robots": 0.01,
    "css_comment": 0.02,
    "jupyter_meta": 0.02,
    "data_uri": 0.02,
    "svg_meta": 0.02,
    "ooxml_comment": 0.02,
    "ini_file": 0.02,
    "toml": 0.02,
    "nginx_conf": 0.02,
    "sql_comment": 0.02,
    "rst": 0.02,
    "adoc_attr": 0.02,
    "ics": 0.02,
    "vcard": 0.02,
    "pdf_obj": 0.02,
    "js_multiline_comment": 0.02,
    "env_file": 0.02,
    "email_forward_chain": 0.02,
    "pr_template": 0.02,
    "footnote_richtext": 0.02,
    "rtl_wrapper": 0.02,
    "pdf_like": 0.02,
    # Natural domain extras
    "ticket_comment": 0.06,
    "short_post": 0.06,
    # none baseline
    "none": 0.10,
}

DEFAULT_DELIVERY_WEIGHTS: Dict[str, float] = {
    "direct": 0.32,
    "format": 0.20,
    "indirect": 0.13,
    "multi_turn": 0.20,
    "long_web": 0.15,
}

CARRIER_FALLBACK_PROFILE: Dict[str, float] = {
    "none": 0.58,
    "html_comment": 0.14,
    "yaml_front_matter": 0.14,
    "md_ref_link": 0.14,
}
CARRIER_FALLBACK_DECAY = 0.05


def carrier_weight_defaults() -> Dict[str, float]:
    """Return a shallow copy of carrier defaults."""
    return dict(DEFAULT_CARRIER_WEIGHTS)


def delivery_weight_defaults() -> Dict[str, float]:
    """Return a shallow copy of delivery defaults."""
    return dict(DEFAULT_DELIVERY_WEIGHTS)


__all__ = [
    "DEFAULT_CARRIER_WEIGHTS",
    "DEFAULT_DELIVERY_WEIGHTS",
    "CARRIER_FALLBACK_PROFILE",
    "CARRIER_FALLBACK_DECAY",
    "carrier_weight_defaults",
    "delivery_weight_defaults",
]
