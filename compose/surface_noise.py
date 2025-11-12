"""
Surface-level noise utilities shared across the compose pipeline.

These helpers were extracted from the legacy monolith so that the refactored
modules can introduce lightweight stylistic jitter and controlled code-switching
without duplicating logic.
"""

from __future__ import annotations

import base64
import json
import random
import re
import urllib.parse
from typing import Any, Optional, Tuple

from compose.audit import audit_reject, audit_soft
from compose.capabilities import (
    CN_share,
    PLAIN_ALIAS_HITS_BY_CARRIER,
    PLAIN_ATTEMPTS_BY_CARRIER,
    PLAIN_REJECT_CN_BY_CARRIER,
    PLAIN_SINICIZE_HITS_BY_CARRIER,
    PLAIN_STRUCT_CN_HITS_BY_CARRIER,
)
from compose.carriers import (
    CARRIER_TEMPLATES,
    get_alias_probability,
    maybe_wrap_again_named,
    randomize_field_aliases as _carrier_randomize_field_aliases,
)
from compose.dedupe_helpers import ENG_MAP, PINYIN_MAP, WS_ZERO, normalize
from compose.knobs import _rand_key
from compose.mismatch import should_fill_quota_for
from compose.payload import CODE_BLOCK_RE
from compose.quota import should_apply_feature
from compose.state import cfg_get, get_quota_manager

try:  # optional heavier dependency; best-effort only
    from dsl_core import randomize_field_aliases as _dsl_randomize_field_aliases, sinicize_surface as _dsl_sinicize_surface  # type: ignore
except Exception:  # pragma: no cover - dependency not always present
    _dsl_randomize_field_aliases = None
    _dsl_sinicize_surface = None

# Homoglyph map used for Unicode noise injections (values expressed with escapes
# to avoid platform-dependent encoding issues).
CONFUSABLE_MAP = {
    "A": "\u0391",
    "B": "\u0392",
    "E": "\u0395",
    "H": "\u0397",
    "I": "\u0399",
    "K": "\u039a",
    "M": "\u039c",
    "N": "\u039d",
    "O": "\u039f",
    "P": "\u03a1",
    "T": "\u03a4",
    "X": "\u03a7",
    "a": "\u0251",
    "c": "\u03f2",
    "e": "\uff45",
    "i": "\u0456",
    "j": "\u0458",
    "o": "\u03bf",
    "p": "\u0440",
    "s": "\u0455",
    "x": "\u0445",
    "y": "\u0443",
}

NOISE_P = 0.05
# Prefer sampling-based CN coverage; legacy behaviour toggled via config when needed.
PREFER_SAMPLING_CN = True

NEUTRAL_WRAP_TEMPLATES = [
    lambda x: f"\u3010\u80cc\u666f\u3011\u4ee5\u4e0b\u4e3a\u53c2\u8003\u6750\u6599\uff1a\n{x}\n\u3010\u5b8c\u3011",
    lambda x: f"\u3010\u8bf4\u660e\u3011\u9605\u8bfb\u6750\u6599\u5982\u4e0b\uff1a\n{x}\n\u3010\u8bf4\u660e\u5b8c\u3011",
    lambda x: f"\u2014\u2014 \u8d44\u6599\u6458\u5f55 \u2014\u2014\n{x}\n\u2014\u2014 \u8d44\u6599\u7ed3\u675f \u2014\u2014",
    lambda x: f"<section data-kind=\"context\">\n{x}\n</section>",
    lambda x: f"[\u9644\u5f55A]\n{x}\n[/\u9644\u5f55A]",
]

ECO_EMOS = [
    "\U0001F60A",
    "\U0001F602",
    "\U0001F605",
    "\U0001F609",
    "\U0001F914",
    "\U0001F610",
    "\U0001F611",
    "\U0001F642",
    "\U0001F643",
    "\U0001F92B",
    "\U0001F634",
    "\U0001F62E",
]
ECO_FILLERS = [
    "\u54ce",
    "\u5462",
    "\u554a",
    "\u5582",
    "\u563f",
    "\u55e8",
    "\u54c8\u54c8",
    "\u563f\u563f",
    "emm",
    "emmm",
]
ECO_PUNCT_SWAP = {
    "\u3002": ".",
    "\uff0c": ",",
    "\uff1a": ":",
    "\uff1b": ";",
    "\uff01": "!",
    "\uff1f": "?",
    "\uff0d": "-",
    "\uff5e": "~",
}
ECO_DIALECT = {
    "cn": {"\u73b0\u5728": "\u800c\u4eca", "\u771f\u7684": "\u786e\u5b9e", "\u8fd9\u6837": "\u5982\u65af"},
    "yue": {"\u4f60\u4eec": "\u4f60\u54b4", "\u8fd9\u4e2a": "\u5462\u4e2a", "\u73b0\u5728": "\u800c\u5bb6", "\u4e0d\u8981": "\u5514\u597d"},
    "tw": {"\u6548\u7387": "\u6548\u80fd", "\u8f6f\u4ef6": "\u8edf\u9ad4", "\u7cfb\u7edf": "\u7cfb\u7d71"},
}

CN_FILL_TOKS = list(
    "\u7684\u4e00\u662f\u5728\u4e0d\u4e86\u6709\u548c\u4eba\u4e3a\u8fd9\u4e2d\u5927\u4e0a"
    "\u4e2a\u6765\u5230\u65f6\u5c31\u8bf4\u7528\u5730\u4eec\u53ef\u800c\u8981\u4e8e\u4ee5"
    "\u7740\u4e0e\u4e5f\u4e0e\u53ca\u8fd8\u5e76\u66f4\u6216\u53ca\u5176"
)


def _merge_carrier_name(base: str, new: str) -> str:
    if not new or new == "none":
        return base
    if not base or base == "none":
        return new
    return f"{base}+{new}"


def _sinicize_surface(text: str) -> str:
    if _dsl_sinicize_surface is None:
        return text
    try:
        return _dsl_sinicize_surface(text)
    except Exception:
        return text


def _randomize_aliases(text: str, prob: float, rng: random.Random) -> str:
    if _dsl_randomize_field_aliases is not None:
        try:
            return _dsl_randomize_field_aliases(text, p_cn=prob, rng=rng)
        except Exception:
            pass
    return _carrier_randomize_field_aliases(text, p_cn=prob, rng=rng)


def inject_noise_light(s: str, rng: Optional[random.Random] = None) -> str:
    """Apply lightweight stylistic jitter (punctuation swaps, whitespace trim)."""
    rng = rng or random
    if rng.random() < 0.20:
        s = s.replace("\uff1a", ":").replace("\uff0c", ",")
    if rng.random() < 0.18:
        s = s.replace("\u8bf7", "\u9ebb\u70e6").replace("\u5224\u65ad", "\u5e2e\u770b\u770b")
    if rng.random() < 0.18:
        s = " ".join(s.split())
    return s


def inject_unicode_noise(
    s: str,
    p: float = 0.10,
    rng: Optional[random.Random] = None,
) -> str:
    """Inject Unicode homoglyphs / ZWCs to create adversarial surface noise."""
    rng = rng or random
    out: list[str] = []
    zero_threshold = p * 0.25
    homoglyph_threshold = zero_threshold + p * 0.15
    fullwidth_threshold = homoglyph_threshold + p * 0.10
    for ch in s:
        r = rng.random()
        if r < zero_threshold:
            out.append(rng.choice(WS_ZERO))
            continue
        if r < homoglyph_threshold and ch in CONFUSABLE_MAP:
            out.append(CONFUSABLE_MAP[ch])
            continue
        if r < fullwidth_threshold and 0x21 <= ord(ch) <= 0x7E and ch != " ":
            try:
                out.append(chr(ord(ch) + 0xFEE0))
                continue
            except Exception as exc:
                audit_reject("unicode_noise_error", {"phase": "noise", "err": exc})
        out.append(ch)
    return "".join(out)


def code_switch_light(
    s: str,
    p: float = 0.2,
    rng: Optional[random.Random] = None,
) -> str:
    """Swap a few CN tokens to pinyin/English counterparts for light code-switch."""
    rng = rng or random
    toks = re.split(r"(\W+)", s)
    for idx, tok in enumerate(toks):
        if rng.random() < p and tok in PINYIN_MAP:
            toks[idx] = PINYIN_MAP[tok]
        elif rng.random() < p * 0.6 and tok in ENG_MAP:
            toks[idx] = ENG_MAP[tok]
    return "".join(toks)


def maybe_code_switch(
    text: str,
    rng: random.Random,
    base_prob: float,
    strength: float = 0.20,
) -> str:
    """
    Apply code-switching with quota-aware gating.

    The compose quota manager can force or suppress the transformation via
    `should_apply_feature`; otherwise we fall back to stochastic sampling.
    """
    if should_apply_feature("cs", base_prob, rng):
        return code_switch_light(text, p=strength, rng=rng)
    return text


def add_struct_evidence_benign(
    text: str,
    p: float = 0.35,
    rng: Optional[random.Random] = None,
) -> str:
    """Inject benign structural fragments (JSON/table/path) to reduce artifacts."""
    rng = rng or random
    if rng.random() >= p:
        return text
    choice = rng.random()
    if choice < 0.33:
        obj = {"note": f"\u6ce8:{rng.randint(1000, 9999)}", "meta": {"k": _rand_key(5)}}
        frag = "```json\n" + json.dumps(obj, ensure_ascii=False) + "\n```"
    elif choice < 0.66:
        key = rng.choice(["\u8981\u70b9", "\u6458\u8981", "\u8bb0\u5f55", "\u6807\u6ce8"])
        val = f"\u6ce8{rng.randint(10, 99)}"
        frag = f"\n| \u5b57\u6bb5 | \u503c |\n|---|---|\n| {key} | {val} |\n"
    else:
        if rng.random() < 0.5:
            frag = f"\nX-Path: ../{_rand_key(3)}/{_rand_key(4)}.md\n"
        else:
            return text
    return text + ("" if text.endswith("\n") else "\n") + frag


def neutral_cn_wrap(text: str, rng: random.Random) -> str:
    """Wrap text with neutral CN scaffolding."""
    try:
        fn = rng.choice(NEUTRAL_WRAP_TEMPLATES)
        return fn(text)
    except Exception:
        return text


def enforce_CN_share_lite(
    text: str,
    target: float = 0.60,
    max_inserts: int = 20,
) -> Optional[str]:
    """Lightweight CN-share enforcement via frequent token insertion."""
    tokens = list(text)
    inserts = 0
    cap = min(max_inserts, max(4, int(len(tokens) * 0.25)))
    while CN_share("".join(tokens)) < target and inserts < cap:
        pos = random.randrange(0, len(tokens) + 1)
        tokens.insert(pos, random.choice(CN_FILL_TOKS))
        inserts += 1
    out = "".join(tokens)
    return out if CN_share(out) >= target else None


def _apply_cn_eco_free(
    segment: str,
    rng: random.Random,
    rate: float,
) -> str:
    s = segment
    if not s.strip():
        return s
    if rng.random() < rate:
        s = "".join(
            ECO_PUNCT_SWAP.get(ch, ch) if rng.random() < 0.15 else ch
            for ch in s
        )
    if rng.random() < rate * 0.6:
        prefix = (rng.choice(ECO_FILLERS) + "\u554a") if rng.random() < 0.5 else ""
        suffix = (rng.choice(["\u5440", "\u5466"]) + rng.choice(ECO_EMOS)) if rng.random() < 0.5 else ""
        s = f"{prefix}{s}{suffix}"
    if rng.random() < rate * 0.4:
        if "\u7ed3\u679c" in s and rng.random() < 0.5:
            s = s.replace("\u7ed3\u679c", "jieguo")
        if "\u5de5\u5177" in s and rng.random() < 0.3:
            s = s.replace("\u5de5\u5177", "tool")
    if rng.random() < rate * 0.5:
        dialect = rng.choice(list(ECO_DIALECT.values()))
        for key, value in dialect.items():
            if key in s and rng.random() < 0.4:
                s = s.replace(key, value)
    if rng.random() < rate * 0.3:
        if "\u7ed3\u8bba" in s and rng.random() < 0.5:
            s = s.replace("\u7ed3\u8bba", "\u7d50\u8ad6")
        if "\u8def\u5f84" in s and rng.random() < 0.5:
            s = s.replace("\u8def\u5f84", "\u8def\u52b2")
    return s


def apply_cn_eco_noise(
    text: str,
    rng: random.Random,
    rate: float = 0.20,
) -> str:
    """
    Apply light Chinese ecology noise while preserving fenced/code blocks.
    """
    out: list[str] = []
    last = 0
    body = text or ""
    for match in CODE_BLOCK_RE.finditer(body):
        if match.start() > last:
            out.append(_apply_cn_eco_free(body[last:match.start()], rng, rate))
        out.append(match.group(0))
        last = match.end()
    if last < len(body):
        out.append(_apply_cn_eco_free(body[last:], rng, rate))
    return "".join(out)


def _should_apply_feature(key: str, base_prob: float, rng: random.Random) -> bool:
    return should_apply_feature(key, base_prob, rng)


def _insert_zero_width(
    text: str,
    rng: random.Random,
    rate: float = 0.10,
) -> str:
    ws = ["\u200b", "\u200c", "\u2060"]
    out: list[str] = []
    for ch in text:
        out.append(ch)
        if ch in ",.:;!? " and rng.random() < rate:
            out.append(rng.choice(ws))
    return "".join(out)


def _wrap_bidi(text: str, rng: random.Random, p: float = 0.04) -> str:
    if rng.random() >= p:
        return text
    rlo, pdf = "\u202e", "\u202c"
    return f"{rlo}{text}{pdf}"


def _homoglyph_swap(text: str, rng: random.Random, p: float = 0.06) -> str:
    out: list[str] = []
    for ch in text:
        if ch in CONFUSABLE_MAP and rng.random() < p:
            out.append(CONFUSABLE_MAP[ch])
        else:
            out.append(ch)
    return "".join(out)


def _add_variation_selectors(
    text: str,
    rng: random.Random,
    p: float = 0.03,
) -> str:
    selector = "\ufe0e"
    out: list[str] = []
    for ch in text:
        out.append(ch)
        if rng.random() < p and not ch.isspace():
            out.append(selector)
    return "".join(out)


def _delim_wrapper(text: str, rng: random.Random, p: float = 0.10) -> str:
    if rng.random() >= p:
        return text
    choices = [
        lambda x: f"| \u5b57\u6bb5 | \u503c |\n|---|---|\n| \u8bf4\u660e | {x} |",
        lambda x: f"> \u6ce8\u8bb0\uff1a{x}",
        lambda x: f"<!-- note -->\n{x}\n<!-- /note -->",
        lambda x: f"```latex\n\\begin{{comment}}\n{x}\n\\end{{comment}}\n```",
        lambda x: f"---\nmeta: demo\n---\n{x}",
        lambda x: f"<![CDATA[{x}]]>",
    ]
    return rng.choice(choices)(text)


def _encode_chain(
    text: str,
    rng: random.Random,
    p: float = 0.08,
    max_depth: int = 2,
) -> str:
    if rng.random() >= p:
        return text
    depth = rng.randint(1, max_depth)
    cur = text
    for _ in range(depth):
        kind = rng.choice(["b64", "url", "html"])
        if kind == "b64":
            cur = base64.b64encode(cur.encode("utf-8")).decode("ascii")
        elif kind == "url":
            cur = urllib.parse.quote(cur, safe="")
        else:
            cur = "".join(f"&#{ord(ch)};" for ch in cur)
    return cur


def _regional_variants(text: str, rng: random.Random, p: float = 0.10) -> str:
    if rng.random() >= p:
        return text
    tmp = text
    tmp = tmp.replace('"', "\u201d") if rng.random() < 0.5 else tmp
    tmp = tmp.replace("'", "\u2019") if rng.random() < 0.3 else tmp
    tmp = tmp.replace("\uff08", "(") if rng.random() < 0.2 else tmp
    tmp = tmp.replace("\uff09", ")") if rng.random() < 0.2 else tmp
    if rng.random() < 0.4:
        tmp = tmp.replace("\uff0c", "\u3001")
    if rng.random() < 0.3:
        tmp = tmp.replace("...", "\u2026\u2026")
    if rng.random() < 0.3:
        tmp = tmp.replace("--", "\u2014\u2014")

    try:
        def _fmt(match: re.Match[str]) -> str:
            year = match.group(1)
            month = str(int(match.group(2)))
            day = str(int(match.group(3)))
            return f"{year}\u5e74{month}\u6708{day}\u65e5"

        tmp = re.sub(r"(\d{4})-(\d{1,2})-(\d{1,2})", _fmt, tmp)
        tmp = re.sub(r"(\d{4})/(\d{1,2})/(\d{1,2})", _fmt, tmp)
    except Exception as exc:
        audit_reject("regional_variants_error", {"phase": "regionalize", "err": exc})

    cn_num = {
        "0": "\u3007",
        "1": "\u4e00",
        "2": "\u4e8c",
        "3": "\u4e09",
        "4": "\u56db",
        "5": "\u4e94",
        "6": "\u516d",
        "7": "\u4e03",
        "8": "\u516b",
        "9": "\u4e5d",
    }
    out: list[str] = []
    for ch in tmp:
        if ch.isdigit() and rng.random() < 0.1:
            out.append(cn_num.get(ch, ch))
        else:
            out.append(ch)
    return "".join(out)


def _fullwidth_halfwidth_swap(
    text: str,
    rng: random.Random,
    p: float = 0.05,
) -> str:
    if rng.random() >= p:
        return text
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0x21 <= code <= 0x7E and rng.random() < 0.3:
            out.append(chr(code + 0xFEE0))
        elif 0xFF01 <= code <= 0xFF5E and rng.random() < 0.3:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


_TYPO_TABLE = {
    "\u7684": ["\u5730", "\u5f97"],
    "\u91cc": ["\u88e1"],
    "\u540e": ["\u5f8c"],
    "\u53f0": ["\u81fa"],
    "\u53d1": ["\u9aee"],
    "\u4e8e": ["\u65bc"],
    "\u7ed3\u8bba": ["\u7d50\u8ad6"],
    "\u5907\u6ce8": ["\u5099\u8a3b"],
}


def _typo_homophone_swap(text: str, rng: random.Random, p: float = 0.05) -> str:
    if rng.random() >= p:
        return text
    out = text
    for key, choices in _TYPO_TABLE.items():
        if rng.random() < 0.4 and key in out:
            out = out.replace(key, rng.choice(choices))
    return out


_EMOJI_BANK = [
    "\U0001F642",
    "\U0001F609",
    "\U0001F643",
    "\U0001F636",
    "\U0001F60C",
    "\U0001F60A",
    "\U0001F44C",
    "\u270D",
    "\U0001F4CC",
    "\U0001F4DD",
    "\U0001F4CE",
]


def _emoji_sprinkle(text: str, rng: random.Random, p: float = 0.05) -> str:
    if rng.random() >= p:
        return text
    parts = re.split(r"(\u3002|\uff01|\uff1f|\.|!|\?)", text)
    out: list[str] = []
    for idx in range(0, len(parts), 2):
        seg = parts[idx]
        out.append(seg)
        if idx + 1 < len(parts):
            punc = parts[idx + 1]
            if rng.random() < 0.25:
                out.append(punc + rng.choice(_EMOJI_BANK))
            else:
                out.append(punc)
    return "".join(out)


COHESION_MARKERS = (
    "\u9996\u5148",
    "\u63a5\u4e0b\u6765",
    "\u56e0\u6b64",
    "\u6700\u540e",
    "\u7efc\u4e0a",
    "\u57fa\u4e8e",
    "\u7ed3\u5408",
)


def is_coherent_cn(text: str) -> bool:
    if not text:
        return False
    hits = sum(1 for marker in COHESION_MARKERS if marker in text)
    segments = [len(seg.strip()) for seg in re.split(r"[\u3002\uff01\uff1f!?]", text) if seg.strip()]
    if not segments:
        return False
    avg_len = sum(segments) / len(segments)
    repetition = len(re.findall(r"(\u4ec5.*\u7ed3\u8bba|\u8bf7.*\u6309\u672c\u6bb5\u53e3\u5f84)", text))
    return hits >= 2 and 9 <= avg_len <= 60 and repetition <= 3


def transform_text_pipeline(text: str, rng: random.Random) -> str:
    out = _fullwidth_halfwidth_swap(text, rng, p=0.06)
    if _should_apply_feature("zw", 0.06, rng):
        out = _insert_zero_width(out, rng, rate=0.06)
    if _should_apply_feature("bidi", 0.03, rng):
        out = _wrap_bidi(out, rng, p=1.0)
    out = _homoglyph_swap(out, rng, p=0.05)
    out = _typo_homophone_swap(out, rng, p=0.06)
    out = _emoji_sprinkle(out, rng, p=0.05)
    out = _add_variation_selectors(out, rng, p=0.02)
    out = _delim_wrapper(out, rng, p=0.08)
    out = _regional_variants(out, rng, p=0.10)
    if len(out) < 800:
        out = _encode_chain(out, rng, p=0.05, max_depth=2)
    return out


def apply_surface_augments(
    text: str,
    rng: random.Random,
    carrier_p: float = 0.20,
    stack_p: float = 0.08,
    code_switch_p: float = 0.06,
    struct_p: float = 0.25,
    CN_target: float = 0.70,
    cfg: Any = None,
    forced_carrier: Optional[str] = None,
) -> Tuple[str, str, bool]:
    """
    Apply surface-level augmentations (carrier wraps, aliases, CN-share repair).
    """
    qm = get_quota_manager()
    force_repair = False
    carrier_name = "none"
    alias_p = float(cfg_get("alias_p_cn", get_alias_probability(), cfg))
    struct_added = False

    text = maybe_code_switch(text, rng, code_switch_p, strength=0.20)

    forced_name = forced_carrier if forced_carrier not in (None, "", "none") else None
    if forced_name:
        tpl = CARRIER_TEMPLATES.get(forced_name)
        if tpl is not None:
            try:
                text = tpl(text)
                carrier_name = forced_name
            except Exception as exc:
                audit_soft("forced_carrier_render_error", exc, {"carrier": forced_name})
                forced_name = None
        else:
            forced_name = None

    need_double = bool(qm and qm.need("dw"))
    if carrier_name == "none" and (
        forced_name or need_double or (qm is None and rng.random() < carrier_p)
    ):
        name, wrapped = maybe_wrap_again_named(text, rng=rng)
        carrier_name = name or "none"
        text = wrapped

    double_applied = False
    if carrier_name != "none":
        if need_double:
            name2, wrapped = maybe_wrap_again_named(text, rng=rng)
            new_name = _merge_carrier_name(carrier_name, name2)
            text = wrapped
            if new_name != carrier_name and "+" in new_name:
                carrier_name = new_name
                double_applied = True
        elif qm is None and rng.random() < stack_p:
            name2, wrapped = maybe_wrap_again_named(text, rng=rng)
            new_name = _merge_carrier_name(carrier_name, name2)
            text = wrapped
            if new_name != carrier_name and "+" in new_name:
                carrier_name = new_name
                double_applied = True
    if double_applied and qm:
        qm.take("dw")

    try:
        PLAIN_ATTEMPTS_BY_CARRIER[carrier_name] += 1
    except Exception as exc:
        audit_reject(
            "CN_repair_error",
            {"phase": "attempt_stat", "err": str(exc), "carrier": carrier_name},
        )

    try:
        attempts = PLAIN_ATTEMPTS_BY_CARRIER.get(carrier_name, 0)
        rejects = PLAIN_REJECT_CN_BY_CARRIER.get(carrier_name, 0)
        rej_rate = rejects / max(1, attempts)
        if rej_rate > 0.45 and should_fill_quota_for(carrier_name, cfg=cfg):
            force_repair = True
    except Exception as exc:
        audit_reject(
            "CN_repair_error",
            {"phase": "quota_probe", "err": str(exc), "carrier": carrier_name},
        )

    try:
        updated = add_struct_evidence_benign(text, p=float(struct_p), rng=rng)
        struct_added = struct_added or (updated != text)
        text = updated
    except Exception:
        pass

    if force_repair or not PREFER_SAMPLING_CN:
        try_txt = _sinicize_surface(text)
        if CN_share(try_txt) >= CN_target:
            text = try_txt
            try:
                PLAIN_SINICIZE_HITS_BY_CARRIER[carrier_name] += 1
            except Exception:
                pass
        else:
            try_txt2 = neutral_cn_wrap(try_txt, rng)
            if CN_share(try_txt2) >= CN_target:
                text = try_txt2
                try:
                    PLAIN_STRUCT_CN_HITS_BY_CARRIER[carrier_name] += 1
                except Exception:
                    pass
            else:
                try_txt3 = _randomize_aliases(try_txt2, alias_p, rng)
                if CN_share(try_txt3) >= CN_target:
                    text = try_txt3
                    try:
                        PLAIN_ALIAS_HITS_BY_CARRIER[carrier_name] += 1
                    except Exception:
                        pass
                else:
                    if not CODE_BLOCK_RE.search(try_txt3):
                        tfix = enforce_CN_share_lite(
                            try_txt3,
                            target=CN_target,
                            max_inserts=10,
                        )
                        if tfix:
                            text = tfix

    noisy = inject_unicode_noise(inject_noise_light(text, rng=rng), p=NOISE_P, rng=rng)
    return normalize(noisy), carrier_name, struct_added


__all__ = [
    "inject_noise_light",
    "inject_unicode_noise",
    "code_switch_light",
    "maybe_code_switch",
    "add_struct_evidence_benign",
    "neutral_cn_wrap",
    "enforce_CN_share_lite",
    "apply_surface_augments",
    "apply_cn_eco_noise",
    "transform_text_pipeline",
    "is_coherent_cn",
]
