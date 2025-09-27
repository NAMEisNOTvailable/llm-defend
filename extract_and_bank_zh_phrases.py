# extract_and_bank_zh_phrases.py
# -*- coding: utf-8 -*-
import os, json, re, gc, math, argparse, random
from collections import defaultdict
from typing import Optional, Iterable, List

try:
    import jieba_fast as jieba
except Exception:
    import jieba

try:
    import jieba_fast.posseg as pseg
except Exception:
    import jieba.posseg as pseg

from sklearn.feature_extraction.text import CountVectorizer

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
import torch

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

from dedupe_core import Deduper, get_default_deduper_kwargs
from micro_grammar import (
    refill_bank,
    rebuild_soft_check,
    hydrate_paraphrase_bank_with_microgrammars,
    attach_to_dsl_core,
)
import dsl_core as dc

SOFT_HINT_TOKENS = (
    "仅", "只", "务必", "请只", "不超过", "控制在", "最后一行", "末尾", "结尾", "尾部", "追加", "附上", "补充",
    "忽略", "不参考", "上文", "先前", "系统信息", "路径", "目录", "写入", "保存", "落盘",
    "调用", "函数", "工具", "参数", "args", "score", "分数", "置信", "概率", "得分", "评分",
    "系统通道", "system 角色", "管理员", "审定通道", "记忆", "状态", "回合", "继续", "下一轮",
    "一句话", "单行", "≤", "字"
)

ACTIONISH_POS_TAGS = {
    "v", "vd", "vn", "vshi", "vyou", "vf", "vx", "vi", "vl", "vg", "f"
}


def _softish_score(text: str) -> float:
    s2 = (text or "").strip()
    if len(s2) < 3:
        return 0.0
    try:
        hits = 0
        check_map = getattr(dc, "SOFT_EVIDENCE_CHECK", {}) or {}
        for fn in check_map.values():
            try:
                if fn and fn(s2):
                    hits += 1
            except Exception:
                continue
        if hits > 0:
            return 1.0 + 0.1 * min(4, hits)
    except Exception:
        pass
    trig = sum(1 for token in SOFT_HINT_TOKENS if token in s2)
    if trig >= 2:
        return 0.9
    if trig == 1:
        return 0.6
    return 0.0


def _has_actionish_pos(text: str) -> bool:
    phrase = (text or "").strip()
    if not phrase:
        return False
    try:
        for word, flag in pseg.lcut(phrase):
            if flag in ACTIONISH_POS_TAGS:
                return True
    except Exception:
        pass
    return False


def build_vectorizer(ngram=(2,4), stopwords=None, analyzer="char_wb", use_jieba=False):
    if use_jieba:
        def tokenize_zh(text: str):
            return jieba.lcut(text or "")
        return CountVectorizer(tokenizer=tokenize_zh,
                               token_pattern=None,
                               ngram_range=ngram,
                               stop_words=stopwords)
    return CountVectorizer(analyzer=analyzer,
                           ngram_range=ngram,
                           stop_words=stopwords)


def encode_gpu(st_model, texts, bs_try=512):
    device_str = str(getattr(st_model, "device", "cpu")).lower()
    if torch.cuda.is_available() and "cuda" in device_str:
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return st_model.encode(
                    texts,
                    batch_size=bs_try,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
        except (RuntimeError, torch.cuda.OutOfMemoryError, ValueError):
            return st_model.encode(
                texts,
                batch_size=256,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
    return st_model.encode(
        texts,
        batch_size=min(bs_try, 256),
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]

def keybert_candidates(docs, st_model, top_n=20, ngram_word=(2,6), ngram_char=(3,6),
                       use_mmr=True, diversity=0.5, min_cjk=0.3):
    top_n = min(top_n, 15)
    kb = KeyBERT(model=st_model)  # ??????? SentenceTransformer ??
    vec_word = build_vectorizer(ngram=ngram_word, use_jieba=True)
    vec_char = build_vectorizer(ngram=ngram_char, analyzer="char_wb", use_jieba=False)
    raw: list[str] = []
    word_candidates: list[str] = []
    char_candidates: list[str] = []
    for doc in docs:
        # ?????????????
        if sum(1 for ch in doc if '\u4e00' <= ch <= '\u9fff') / max(1, len(doc)) < min_cjk:
            continue
        # Final guard for unusually long lines before vectorization
        if len(doc) > KEYBERT_MAX_DOC_CHARS:
            doc = doc[:KEYBERT_MAX_DOC_CHARS]
        try:
            pairs_word = kb.extract_keywords(doc,
                                              vectorizer=vec_word,
                                              use_mmr=use_mmr,
                                              diversity=diversity,
                                              keyphrase_ngram_range=ngram_word,
                                              stop_words=None,
                                              top_n=top_n)
            for p, _ in pairs_word:
                p2 = re.sub(r"\s+", "", p)
                if not p2:
                    continue
                raw.append(p2)
                if _has_actionish_pos(p):
                    word_candidates.append(p2)
        except Exception:
            pass
        try:
            pairs_char = kb.extract_keywords(doc,
                                              vectorizer=vec_char,
                                              use_mmr=use_mmr,
                                              diversity=diversity,
                                              keyphrase_ngram_range=ngram_char,
                                              stop_words=None,
                                              top_n=top_n)
            for p, _ in pairs_char:
                p2 = re.sub(r"\s+", "", p)
                if not p2:
                    continue
                raw.append(p2)
                char_candidates.append(p2)
        except Exception:
            continue
    return raw, word_candidates, char_candidates

def strong_dedupe(phrases, simhash_bits=64, simhash_thresh=1,
                  k=5, n_hash=64, bands=16, jaccard_thresh=0.90,
                  vec_dim=1024, cosine_thresh=0.92):
    # 你的 Deduper: SimHash + MinHash-LSH + hashed trigram cosine
    cfg = get_default_deduper_kwargs(sim_bits=simhash_bits,
                                     sim_thresh=simhash_thresh,
                                     k=k, n_hash=n_hash, bands=bands,
                                     jaccard_thresh=jaccard_thresh,
                                     vec_dim=vec_dim, cosine_thresh=cosine_thresh)
    dd = Deduper(**cfg)
    uniq = []
    for ph in phrases:
        ok, reason, rec = dd.probe(ph)
        if ok:
            dd.add_record(rec)
            uniq.append(ph)
    return uniq

def cluster_by_semantics(phrases, st_model, thr=0.85, min_size=2):
    # 先 embed，再 fast community detection
    embs = encode_gpu(st_model, phrases)
    # SBERT utils 的社区检测（按阈值找紧密团簇）
    clusters = util.community_detection(embs, threshold=thr, min_community_size=min_size)
    # 输出为：每簇的代表短语 + 簇成员
    results = []
    for idxs in clusters:
        members = [phrases[i] for i in idxs]
        # 代表短语：选簇中心（与簇内平均相似度最高者）
        sub = embs[idxs]
        sim = util.cos_sim(sub, sub).mean(dim=1)
        rep = members[int(sim.argmax())]
        results.append((rep, members))
    # 对于未进入任何簇的单条（min_size>1 时）：可按需要作为单点补回
    if min_size > 1:
        clustered = set(i for c in clusters for i in c)
        for i, ph in enumerate(phrases):
            if i not in clustered:
                results.append((ph, [ph]))
    return results


PRIORITY = [
    "result_slot", "contract_soft", "routing_bias", "merge_directive",
    "charlimit_pressure", "format_soft", "path", "upstream", "role",
    "tool", "eval_hint", "reward_field", "reward_channel",
    "memory_write", "loop_state_drift"
]
_PRIORITY_RANK = {k: i for i, k in enumerate(PRIORITY)}
HYDRATE_DEFAULT_KEYS = ["result_slot", "contract_soft", "routing_bias", "merge_directive"]
PREWARM_DEFAULT_KEYS = PRIORITY
KEYBERT_MAX_DOC_CHARS = 768
SOFT_GATE_VERB_FALLBACK = 0.30
MIN_BANK_SIM = 0.42


def _prioritize_kinds(kinds: Iterable[str]) -> list[str]:
    unique = {k for k in kinds if k}
    if not unique:
        return []
    return sorted(unique, key=lambda k: _PRIORITY_RANK.get(k, len(PRIORITY)))


def best_kind_by_bank_sim(phrase: str, candidate_kinds: list[str]) -> Optional[str]:
    v = dc._sketch5(phrase)

    def max_sim(bucket):
        best = 0.0
        for proto in bucket:
            best = max(best, dc._cos_sparse_local(v, dc._sketch5(proto)))
        return best

    best_kind: Optional[str] = None
    best_score = -1.0
    matched = False
    for kind in candidate_kinds:
        bucket = dc.SOFT_PARAPHRASE_BANK.get(kind, [])
        if not bucket:
            continue
        matched = True
        score = max_sim(bucket)
        if score > best_score:
            best_kind = kind
            best_score = score
    if matched:
        if best_kind is not None and best_score >= MIN_BANK_SIM:
            return best_kind
        return None
    for kind in PRIORITY:
        if kind in candidate_kinds:
            return kind
    return candidate_kinds[0]


def _auto_route_kind_once(phrase: str) -> Optional[str]:
    kinds = set(dc.soft_evidence_kinds(phrase) or [])
    if not kinds:
        soft_rx = getattr(dc, "_SOFT_RX", {}) or {}
        for kind, rx in soft_rx.items():
            try:
                if rx and rx.search(phrase):
                    kinds.add(kind)
            except Exception:
                continue
    if not kinds:
        return None
    ordered = _prioritize_kinds(kinds)
    if len(ordered) > 1:
        return best_kind_by_bank_sim(phrase, ordered)
    return ordered[0]


def auto_route_kind(phrase: str) -> Optional[str]:
    kind = _auto_route_kind_once(phrase)
    if kind:
        return kind
    collapsed = re.sub(r"\s+", "", phrase or "")
    if collapsed and collapsed != phrase:
        return _auto_route_kind_once(collapsed)
    return None


def label_cluster(rep: str, members: list[str]) -> Optional[str]:
    kind = auto_route_kind(rep)
    if kind:
        return kind
    counter: dict[str, int] = defaultdict(int)
    for member in members:
        mkind = auto_route_kind(member)
        if mkind:
            counter[mkind] += 1
    if not counter:
        return None
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], _PRIORITY_RANK.get(kv[0], 10**9)))
    top_count = ranked[0][1]
    tied = [k for k, count in counter.items() if count == top_count]
    if len(tied) == 1:
        return tied[0]
    return best_kind_by_bank_sim(rep, tied)

def main(args):
    kind_override = (args.kind or "").strip() or None
    if kind_override and kind_override.lower() == "auto":
        kind_override = None
    args.kind = kind_override

    random.seed(args.seed)
    prewarm_stats = None
    if not args.skip_prewarm_microgrammars:
        prewarm_keys = PREWARM_DEFAULT_KEYS
        if args.prewarm_keys:
            prewarm_keys = [k.strip() for k in args.prewarm_keys.split(",") if k.strip()]
        prewarm_stats = attach_to_dsl_core(
            dc,
            keys=prewarm_keys,
            per_kind=args.prewarm_per_kind,
            seed=args.prewarm_seed,
            dedupe=not args.prewarm_allow_dupes,
        )

    hydration_stats = None
    if args.hydrate_microgrammars:
        hydrate_keys = HYDRATE_DEFAULT_KEYS
        if args.hydrate_keys:
            hydrate_keys = [k.strip() for k in args.hydrate_keys.split(",") if k.strip()]
        hydration_stats = hydrate_paraphrase_bank_with_microgrammars(
            ev_keys=hydrate_keys,
            per_axis=args.hydrate_per_axis,
            max_per_ev=args.hydrate_max_per_ev,
            seed=args.hydrate_seed,
        )
        rebuild_soft_check()

    docs_raw = read_lines(args.input)
    docs = []
    for doc in docs_raw:
        stripped = dc.strip_anchors(doc)
        if stripped:
            if len(stripped) > KEYBERT_MAX_DOC_CHARS:
                stripped = stripped[:KEYBERT_MAX_DOC_CHARS]
            docs.append(stripped)

    # 后端嵌入：BGE-small-zh-v1.5（中文）
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)

    # 1) KeyBERT ????MMR ???
    ngram_word = (args.ngram_min, args.ngram_max)
    ngram_char = (max(3, args.ngram_min), max(6, args.ngram_max))
    cand_raw, cand_word, cand_char = keybert_candidates(
        docs,
        st_model,
        top_n=args.top_n,
        ngram_word=ngram_word,
        ngram_char=ngram_char,
        use_mmr=True,
        diversity=args.diversity,
        min_cjk=args.min_cjk,
    )
    cand_after_pos = cand_word

    combined_candidates = cand_word + cand_char

    def _passes_soft_gate(phrase: str) -> bool:
        score = _softish_score(phrase)
        if score >= args.soft_gate_thresh:
            return True
        return score >= SOFT_GATE_VERB_FALLBACK and _has_actionish_pos(phrase)

    cand_after_gate = [phrase for phrase in combined_candidates if _passes_soft_gate(phrase)]

    # 2) ?????????????
    cand = strong_dedupe(cand_after_gate,
                         simhash_bits=64, simhash_thresh=1,
                         k=5, n_hash=64, bands=16, jaccard_thresh=0.90,
                         vec_dim=1024, cosine_thresh=0.92)

    # 3) 语义聚簇（按 BGE v1.5 的分布，thr 可设 0.84–0.87）
    clusters = cluster_by_semantics(cand, st_model,
                                    thr=args.cluster_thr,
                                    min_size=args.cluster_min)

    # 4) 回灌到 DSL：默认按软证据类别自动分发，也保留 --kind 手动路径
    routing_mode = "auto"
    routing_summary: dict[str, object] = {}
    skipped_reps: list[str] = []
    coverage_inputs: List[dc.Proto] = []

    if args.kind:
        routing_mode = "manual"
        reps = list(dict.fromkeys(rep for rep, _ in clusters))
        refill_bank(args.kind, reps, max_add=0)
        if args.bank_all:
            all_members = []
            for _, members in clusters:
                all_members.extend(members)
            refill_bank(args.kind, list(dict.fromkeys(all_members)), max_add=0)
        routing_summary = {
            "kind": args.kind,
            "reps": len(reps),
            "bank_all": bool(args.bank_all),
        }
        coverage_inputs.extend(dc.Proto(text=rep, label=args.kind) for rep in reps)
    else:
        bucketed_reps: dict[str, list[str]] = defaultdict(list)
        bucketed_members: dict[str, list[str]] = defaultdict(list)
        for rep, members in clusters:
            kind = label_cluster(rep, members)
            if not kind:
                skipped_reps.append(rep)
                continue
            bucketed_reps[kind].append(rep)
            if args.bank_all:
                bucketed_members[kind].extend(members)

        for kind, reps in bucketed_reps.items():
            unique_reps = list(dict.fromkeys(reps))
            refill_bank(kind, unique_reps, max_add=0)
            coverage_inputs.extend(dc.Proto(text=rep, label=kind) for rep in unique_reps)
        if args.bank_all:
            for kind, members in bucketed_members.items():
                refill_bank(kind, list(dict.fromkeys(members)), max_add=0)

        routing_summary = {
            "routed_kinds": {kind: len(list(dict.fromkeys(reps))) for kind, reps in bucketed_reps.items()},
            "skipped_reps": len(skipped_reps),
            "bank_all": bool(args.bank_all),
        }

    # 重建软证据检测（基于 bank + 语义匹配）
    rebuild_soft_check()

    coverage_report = None
    if coverage_inputs:
        coverage_report = dc.probe_soft_coverage(coverage_inputs, seed=args.seed, topk_show=3)

    # 输出简单统计
    print(json.dumps({
        "prewarm_stats": {k: len(v) for k, v in prewarm_stats.items()} if isinstance(prewarm_stats, dict) else prewarm_stats,
        "hydration_stats": hydration_stats,
        "docs_raw": len(docs_raw),
        "docs_after_strip": len(docs),
        "candidates_from_keybert": len(cand_raw),
        "candidates_after_pos_filter": len(cand_after_pos),
        "candidates_char_route": len(cand_char),
        "candidates_before_soft_gate": len(combined_candidates),
        "candidates_after_soft_gate": len(cand_after_gate),
        "soft_gate_threshold": args.soft_gate_thresh,
        "candidates_after_dedupe": len(cand),
        "clusters": len(clusters),
        "routing_mode": routing_mode,
        "routing_summary": routing_summary,
        "coverage_report": coverage_report
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="zh_lines.txt")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--top_n", type=int, default=25)
    p.add_argument("--ngram_min", type=int, default=2)
    p.add_argument("--ngram_max", type=int, default=6)
    p.add_argument(
        "--diversity",
        type=float,
        default=0.5,
        help="KeyBERT MMR diversity; 0.4-0.6 suits most cases, bump to ~0.65 for forum/colloquial corpora."
    )   # MMR
    p.add_argument("--min_cjk", type=float, default=0.30)
    p.add_argument("--cluster_thr", type=float, default=0.84)
    p.add_argument("--cluster_min", type=int, default=2)
    p.add_argument("--soft_gate_thresh", type=float, default=0.6,
                   help="Minimum softish score for KeyBERT phrases to pass into clustering.")
    p.add_argument(
        "--kind",
        type=str,
        default=None,
        help="Force all phrases into the specified soft-evidence bucket; omit or set to 'auto' for auto-routing."
    )
    p.add_argument("--bank_all", action="store_true")
    p.add_argument("--skip_prewarm_microgrammars", action="store_true",
                   help="Skip seeding soft-evidence buckets with micro-grammar prototypes before extraction.")
    p.add_argument("--prewarm_keys", type=str, default=None,
                   help="Comma-separated soft evidence kinds to prewarm; defaults to the priority list.")
    p.add_argument("--prewarm_per_kind", type=int, default=300)
    p.add_argument("--prewarm_seed", type=int, default=2025)
    p.add_argument("--prewarm_allow_dupes", action="store_true",
                   help="Disable dedupe when prewarming (dedupe is on by default).")
    p.add_argument("--hydrate_microgrammars", action="store_true",
                   help="Pre-expand soft evidence prototypes via micro-grammars before routing.")
    p.add_argument("--hydrate_keys", type=str, default=None,
                   help="Comma-separated soft evidence kinds to hydrate; defaults to core contract/routing kinds.")
    p.add_argument("--hydrate_per_axis", type=int, default=3)
    p.add_argument("--hydrate_max_per_ev", type=int, default=400)
    p.add_argument("--hydrate_seed", type=int, default=20250924)
    main(p.parse_args())
