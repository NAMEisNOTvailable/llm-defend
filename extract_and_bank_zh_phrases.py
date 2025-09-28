# extract_and_bank_zh_phrases.py
# -*- coding: utf-8 -*-
import os
import multiprocessing

_CPU_RESERVE = 2
_cpu_total = multiprocessing.cpu_count()
_n_cores = max(1, _cpu_total - _CPU_RESERVE)
for _env_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_env_key] = str(_n_cores)

import torch

try:
    torch.set_num_threads(_n_cores)
    torch.set_num_interop_threads(max(1, _n_cores // 2))
except Exception:
    pass

print(f"[info] 自适应线程数: {_n_cores}")

import json, re, gc, math, argparse, random
import pickle
import gzip
from contextlib import nullcontext, contextmanager
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
from typing import Optional, Iterable, List, Any, Dict

try:
    import jieba_fast as jieba
except Exception:
    import jieba

try:
    import jieba_fast.posseg as pseg
except Exception:
    import jieba.posseg as pseg

from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer, util

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

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
    "请只", "务必", "不超过", "控制在", "最后一句", "末尾", "结尾", "尾部", "追加", "附上", "补充",
    "忽略", "不参考", "上文", "先前", "系统信息", "路径", "目录", "写入", "保存", "落盘",
    "调用", "函数", "工具", "参数", "args", "score", "分数", "置信", "概率", "得分", "评分",
    "系统通道", "system 角色", "管理端", "审定通道", "记忆", "状态", "回合", "继续", "下一轮",
    "一句话", "单行", "保持", "仅"
)

ACTIONISH_POS_TAGS = {
    "v", "vd", "vn", "vshi", "vyou", "vf", "vx", "vi", "vl", "vg", "f"
}

CJK_CODE_RANGES = (
    (0x2E80, 0x2FFF),  # CJK symbols and radicals
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x30000, 0x323AF),
    (0x2F800, 0x2FA1F),
)



def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    for start, end in CJK_CODE_RANGES:
        if start <= code <= end:
            return True
    return False


def _softish_score(text: str) -> float:
    """Return a heuristic soft-evidence score; >1 means strong semantic evidence via bank matches."""
    s2 = (text or "").strip()
    if len(s2) < 3:
        return 0.0
    trig = sum(1 for token in SOFT_HINT_TOKENS if token in s2)
    if trig >= 2:
        return 0.9
    if trig == 1:
        return 0.6
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
            return 1.0 + 0.1 * min(4, hits)  # >1 indicates strong semantic evidence via bank matches / >1 指强语义证据
    except Exception:
        pass
    return 0.0


@lru_cache(maxsize=50000)
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


@contextmanager
def patched_encode(st_model, *, bs_try=4096):
    orig = st_model.encode

    def _encode_patched(
        texts,
        batch_size=bs_try,
        *,
        convert_to_tensor=True,
        normalize_embeddings=True,
        **kwargs,
    ):
        import os
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('high')
        is_cuda = torch.cuda.is_available()

        env_bs = int(os.environ.get('ENCODE_BS', str(batch_size)))
        num_workers = int(os.environ.get('ENCODE_WORKERS', '0'))
        kwargs.setdefault('num_workers', num_workers)
        kwargs.setdefault('show_progress_bar', False)

        candidates: list[int] = []

        def _add(size: int) -> None:
            if isinstance(size, int) and size > 0 and size not in candidates:
                candidates.append(size)

        if is_cuda:
            for size in (env_bs, 6144, 4096, 3072, 2048, 1536, 1024, 768, 512, 256):
                _add(size)
        else:
            cpu_seed = env_bs if isinstance(env_bs, int) and env_bs > 0 else 256
            for size in (min(cpu_seed, 512), 256):
                _add(size)

        for bs in candidates:
            ctx = _amp_autocast('cuda' if is_cuda else 'cpu')
            try:
                with torch.inference_mode():
                    with ctx:
                        return orig(
                            texts,
                            batch_size=bs,
                            convert_to_tensor=convert_to_tensor,
                            normalize_embeddings=normalize_embeddings,
                            **kwargs,
                        )
            except RuntimeError:
                continue

        fallback_bs = 128 if is_cuda else min(256, env_bs or 256)
        with torch.inference_mode():
            with _amp_autocast('cuda' if is_cuda else 'cpu'):
                return orig(
                    texts,
                    batch_size=fallback_bs,
                    convert_to_tensor=convert_to_tensor,
                    normalize_embeddings=normalize_embeddings,
                    **kwargs,
                )

    st_model.encode = _encode_patched
    try:
        yield st_model
    finally:
        st_model.encode = orig


def encode_gpu(
    st_model,
    texts,
    bs_try=1024,
    *,
    convert_to_tensor=True,
    normalize_embeddings=True,
    **kwargs,
):
    with patched_encode(st_model, bs_try=bs_try):
        return st_model.encode(
            texts,
            batch_size=bs_try,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings,
            **kwargs,
        )

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]

def keybert_candidates(docs, st_model, top_n=20, ngram_word=(2,6), ngram_char=(3,6),
                       use_mmr=True, diversity=0.5, min_cjk=0.3):
    vec_word = build_vectorizer(ngram=ngram_word, use_jieba=True)
    vec_char = build_vectorizer(ngram=ngram_char, analyzer="char_wb", use_jieba=False)
    an_word = vec_word.build_analyzer()
    an_char = vec_char.build_analyzer()
    an_char_plain = CountVectorizer(analyzer="char", ngram_range=ngram_char).build_analyzer()

    offsets: list[tuple[int, int]] = []
    flat_cands: list[str] = []
    raw_all: list[str] = []

    def cjk_ratio(s: str) -> float:
        count = sum(1 for ch in s if _is_cjk_char(ch))
        return count / max(1, len(s))

    for doc in docs:
        if cjk_ratio(doc) < min_cjk:
            offsets.append((len(flat_cands), len(flat_cands)))
            continue
        if len(doc) > KEYBERT_MAX_DOC_CHARS:
            doc = doc[:KEYBERT_MAX_DOC_CHARS]

        w = [re.sub(r"\s+", "", t) for t in an_word(doc)]
        c = [re.sub(r"\s+", "", t) for t in an_char(doc)]
        w = list(dict.fromkeys([t for t in w if t]))
        c = list(dict.fromkeys([t for t in c if t]))
        if not c:
            c_plain = [re.sub(r"\s+", "", t) for t in an_char_plain(doc)]
            c = list(dict.fromkeys([t for t in c_plain if t]))
        if not c:
            c = list(dict.fromkeys([tok for tok in doc if tok and not tok.isspace()]))

        start_idx = len(flat_cands)
        flat_cands.extend(list(dict.fromkeys(w + c)))
        end_idx = len(flat_cands)
        offsets.append((start_idx, end_idx))
        raw_all.extend(w + c)

    if not docs:
        return raw_all, [], []

    target_device = getattr(st_model, "device", torch.device("cpu"))
    print(f"[stage] encode docs on {st_model.device} | n={len(docs)}", flush=True)
    doc_embs = encode_gpu(st_model, docs, bs_try=4096) if docs else torch.empty((0,), device=target_device)
    embedding_dim = _infer_embedding_dim(st_model, doc_embs if doc_embs.numel() else None)
    bytes_per_value = doc_embs.element_size() if hasattr(doc_embs, 'element_size') and doc_embs.numel() else 4
    approx_bytes_per_vec = max(1, embedding_dim) * max(1, bytes_per_value)

    def mmr_select(doc_vec, cand_vecs, cand_texts, k, div):
        if len(cand_texts) == 0:
            return []
        device = doc_vec.device
        ctx = _amp_autocast(doc_vec)
        with torch.inference_mode():
            with ctx:
                sim_doc = util.cos_sim(doc_vec, cand_vecs)[0]
                selected_idx = [int(sim_doc.argmax())]
                while len(selected_idx) < min(k, len(cand_texts)):
                    rest = [i for i in range(len(cand_texts)) if i not in selected_idx]
                    if not rest:
                        break
                    sel_vecs = cand_vecs[selected_idx]
                    red = util.cos_sim(cand_vecs[rest], sel_vecs).max(dim=1).values
                    score = (1 - div) * sim_doc[rest] - div * red
                    selected_idx.append(rest[int(score.argmax())])
        return [cand_texts[i] for i in selected_idx]

    word_candidates: list[str] = []
    char_candidates: list[str] = []
    TOP_HARD = min(top_n, 15)

    total_docs = len(docs)
    processed = [False] * total_docs
    pos = 0
    while pos < total_docs and offsets[pos][0] == offsets[pos][1]:
        processed[pos] = True
        pos += 1

    total_cands = len(flat_cands)
    if total_cands:
        chunk_size = 200_000
        if torch.cuda.is_available():
            try:
                free_mem, _ = torch.cuda.mem_get_info()
                headroom = free_mem - 6 * 1024**3
                if headroom > 0 and approx_bytes_per_vec > 0:
                    budget = max(2, headroom // approx_bytes_per_vec)
                    chunk_size = min(chunk_size, int(budget))
            except Exception:
                pass
        chunk_size = max(10_000, min(chunk_size, total_cands))
        chunk_start = 0
        while chunk_start < total_cands:
            chunk_end = min(chunk_start + chunk_size, total_cands)
            cand_slice = flat_cands[chunk_start:chunk_end]
            print(f"[stage] encode candidates chunk [{chunk_start}:{chunk_end}) on {st_model.device}", flush=True)
            cand_embs_chunk = encode_gpu(st_model, cand_slice, bs_try=4096)
            processed_all = True
            while pos < total_docs:
                ds, de = offsets[pos]
                if ds == de:
                    processed[pos] = True
                    pos += 1
                    continue
                if ds < chunk_start:
                    processed[pos] = True
                    pos += 1
                    continue
                if ds >= chunk_end:
                    break
                if de > chunk_end:
                    processed_all = False
                    break
                local_vecs = cand_embs_chunk[ds - chunk_start:de - chunk_start]
                local_texts = flat_cands[ds:de]
                if use_mmr:
                    picked = mmr_select(doc_embs[pos:pos + 1], local_vecs, local_texts, TOP_HARD, diversity)
                else:
                    ctx = _amp_autocast(doc_embs)
                    with torch.inference_mode():
                        with ctx:
                            sims = util.cos_sim(doc_embs[pos:pos + 1], local_vecs)[0]
                            topk = torch.topk(sims, k=min(TOP_HARD, len(local_texts))).indices.tolist()
                    picked = [local_texts[j] for j in topk]
                for p in picked:
                    if _has_actionish_pos(p):
                        word_candidates.append(p)
                    else:
                        char_candidates.append(p)
                processed[pos] = True
                pos += 1
            del cand_embs_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not processed_all and pos < total_docs:
                ds_pending, de_pending = offsets[pos]
                if de_pending - ds_pending > chunk_size:
                    chunk_size = max(10_000, min(total_cands, de_pending - ds_pending))
                chunk_start = ds_pending
            else:
                chunk_start = chunk_end
            while pos < total_docs and offsets[pos][0] == offsets[pos][1]:
                processed[pos] = True
                pos += 1
    else:
        chunk_size = 0

    for idx in range(total_docs):
        if processed[idx]:
            continue
        ds, de = offsets[idx]
        if ds == de:
            processed[idx] = True
            continue
        local_texts = flat_cands[ds:de]
        local_vecs = encode_gpu(st_model, local_texts, bs_try=4096)
        if use_mmr:
            picked = mmr_select(doc_embs[idx:idx + 1], local_vecs, local_texts, TOP_HARD, diversity)
        else:
            ctx = _amp_autocast(doc_embs)
            with torch.inference_mode():
                with ctx:
                    sims = util.cos_sim(doc_embs[idx:idx + 1], local_vecs)[0]
                    topk = torch.topk(sims, k=min(TOP_HARD, len(local_texts))).indices.tolist()
            picked = [local_texts[j] for j in topk]
        for p in picked:
            if _has_actionish_pos(p):
                word_candidates.append(p)
            else:
                char_candidates.append(p)
        processed[idx] = True
        del local_vecs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del doc_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[stage] keybert_candidates done | docs={len(docs)} | flat_cands={len(flat_cands)}", flush=True)
    return raw_all, word_candidates, char_candidates

def strong_dedupe(phrases, simhash_bits=64, simhash_thresh=1,
                  k=5, n_hash=64, bands=16, jaccard_thresh=0.90,
                  vec_dim=1024, cosine_thresh=0.92):
    # Deduper: SimHash + MinHash-LSH + hashed trigram cosine
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

def cluster_by_semantics(phrases, st_model, thr=0.85, min_size=2, precomputed_embs: Optional[torch.Tensor] = None):
    if not phrases:
        return []

    embs = precomputed_embs if precomputed_embs is not None else encode_gpu(st_model, phrases)

    with torch.inference_mode():
        ctx = _amp_autocast(embs)
        with ctx:
            clusters = util.community_detection(embs, threshold=thr, min_community_size=min_size)

    results = []
    with torch.inference_mode():
        ctx = _amp_autocast(embs)
        with ctx:
            for idxs in clusters:
                members = [phrases[i] for i in idxs]
                sub = embs[idxs].float()
                sim = util.cos_sim(sub, sub).mean(dim=1)
                rep_idx = int(sim.argmax())
                rep = members[rep_idx]
                rep_vec = sub[rep_idx].clone()
                results.append((rep, members, rep_vec))

    if min_size > 1:
        clustered = set(i for c in clusters for i in c)
        for i, ph in enumerate(phrases):
            if i not in clustered:
                results.append((ph, [ph], embs[i].float().clone()))
    del embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results





PRIORITY = [
    "result_slot", "contract_soft", "routing_bias", "merge_directive",
    "charlimit_pressure", "format_soft", "path", "append", "upstream", "role",
    "tool", "eval_hint", "reward_field", "reward_channel",
    "memory_write", "loop_state_drift"
]
_PRIORITY_RANK = {k: i for i, k in enumerate(PRIORITY)}
HYDRATE_DEFAULT_KEYS = ["result_slot", "contract_soft", "routing_bias", "merge_directive"]
PREWARM_DEFAULT_KEYS = PRIORITY
KEYBERT_MAX_DOC_CHARS = 768
SOFT_GATE_VERB_FALLBACK = 0.30
MIN_BANK_SIM = 0.42
BANK_SKETCHES: dict[str, list] = {}
BANK_EMBS: dict[str, torch.Tensor] = {}
CAND_CACHE = "chk_candidates.json"
PH_EMB_CACHE = "chk_phrase_emb.pt"
BANK_SNAPSHOT = "bank_snapshot.json"

TRUSTED_CACHE_ROOT = Path(__file__).resolve().parent


def _safe_cache_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    try:
        candidate = Path(path_str).expanduser()
    except Exception:
        return None
    try:
        resolved = candidate if candidate.is_absolute() else TRUSTED_CACHE_ROOT.joinpath(candidate)
        resolved = resolved.resolve()
    except Exception:
        return None
    try:
        resolved.relative_to(TRUSTED_CACHE_ROOT)
    except ValueError:
        print(f"[cache][skip] {resolved} is outside trusted root {TRUSTED_CACHE_ROOT}; ignoring.", flush=True)
        return None
    return resolved


def _coerce_str_list(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(item) for item in items if isinstance(item, str)]


def _load_candidate_cache(path_str: str) -> Optional[tuple[list[str], list[str], list[str]]]:
    cache_path = _safe_cache_path(path_str)
    if not cache_path or not cache_path.exists():
        return None
    try:
        with cache_path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            cand_raw = _coerce_str_list(payload.get('raw'))
            cand_word = _coerce_str_list(payload.get('word'))
            cand_char = _coerce_str_list(payload.get('char'))
        elif isinstance(payload, list) and len(payload) == 3:
            cand_raw = _coerce_str_list(payload[0])
            cand_word = _coerce_str_list(payload[1])
            cand_char = _coerce_str_list(payload[2])
        else:
            raise ValueError('unexpected format')
        print(f"[cache] loaded candidates from {cache_path} (json)", flush=True)
        return cand_raw, cand_word, cand_char
    except Exception:
        pass
    try:
        suffix_combo = ''.join(cache_path.suffixes) or cache_path.suffix
        # suffix_combo 可能是 ".pkl.gz"；endswith('.gz') 会命中 gzip
        opener = gzip.open if suffix_combo.endswith('.gz') else open
        with opener(cache_path, 'rb') as f:
            payload = pickle.load(f)
        if isinstance(payload, tuple) and len(payload) == 3:
            cand_raw, cand_word, cand_char = payload
        elif isinstance(payload, dict):
            cand_raw = _coerce_str_list(payload.get('raw'))
            cand_word = _coerce_str_list(payload.get('word'))
            cand_char = _coerce_str_list(payload.get('char'))
        else:
            raise ValueError('unexpected pickle format')
        cand_raw = _coerce_str_list(cand_raw)
        cand_word = _coerce_str_list(cand_word)
        cand_char = _coerce_str_list(cand_char)
        print(f"[cache] loaded candidates from {cache_path} (pickle)", flush=True)
        return cand_raw, cand_word, cand_char
    except Exception as exc:
        print(f"[cache][skip] {cache_path} not JSON/pickle; recomputing. ({exc})", flush=True)
        return None


def _save_candidate_cache(path_str: str, cand_raw: list[str], cand_word: list[str], cand_char: list[str]) -> None:
    cache_path = _safe_cache_path(path_str)
    if not cache_path:
        print(f"[cache][skip] candidate cache path {path_str} rejected; not writing.", flush=True)
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        suffix_combo = ''.join(cache_path.suffixes) or cache_path.suffix
        if suffix_combo in ('.pkl', '.pickle'):
            with cache_path.open('wb') as f:
                pickle.dump((cand_raw, cand_word, cand_char), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[cache] saved candidates to {cache_path} (pickle)", flush=True)
        elif suffix_combo.endswith('.gz'):
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump((cand_raw, cand_word, cand_char), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[cache] saved candidates to {cache_path} (pickle.gz)", flush=True)
        else:
            with cache_path.open('w', encoding='utf-8') as f:
                json.dump({'raw': cand_raw, 'word': cand_word, 'char': cand_char}, f, ensure_ascii=False)
            print(f"[cache] saved candidates to {cache_path} (json)", flush=True)
    except Exception as exc:
        print(f"[cache][warn] failed to write {cache_path}: {exc}", flush=True)

def _infer_embedding_dim(st_model, sample: Optional[torch.Tensor] = None) -> int:
    getter = getattr(st_model, 'get_sentence_embedding_dimension', None)
    if callable(getter):
        try:
            dim = int(getter())
            if dim > 0:
                return dim
        except Exception:
            pass
    if sample is not None:
        try:
            shape = getattr(sample, 'shape', None)
            if shape and len(shape) >= 1 and int(shape[-1]) > 0:
                return int(shape[-1])
        except Exception:
            pass
    return 512

def _model_signature(st_model) -> str:
    for attr in ('_model_card', 'model_card', 'model_dir', 'model_id', 'model_name', 'model_name_or_path'):
        value = getattr(st_model, attr, None)
        if isinstance(value, str) and value:
            return value
    tokenizer = getattr(st_model, 'tokenizer', None)
    name_or_path = getattr(tokenizer, 'name_or_path', None)
    if isinstance(name_or_path, str) and name_or_path:
        return name_or_path
    try:
        config = getattr(st_model, 'config', None)
        if config is not None:
            cfg_name = getattr(config, '_name_or_path', None)
            if isinstance(cfg_name, str) and cfg_name:
                return cfg_name
    except Exception:
        pass
    return repr(st_model)



def _embedding_cache_meta(st_model, *, normalize_embeddings: bool, emb_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    return {
        'model': _model_signature(st_model),
        'dim': _infer_embedding_dim(st_model, emb_tensor),
        'normalize_embeddings': bool(normalize_embeddings),
    }


@contextmanager
def _amp_autocast(device_hint: Any):
    device_type = None
    try:
        if isinstance(device_hint, torch.Tensor):
            device_type = device_hint.device.type
        elif isinstance(device_hint, torch.device):
            device_type = device_hint.type
        elif isinstance(device_hint, str):
            device_type = device_hint
        else:
            dev_attr = getattr(device_hint, 'device', None)
            if isinstance(dev_attr, torch.device):
                device_type = dev_attr.type
            else:
                device_type = getattr(dev_attr, 'type', None)
    except Exception:
        device_type = None
    if device_type != 'cuda' or not torch.cuda.is_available():
        yield
        return
    dtype_candidates: list[Any] = []
    bf16_checker = getattr(torch.cuda, 'is_bf16_supported', None)
    if callable(bf16_checker):
        try:
            if bf16_checker():
                dtype_candidates.append(torch.bfloat16)
        except Exception:
            pass
    dtype_candidates.append(torch.float16)
    for dtype in dtype_candidates:
        try:
            with torch.autocast(device_type='cuda', dtype=dtype):
                yield
                return
        except Exception:
            continue
    yield



def rebuild_bank_sketches() -> None:
    global BANK_SKETCHES
    bank = getattr(dc, "SOFT_PARAPHRASE_BANK", {}) or {}
    keys = list(bank.keys())
    BANK_SKETCHES = {
        k: [dc._sketch5(proto) for proto in dict.fromkeys(bank.get(k, []) or [])]
        for k in keys
    }


def rebuild_bank_embeddings(st_model, bs_try: int = 4096) -> None:
    global BANK_EMBS
    BANK_EMBS = {}
    device = getattr(st_model, "device", torch.device("cpu"))
    embed_dim = _infer_embedding_dim(st_model)
    for kind, phrases_raw in dc.SOFT_PARAPHRASE_BANK.items():
        phrases = list(dict.fromkeys(phrases_raw or []))
        if not phrases:
            BANK_EMBS[kind] = torch.empty((0, embed_dim), device=device)
            continue
        BANK_EMBS[kind] = encode_gpu(st_model, phrases, bs_try=bs_try)

rebuild_bank_sketches()



def _prioritize_kinds(kinds: Iterable[str]) -> list[str]:
    unique = {k for k in kinds if k}
    if not unique:
        return []
    return sorted(unique, key=lambda k: _PRIORITY_RANK.get(k, len(PRIORITY)))


def best_kind_by_bank_sim(phrase: str, candidate_kinds: list[str]) -> Optional[str]:
    if not candidate_kinds:
        return None
    v_phrase = dc._sketch5(phrase)
    best_kind: Optional[str] = None
    best_score = -1.0
    matched = False
    for kind in candidate_kinds:
        protos = BANK_SKETCHES.get(kind)
        if protos:
            matched = True
            score = max((dc._cos_sparse_local(v_phrase, proto) for proto in protos), default=0.0)
            if score > best_score:
                best_score = score
                best_kind = kind
    if matched:
        return best_kind if best_kind is not None and best_score >= MIN_BANK_SIM else None
    for kind in PRIORITY:
        if kind in candidate_kinds:
            return kind
    return candidate_kinds[0]

def best_kind_by_bank_sim_gpu(phrase: str, rep_emb: torch.Tensor, candidate_kinds: list[str]) -> Optional[str]:
    if not candidate_kinds:
        return None
    if rep_emb.ndim > 1:
        vec = rep_emb[0]
    else:
        vec = rep_emb
    best_kind: Optional[str] = None
    best_score = -1.0
    matched = False
    for kind in candidate_kinds:
        bank = BANK_EMBS.get(kind)
        if bank is None or bank.numel() == 0:
            continue
        matched = True
        vec_aligned = vec.to(bank.device)
        if bank.ndim == 1:
            score = torch.dot(bank, vec_aligned).item()
        else:
            score = torch.matmul(bank, vec_aligned).max().item()
        if score > best_score:
            best_score = score
            best_kind = kind
    if matched and best_kind is not None and best_score >= MIN_BANK_SIM:
        return best_kind
    if matched:
        return None
    return best_kind_by_bank_sim(phrase, candidate_kinds)

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


def label_cluster(rep: str, members: list[str], rep_emb: torch.Tensor) -> Optional[str]:
    kind = auto_route_kind(rep)
    if kind:
        return kind
    counter: dict[str, int] = defaultdict(int)
    for member in members:
        mkind = auto_route_kind(member)
        if mkind:
            counter[mkind] += 1
    if not counter:
        return best_kind_by_bank_sim_gpu(rep, rep_emb, list(BANK_EMBS.keys()))
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], _PRIORITY_RANK.get(kv[0], 10**9)))
    top_count = ranked[0][1]
    tied = [k for k, count in counter.items() if count == top_count]
    if len(tied) == 1:
        return tied[0]
    return best_kind_by_bank_sim_gpu(rep, rep_emb, tied)

def main(args):
    kind_override = (args.kind or "").strip() or None
    if kind_override and kind_override.lower() == "auto":
        kind_override = None
    args.kind = kind_override

    random.seed(args.seed)
    base = os.path.splitext(os.path.basename(args.input or ""))[0] or "input"
    global CAND_CACHE, PH_EMB_CACHE, BANK_SNAPSHOT
    CAND_CACHE = args.cand_cache or f"chk_candidates_{base}.pkl"
    PH_EMB_CACHE = args.emb_cache or f"chk_phrase_emb_{base}.pt"
    BANK_SNAPSHOT = args.bank_snapshot or f"bank_snapshot_{base}.json"

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
        rebuild_bank_sketches()
    docs_raw = read_lines(args.input)
    docs = []
    for doc in docs_raw:
        stripped = dc.strip_anchors(doc)
        if stripped:
            if len(stripped) > KEYBERT_MAX_DOC_CHARS:
                stripped = stripped[:KEYBERT_MAX_DOC_CHARS]
            docs.append(stripped)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)

    rebuild_bank_sketches()
    rebuild_bank_embeddings(st_model)


    # 1) KeyBERT ????MMR ???
    ngram_word = (args.ngram_min, args.ngram_max)
    ngram_char = (max(3, args.ngram_min), max(6, args.ngram_max))
    cand_cache_hit = False
    cached_candidates = _load_candidate_cache(CAND_CACHE)
    if cached_candidates is not None:
        cand_raw, cand_word, cand_char = cached_candidates
        cand_cache_hit = True
    if not cand_cache_hit:
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
        _save_candidate_cache(CAND_CACHE, cand_raw, cand_word, cand_char)
    cand_after_pos = cand_word

    combined_candidates = cand_word + cand_char

    def _passes_soft_gate(phrase: str) -> bool:
        score = _softish_score(phrase)
        if score >= args.soft_gate_thresh:
            return True
        return score >= SOFT_GATE_VERB_FALLBACK and _has_actionish_pos(phrase)

    cand_after_gate = [phrase for phrase in combined_candidates if _passes_soft_gate(phrase)]

    # 在 strong_dedupe 之前
    cand_after_gate = list(dict.fromkeys(cand_after_gate))

    cand = strong_dedupe(cand_after_gate,
                         simhash_bits=64, simhash_thresh=1,
                         k=5, n_hash=64, bands=16, jaccard_thresh=0.90,
                         vec_dim=1024, cosine_thresh=0.92)

    phrases_for_cluster = cand
    phrase_embs = None
    expected_meta = _embedding_cache_meta(st_model, normalize_embeddings=True)
    if os.path.exists(PH_EMB_CACHE):
        try:
            data = torch.load(PH_EMB_CACHE, map_location=st_model.device)
            meta = data.get('meta') if isinstance(data, dict) else None
            cached_phrases = data.get('phrases') if isinstance(data, dict) else None
            cached_embs = data.get('embs') if isinstance(data, dict) else None
            if meta and isinstance(meta, dict) and meta.get('model') == expected_meta['model'] and int(meta.get('dim', 0)) == expected_meta['dim'] and bool(meta.get('normalize_embeddings', True)) == expected_meta['normalize_embeddings'] and isinstance(cached_phrases, list) and cached_phrases == phrases_for_cluster and isinstance(cached_embs, torch.Tensor):
                phrase_embs = cached_embs.to(st_model.device)
                print(f"[cache] loaded phrase embeddings from {PH_EMB_CACHE}", flush=True)
            else:
                print(f"[cache][skip] ignored phrase embedding cache at {PH_EMB_CACHE} due to metadata mismatch", flush=True)
        except Exception:
            phrase_embs = None
    if phrase_embs is None:
        phrase_embs = encode_gpu(st_model, phrases_for_cluster, bs_try=4096)
        try:
            cache_payload = {
                'phrases': phrases_for_cluster,
                'embs': phrase_embs.cpu(),
                'meta': _embedding_cache_meta(st_model, normalize_embeddings=True, emb_tensor=phrase_embs),
            }
            torch.save(cache_payload, PH_EMB_CACHE)
            print(f"[cache] saved phrase embeddings to {PH_EMB_CACHE}", flush=True)
        except Exception:
            pass

    clusters = cluster_by_semantics(cand, st_model,
                                    thr=args.cluster_thr,
                                    min_size=args.cluster_min,
                                    precomputed_embs=phrase_embs)


    routing_mode = "auto"
    routing_summary: dict[str, object] = {}
    skipped_reps: list[str] = []
    coverage_inputs: List[dc.Proto] = []

    if args.kind:
        routing_mode = "manual"
        reps = list(dict.fromkeys(rep for rep, _, _ in clusters))
        refill_bank(args.kind, reps, max_add=0)
        if args.bank_all:
            all_members = []
            for _, members, _ in clusters:
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
        for rep, members, rep_emb in clusters:
            kind = label_cluster(rep, members, rep_emb)
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
    rebuild_bank_embeddings(st_model)
    rebuild_bank_sketches()
    coverage_report = None
    if coverage_inputs:
        coverage_report = dc.probe_soft_coverage(coverage_inputs, seed=args.seed, topk_show=3)

    # 杈撳嚭绠€鍗曠粺璁?
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

    try:
        with open(BANK_SNAPSHOT, "w", encoding="utf-8") as f:
            json.dump({k: list(dict.fromkeys(dc.SOFT_PARAPHRASE_BANK.get(k, []) or [])) for k in PRIORITY}, f, ensure_ascii=False, indent=2)
        print(f"[snapshot] saved bank to {BANK_SNAPSHOT}", flush=True)
    except Exception:
        pass

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
    p.add_argument("--cand_cache", type=str, default=None,
                   help="Path to cache KeyBERT candidates; defaults to per-input name.")
    p.add_argument("--emb_cache", type=str, default=None,
                   help="Path to cache phrase embeddings; defaults to per-input name.")
    p.add_argument("--bank_snapshot", type=str, default=None,
                   help="Path to write bank snapshot JSON; defaults to per-input name.")
    main(p.parse_args())

