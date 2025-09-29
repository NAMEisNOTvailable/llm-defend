# extract_and_bank_zh_phrases.py
# -*- coding: utf-8 -*-
import unicodedata
import os
import multiprocessing
import logging
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

os.environ.setdefault("NUMBA_NUM_THREADS", str(_n_cores))
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
try:
    torch.set_num_threads(_n_cores)
    torch.set_num_interop_threads(max(1, _n_cores // 2))
except Exception:
    pass
import json, re, gc, math, argparse, random, hashlib, inspect
import numpy as np
from contextlib import contextmanager
try:
    from stable_random import random_module_binding as _seed_ctx
except Exception:
    @contextmanager
    def _seed_ctx(tag, seed):
        import random
        try:
            import numpy as np
        except Exception:
            np = None
        state_py = random.getstate()
        np_state = None
        if np is not None:
            try:
                np_state = np.random.get_state()
            except Exception:
                np_state = None
        random.seed(seed)
        if np is not None:
            try:
                np.random.seed(seed % (2**32 - 1))
            except Exception:
                pass
        try:
            import torch as _torch_seed
            _torch_seed.manual_seed(seed)
        except Exception:
            pass
        try:
            yield
        finally:
            random.setstate(state_py)
            if np is not None and np_state is not None:
                try:
                    np.random.set_state(np_state)
                except Exception:
                    pass
import pickle
import gzip
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
from typing import Optional, Iterable, List, Any, Dict
from threading import Lock
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    import jieba_fast as jieba
except Exception:
    import jieba
try:
    import jieba_fast.posseg as pseg
except Exception:
    import jieba.posseg as pseg
from sklearn.feature_extraction.text import CountVectorizer
try:
    from FlagEmbedding import FlagReranker
except Exception:
    FlagReranker = None
try:
    import hanlp
except Exception:
    hanlp = None
try:
    import umap
except Exception:
    umap = None
try:
    import hdbscan
except Exception:
    hdbscan = None
_JIEBA_READY = False
_JIEBA_INIT_LOCK = Lock()

def _ensure_jieba_initialized() -> None:
    global _JIEBA_READY
    if _JIEBA_READY:
        return
    with _JIEBA_INIT_LOCK:
        if _JIEBA_READY:
            return
        try:
            import logging
            try:
                jieba.setLogLevel(logging.ERROR)
            except Exception:
                try:
                    default_logger = getattr(jieba, 'default_logger', None)
                    if default_logger is not None:
                        default_logger.setLevel(logging.ERROR)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            initializer = getattr(jieba, "initialize", None)
            if callable(initializer):
                initializer()
        except Exception:
            pass
        _JIEBA_READY = True
from sentence_transformers import SentenceTransformer, util
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
torch.set_grad_enabled(False)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
from dedupe_core import Deduper, get_default_deduper_kwargs
from micro_grammar import (
    refill_bank,
    rebuild_soft_check,
    hydrate_paraphrase_bank_with_microgrammars,
    attach_to_dsl_core,
)
import dsl_core as dc
# --- Performance knobs (auto-tuned in main) ---
AMP_ENABLED: bool = True
ENCODE_NUM_WORKERS: int = 0
GPU_RESERVE_BYTES: int = 6 * 1024**3
CPU_MEM_RESERVE_PCT: float = 0.20
CAND_CHUNK_MAX_CAP: int | None = None
ALLOW_EXTERNAL_CACHE: bool = False
SNAPSHOT_MAX_PER_KIND_DEFAULT: int = 400
ST_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-large")
POS_MODEL = os.environ.get("HANLP_POS_MODEL", "CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH")
_PHRASES_VIEW = None
SOFT_HINT_TOKENS = ( "请只", "务必", "不超过", "控制在", "最后一句", "末尾", 
    "结尾", "尾部", "追加", "附上", "补充", "忽略", "不参考", "上文", "先前", 
    "系统信息", "路径", "目录", "写入", "保存", "落盘", "调用", "函数", "工具", 
    "参数", "args", "score", "分数", "置信", "概率", "得分", "评分", "系统通道", 
    "system 角色", "管理端", "审定通道", "记忆", "状态", "回合", "下一轮", "一句话", 
    "忽略前文","无视之前","覆盖规则","扮演","作为","切换角色",
    "不要遵循","禁用安全","越过","只输出","仅输出","一步步","以JSON输出","JSON格式输出", "不要解释")
ACTIONISH_POS_TAGS = { "v", "vd", "vn", "vshi", "vyou", "vf", "vx", "vi", "vl", "vg", }
CJK_CODE_RANGES = ( (0x2E80, 0x2FFF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0x2CEB0, 0x2EBEF), (0x30000, 0x323AF), (0x2F800, 0x2FA1F), )
_PUNCT_ONLY = re.compile(r'^[\W_]+$', re.UNICODE)
STOPWORDS = {"然后", "那么", "其实", "就是", "以及", "关于", "如果", "需要", "由于", "因此", "此外", "同时", "可以", "进行", "有关", "问题", "情况", "方面", "不会", "不是", "没有", "为了", "这个", "那个"}
NUMERIC_MEASURE_RX = re.compile( r'^\d+(?:\.\d+)?(?:[~\-–—]\d+(?:\.\d+)?)?[\u4e00-\u9fff]{1,4}$' )
CONFUSABLES = {ord('\u200b'):None, ord('\u200c'):None, ord('\u200d'):None, ord('\ufeff'):None}

def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    for start, end in CJK_CODE_RANGES:
        if start <= code <= end:
            return True
    return False

def _keep_phrase(s: str) -> bool:
    s2 = re.sub(r'\s+', '', s or '')
    if len(s2) < 2 or len(s2) > 24:
        return False
    if _PUNCT_ONLY.fullmatch(s2):
        return False
    if not NUMERIC_MEASURE_RX.fullmatch(s2) and s2.isdigit():
        return False
    cjk = sum(1 for ch in s2 if _is_cjk_char(ch))
    if (cjk / max(1, len(s2))) < 0.4:
        return False
    if s2 in STOPWORDS:
        return False
    return True
# ===== High-precision harvesters for mechanism phrases =====
HARVEST_RX = {
    "ignore": re.compile(r"(忽略|无视|不要遵循|覆盖)(上文|之前|系统|规则|限制|政策)"),
    "append": re.compile(r"(?:在|到)?(最后|末尾|结尾).{0,4}(追加|附上|补充|输出|只输出)"),
    "role":   re.compile(r"(扮演|作为|以)(?:一名|一个)?([^，。；\s]{1,10})(?:身份|角色)"),
    "path":   re.compile(
        r"((保存|写入|输出).{0,6}(路径|目录|文件))|(/|\\|[A-Za-z]:\\|\.json\b|\.txt\b|\.csv\b)"
    ),
    "tool":   re.compile(r"(调用|运行|执行|使用).{0,8}(工具|函数|API|模型)(?!.*(语言|方法))"),
    "eval_hint": re.compile(r"(得分|评分|分数|评分标准|置信|概率|评分细则)"),
    "memory_write": re.compile(r"(记忆|存储|写入|保存)(?:到|至)?(系统|状态|内存|磁盘|文件)"),
    "no_explain": re.compile(r"(不要|无需|不必)(解释|说明|理由|分析)"),
    "json_out":  re.compile(r"(以|用)?JSON(格式)?(输出|返回)|输出为JSON"),
}


HARVEST_PASSTHRU_RX = [
    re.compile(r"^(忽略|无视|不要遵循|覆盖)(上文|之前|系统|规则|限制|政策)$"),
    re.compile(r"^末尾追加$"),
    re.compile(r"^保存到路径$"),
    re.compile(r"^调用工具$"),
    re.compile(r"^评分标准$"),
    re.compile(r"^写入(系统|状态|内存|磁盘|文件)$"),
    re.compile(r"^以.{1,10}身份$"),
    re.compile(r"^(以|用)?JSON(格式)?(输出|返回)$"),   
    re.compile(r"^(不要|无需|不必)(解释|说明|理由|分析)$"),  
]
def _norm_detect(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").translate(CONFUSABLES)
def _is_harvest_prototype(s: str) -> bool:
    s = _norm_detect(s)
    s = re.sub(r"\s+", "", s or "")
    for rx in HARVEST_PASSTHRU_RX:
        if rx.fullmatch(s):
            return True
    return False

def harvest_rule_based_phrases(text: str) -> list[str]:
    s = _norm_detect(text)
    s = re.sub(r"\s+", "", s)
    out: list[str] = []
    def add(ph: str):
        ph = (ph or "").strip()
        if ph and _keep_phrase(ph):
            out.append(ph)
    m = HARVEST_RX["ignore"].search(s)
    if m:
        add(m.group(1) + m.group(2))
    if HARVEST_RX["append"].search(s):
        add("末尾追加")
    m = HARVEST_RX["role"].search(s)
    if m:
        role = m.group(2)
        add(f"以{role}身份")
    if HARVEST_RX["tool"].search(s):
        add("调用工具")
    if HARVEST_RX["eval_hint"].search(s):
        add("评分标准")
    if HARVEST_RX["no_explain"].search(s):
        add("不要解释")
    if HARVEST_RX["json_out"].search(s):
        add("JSON格式输出")
    m = HARVEST_RX["memory_write"].search(s)
    m_path = HARVEST_RX["path"].search(s)
    if m_path:
        low = s.lower()
        if ("http://" not in low) and ("https://" not in low):
            add("保存到路径")
    if m:
        tgt = m.group(2)
        add(f"写入{tgt}")
    return list(dict.fromkeys(out))

def _ensure_reranker() -> Optional["FlagReranker"]:
    if FlagReranker is None:
        return None
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    with _RERANKER_LOCK:
        if _RERANKER is not None:
            return _RERANKER
        try:
            use_fp16 = torch.cuda.is_available()
            reranker = FlagReranker(RERANK_MODEL, use_fp16=use_fp16)
        except Exception:
            reranker = None
        _RERANKER = reranker
        return _RERANKER

def _rerank_kind_by_ce(phrase: str, candidate_kinds: list[str]) -> Optional[str]:
    if not candidate_kinds:
        return None
    rr = _ensure_reranker()
    if rr is None:
        return None
    pairs = []
    for kind in candidate_kinds:
        protos = dc.SOFT_PARAPHRASE_BANK.get(kind) or []
        snippets = list(dict.fromkeys(protos))[:5]
        right = f"[类别]{kind} [原型]{' '.join(snippets) or kind}"
        pairs.append([phrase, right])
    try:
        scores = rr.compute_score(pairs, normalize=True)
    except Exception:
        return None
    try:
        scores = list(scores)
    except Exception:
        return None
    if not scores:
        return None
    return candidate_kinds[int(np.argmax(scores))]

def _sanitize_bank_for_snapshot(
    bank: dict[str, list[str]] | None,
    *,
    thresh: float,
    max_per_kind: int | None,
    seed: int,
) -> dict[str, list[str]]:
    sanitized: dict[str, list[str]] = {}
    if not bank:
        return {k: [] for k in PRIORITY}
    base_seed = int(seed) & 0xFFFFFFFF
    enforce_gate = max(thresh, 0.0)
    for kind in PRIORITY:
        phrases = bank.get(kind) or []
        seen: set[str] = set()
        cleaned: list[str] = []
        for phrase in phrases:
            s = (phrase or '').strip()
            if not s or s in seen:
                continue
            if not _keep_phrase(s):
                continue
            try:
                if not _passes_soft_gate_threshold(s, enforce_gate):
                    continue
            except Exception:
                continue
            routed = auto_route_kind(s)
            if routed and routed != kind:
                continue
            protos = BANK_SKETCHES.get(kind)
            if protos:
                strict = best_kind_by_bank_sim(s, [kind], min_sim=MIN_BANK_SIM_SNAPSHOT)
                if strict != kind:
                    continue
            seen.add(s)
            cleaned.append(s)
        limit = max_per_kind if isinstance(max_per_kind, int) and max_per_kind > 0 else None
        if limit and len(cleaned) > limit:
            import random
            with _seed_ctx(f'snapshot_balance_{kind}', base_seed + _PRIORITY_RANK.get(kind, 0)):
                idx = sorted(random.sample(range(len(cleaned)), limit))
            cleaned = [cleaned[i] for i in idx]
        sanitized[kind] = cleaned
    return sanitized

def _ensure_hanlp():
    global _HANLP, _HANLP_MODULE_LOGGED, _HANLP_LOAD_LOGGED
    if hanlp is None:
        if not _HANLP_MODULE_LOGGED:
            print("[hanlp] module not available; POS tagging disabled.", flush=True)
            _HANLP_MODULE_LOGGED = True
        return None
    if _HANLP is not None:
        return _HANLP
    with _HANLP_LOCK:
        if _HANLP is not None:
            return _HANLP
        model_target = POS_MODEL
        try:
            model_target = getattr(hanlp.pretrained.mtl, POS_MODEL, POS_MODEL)
        except Exception:
            model_target = POS_MODEL
        model_label = str(model_target)
        try:
            _HANLP = hanlp.load(model_target)
        except Exception as exc:
            _HANLP = None
            if not _HANLP_LOAD_LOGGED:
                print(f"[hanlp] failed to load POS model {model_label}: {exc}", flush=True)
                _HANLP_LOAD_LOGGED = True
        else:
            if _HANLP is None and not _HANLP_LOAD_LOGGED:
                print(f"[hanlp] POS model {model_label} returned None; POS tagging disabled.", flush=True)
                _HANLP_LOAD_LOGGED = True
        return _HANLP

    
def _softish_score(text: str) -> float:
    """Return a heuristic soft-evidence score; >1 means strong semantic evidence via bank matches."""
    raw = text or ""
    s2 = _norm_detect(raw)
    s2 = re.sub(r"\s+", "", s2)
    if len(s2) < 3:
        return 0.0
    trig = sum(1 for token in SOFT_HINT_TOKENS if token in s2)
    if trig >= 2: return 0.9
    if trig == 1: return 0.6

    try:
        hits = 0
        check_map = getattr(dc, "SOFT_EVIDENCE_CHECK", {}) or {}
        for fn in check_map.values():
            try:
                if fn and fn(raw):   # ← 用原文
                    hits += 1
            except Exception:
                continue
        if hits > 0:
            return 1.0 + 0.1 * min(4, hits)
    except Exception:
        pass
    return 0.0

@lru_cache(maxsize=50000)
def _has_actionish_pos(text: str) -> bool:
    phrase = (text or "").strip()
    if not phrase:
        return False
    try:
        hanlp_tagger = _ensure_hanlp()
        if hanlp_tagger is not None:
            doc = hanlp_tagger(phrase)
            pos_tags = doc.get("pos") if isinstance(doc, dict) else getattr(doc, "pos", None)
            if pos_tags:
                return any(str(tag).startswith(("V", "v")) for tag in pos_tags)
    except Exception:
        pass
    try:
        for _, flag in pseg.lcut(phrase):
            if flag in ACTIONISH_POS_TAGS:
                return True
    except Exception:
        pass
    return False

def build_vectorizer(ngram=(2,4), stopwords=None, analyzer="char_wb", use_jieba=False):
    if use_jieba:
        _ensure_jieba_initialized()
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
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        is_cuda = torch.cuda.is_available()
        raw_env = os.environ.get('ENCODE_BS')
        try:
            env_bs = int(raw_env) if (raw_env is not None) else int(batch_size)
        except Exception:
            env_bs = int(batch_size)
        try:
            sig = inspect.signature(orig)
            params = set(sig.parameters.keys())
        except Exception:
            params = set()
        if 'num_workers' in params:
            kwargs['num_workers'] = max(0, int(ENCODE_NUM_WORKERS))
        kwargs.setdefault('show_progress_bar', False)
        if 'device' in params:
            kwargs.setdefault('device', st_model.device)
        candidates: list[int] = []
        def _add(size: int) -> None:
            if isinstance(size, int) and size > 0 and size not in candidates:
                candidates.append(size)
        if is_cuda:
            for size in (env_bs, 8192, 6144, 4096, 3072, 2048, 1536, 1024, 768, 512, 256):
                _add(size)
        else:
            cpu_seed = env_bs if isinstance(env_bs, int) and env_bs > 0 else 256
            for size in (min(cpu_seed, 512), 256, 128):
                _add(size)
        for bs in candidates:
            try:
                with torch.inference_mode(), _amp_autocast('cuda' if is_cuda else 'cpu'):
                    return orig(
                        texts,
                        batch_size=bs,
                        convert_to_tensor=convert_to_tensor,
                        normalize_embeddings=normalize_embeddings,
                        **kwargs,
                    )
            except (RuntimeError, TypeError, ValueError) as exc:
                msg = str(exc)
                if 'additional keyword arguments' in msg or 'got an unexpected keyword argument' in msg:
                    m = re.search(r"\[(.*?)\]", msg)
                    unknowns: list[str] = []
                    if m:
                        unknowns = [t.strip().strip("'\"") for t in m.group(1).split(',') if t.strip()]
                    for k in (unknowns or ['num_workers', 'device', 'show_progress_bar']):
                        if k in kwargs:
                            kwargs.pop(k, None)
                    continue
                continue
        fallback_bs = 128 if is_cuda else min(256, env_bs or 256)
        with torch.inference_mode(), _amp_autocast('cuda' if is_cuda else 'cpu'):
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

def _auto_encode_batch_size(st_model, default_bs: int, data_len: int) -> int:
    try:
        device = getattr(st_model, "device", torch.device("cpu"))
    except Exception:
        device = torch.device("cpu")
    target_len = max(1, int(data_len))
    base = max(1, int(default_bs))
    if device.type != 'cuda' or not torch.cuda.is_available():
        return min(base, target_len) if data_len else base
    try:
        with torch.cuda.device(device):
            free_mem, total_mem = torch.cuda.mem_get_info()
    except Exception:
        return min(base, target_len) if data_len else base
    reserve = max(int(0.05 * total_mem), 256 * 1024**2)
    headroom = max(0, int(free_mem) - reserve)
    if headroom <= 0:
        return min(base, target_len) if data_len else base
    try:
        sample_param = next(st_model.parameters())
        dtype = sample_param.dtype
    except Exception:
        dtype = torch.float32
    try:
        dtype_size = torch.tensor([], dtype=dtype).element_size()
    except Exception:
        dtype_size = 4
    embed_dim = _infer_embedding_dim(st_model) or 1024
    per_item_bytes = max(dtype_size * embed_dim * 6, dtype_size * 512)
    guess = headroom // per_item_bytes if per_item_bytes > 0 else base
    if guess <= 0:
        return min(base, target_len) if data_len else base
    guess = max(base, int(guess))
    guess = min(65536, guess)
    if data_len:
        return min(target_len, guess)
    return guess

def encode_gpu(
    st_model,
    texts,
    bs_try=1024,
    *,
    convert_to_tensor=True,
    normalize_embeddings=True,
    **kwargs,
):
    try:
        data_len = len(texts)  # type: ignore[arg-type]
    except Exception:
        data_len = 0
    base_bs = max(1, int(bs_try or 1024))
    auto_bs = _auto_encode_batch_size(st_model, base_bs, data_len or base_bs)
    eff_bs = max(1, int(auto_bs))
    with patched_encode(st_model, bs_try=eff_bs):
        return st_model.encode(
            texts,
            batch_size=eff_bs,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings,
            **kwargs,
        )

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]

def _bank_sim_scores_for(local_vecs: torch.Tensor, kinds: list[str]) -> Optional[torch.Tensor]:
    """??????? bank ?????????? kinds ?? max?????? bank ?? None?"""
    sims = None
    for k in kinds:
        bank = BANK_EMBS.get(k)
        if bank is None or bank.numel() == 0:
            continue
        sc = torch.matmul(local_vecs, bank.T).max(dim=1).values
        sims = sc if sims is None else torch.maximum(sims, sc)
    return sims

def _fused_rescore(
    doc_vec: torch.Tensor,
    local_vecs: torch.Tensor,
    texts: list[str],
    kinds: list[str],
    gamma: float = 0.35,
) -> list[str]:
    """???????doc??? + bank??????????gamma?[0,1]"""
    with torch.inference_mode():
        s_doc = util.cos_sim(doc_vec, local_vecs)[0]
    s_bank = _bank_sim_scores_for(local_vecs, kinds)
    if s_bank is None:
        order = torch.argsort(s_doc, descending=True).tolist()
        return [texts[i] for i in order]
    def _norm(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        mn, mx = float(x.min()), float(x.max())
        return (x - mn) / max(1e-8, (mx - mn))
    s = (1 - gamma) * _norm(s_doc) + gamma * _norm(s_bank)
    order = torch.argsort(s, descending=True).tolist()
    return [texts[i] for i in order]

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
    seed_by_doc: list[list[str]] = []
    def cjk_ratio(s: str) -> float:
        count = sum(1 for ch in s if _is_cjk_char(ch))
        return count / max(1, len(s))
    for doc in docs:
        seeds = harvest_rule_based_phrases(doc)
        seed_by_doc.append(seeds)
        if cjk_ratio(doc) < min_cjk:
            offsets.append((len(flat_cands), len(flat_cands)))
            continue
        if len(doc) > KEYBERT_MAX_DOC_CHARS:
            doc = doc[:KEYBERT_MAX_DOC_CHARS]
        w = [re.sub(r"\s+", "", t) for t in an_word(doc)]
        c = [re.sub(r"\s+", "", t) for t in an_char(doc)]
        w = list(dict.fromkeys([t for t in w if t]))
        w = [t for t in w if _keep_phrase(t)]
        c = list(dict.fromkeys([t for t in c if t]))
        c = [t for t in c if _keep_phrase(t)]
        if not c:
            c_plain = [re.sub(r"\s+", "", t) for t in an_char_plain(doc)]
            c = list(dict.fromkeys([t for t in c_plain if t]))
            c = [t for t in c if _keep_phrase(t)]
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
    if doc_embs.numel() and doc_embs.device != target_device:
        try:
            doc_embs = doc_embs.to(target_device)
        except Exception:
            pass
    embedding_dim = _infer_embedding_dim(st_model, doc_embs if doc_embs.numel() else None)
    bytes_per_value = doc_embs.element_size() if hasattr(doc_embs, "element_size") and doc_embs.numel() else 4
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
    seed_seen: set[str] = set()
    TOP_HARD = max(1, top_n)
    total_docs = len(docs)
    processed = [False] * total_docs
    pos = 0
    while pos < total_docs and offsets[pos][0] == offsets[pos][1]:
        processed[pos] = True
        pos += 1
    total_cands = len(flat_cands)
    deferred_docs: set[int] = set()
    if total_cands:
        # initial large cap; refine based on available resources
        chunk_size = 800_000
        chunk_budget_source = None
        is_cuda = torch.cuda.is_available()
        def _encode_cand_slice(slice_texts, bs_first=4096):
            nonlocal chunk_size
            try_sizes = [bs_first, 3072, 2048, 1536, 1024]
            last_exc = None
            for bs in try_sizes:
                try:
                    return encode_gpu(st_model, slice_texts, bs_try=bs)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        last_exc = e
                        chunk_size = max(int(chunk_size * 0.6), 1024)  # 同步收缩 chunk
                        continue
                    else:
                        raise
            if last_exc:
                raise last_exc
        cap = CAND_CHUNK_MAX_CAP if CAND_CHUNK_MAX_CAP is not None else (50000 if is_cuda else 10000)
        def _memory_budget() -> tuple[Optional[int], Optional[str]]:
            if approx_bytes_per_vec <= 0:
                return None, None
            if is_cuda:
                try:
                    free_mem, _ = torch.cuda.mem_get_info()
                    headroom = max(0, int(free_mem) - int(GPU_RESERVE_BYTES))
                    if headroom > 0:
                        return max(2, headroom // approx_bytes_per_vec), 'cuda'
                except Exception:
                    pass
            else:
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    reserve = max(384 * 1024**2, int(CPU_MEM_RESERVE_PCT * mem.total))
                    headroom = max(0, mem.available - reserve)
                    if headroom > 0:
                        return max(2, headroom // approx_bytes_per_vec), 'cpu'
                except Exception:
                    pass
            return None, None
        def _rebudget_chunk(desired: int, current: int) -> tuple[int, str]:
            desired = max(1, min(int(desired), total_cands))
            desired = min(desired, cap)
            budget, source = _memory_budget()
            if budget is not None:
                desired = min(desired, budget)
            if desired < current:
                return current, 'prev'
            return desired, (source or 'cap')
        chunk_size, chunk_budget_source = _rebudget_chunk(chunk_size, 1)
        print(f"[perf] candidate encode chunk_size={chunk_size} source={chunk_budget_source}", flush=True)
        chunk_start = 0
        while chunk_start < total_cands:
            chunk_end = min(chunk_start + chunk_size, total_cands)
            cand_slice = flat_cands[chunk_start:chunk_end]
            print(f"[stage] encode candidates chunk [{chunk_start}:{chunk_end}) on {st_model.device}", flush=True)
            cand_embs_chunk = _encode_cand_slice(cand_slice, bs_first=4096)
            if doc_embs.device.type != cand_embs_chunk.device.type:
                cand_embs_chunk = cand_embs_chunk.to(doc_embs.device)
                if cand_embs_chunk.device.type == "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            processed_all = True
            pending_idx: Optional[int] = None
            while pos < total_docs:
                ds, de = offsets[pos]
                doc_idx = pos
                if ds == de:
                    for seed in seed_by_doc[doc_idx]:
                        if seed and seed not in seed_seen:
                            word_candidates.append(seed)
                            seed_seen.add(seed)
                    processed[doc_idx] = True
                    pos += 1
                    continue
                if ds < chunk_start:
                    if not processed[doc_idx]:
                        deferred_docs.add(doc_idx)
                    pos += 1
                    continue
                if ds >= chunk_end:
                    break
                if de > chunk_end:
                    span = de - ds
                    if span > chunk_size:
                        new_chunk_size, new_source = _rebudget_chunk(span, chunk_size)
                        if new_chunk_size != chunk_size and new_chunk_size >= span:
                            print(f"[perf] rebudget chunk_size={new_chunk_size} (pending={span}) source={new_source}", flush=True)
                            chunk_size = new_chunk_size
                            chunk_budget_source = new_source
                            processed_all = False
                            pending_idx = pos
                            break
                        deferred_docs.add(doc_idx)
                        processed[doc_idx] = False
                        pos += 1
                        continue
                    processed_all = False
                    pending_idx = pos
                    break
                local_vecs = cand_embs_chunk[ds - chunk_start:de - chunk_start]
                local_texts = flat_cands[ds:de]
                doc_vec = doc_embs[doc_idx:doc_idx + 1]
                if doc_vec.device != local_vecs.device:
                    doc_vec = doc_vec.to(local_vecs.device)
                if use_mmr:
                    picked = mmr_select(doc_vec, local_vecs, local_texts, TOP_HARD, diversity)
                else:
                    ctx = _amp_autocast(doc_vec)
                    with torch.inference_mode():
                        with ctx:
                            sims = util.cos_sim(doc_vec, local_vecs)[0]
                            topk = torch.topk(sims, k=min(TOP_HARD, len(local_texts))).indices.tolist()
                    picked = [local_texts[j] for j in topk]
                seeds = [s for s in seed_by_doc[doc_idx] if _keep_phrase(s)]
                if seeds:
                    picked = list(dict.fromkeys(seeds + picked))
                doc_text = _norm_detect(docs[doc_idx])  # / idx 路径同理
                kinds_present = list(set(dc.soft_evidence_kinds(doc_text) or []))
                if kinds_present:
                    idx_map = {t: i for i, t in enumerate(local_texts)}
                    remain = [t for t in local_texts if t not in picked]
                    if remain:
                        remain_idx = torch.tensor([idx_map[t] for t in remain], device=local_vecs.device)
                        remain_vecs = local_vecs.index_select(0, remain_idx)
                        s_bank = _bank_sim_scores_for(remain_vecs, kinds_present)
                        if s_bank is not None:
                            topk = min(len(remain), max(1, len(kinds_present)))
                            cand_idx = torch.topk(s_bank, k=topk).indices.tolist()
                            extras = [remain[i] for i in cand_idx]
                            picked = list(dict.fromkeys(picked + extras))
                # ---- C. ??????? ----
                idx_map = {t: i for i, t in enumerate(local_texts)}
                sel_idx = torch.tensor([idx_map[t] for t in picked if t in idx_map], device=local_vecs.device)
                if sel_idx.numel():
                    sel_vecs = local_vecs.index_select(0, sel_idx)
                    picked = _fused_rescore(
                        doc_vec,
                        sel_vecs,
                        [t for t in picked if t in idx_map],
                        kinds_present,
                        gamma=0.35,
                    )
                picked = picked[:TOP_HARD]
                picked_set = set(picked)
                for p in picked:
                    if _has_actionish_pos(p):
                        word_candidates.append(p)
                    else:
                        char_candidates.append(p)
                for seed in seed_by_doc[doc_idx]:
                    if not seed:
                        continue
                    if seed in picked_set:
                        seed_seen.add(seed)
                        continue
                    if seed not in seed_seen:
                        word_candidates.append(seed)
                        seed_seen.add(seed)
                processed[doc_idx] = True
                pos += 1
            del cand_embs_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not processed_all and pending_idx is not None and pending_idx < total_docs:
                ds_pending, _ = offsets[pending_idx]
                chunk_start = ds_pending
            else:
                chunk_start = chunk_end
            while pos < total_docs and offsets[pos][0] == offsets[pos][1]:
                doc_idx = pos
                for seed in seed_by_doc[doc_idx]:
                    if seed and seed not in seed_seen:
                        word_candidates.append(seed)
                        seed_seen.add(seed)
                processed[doc_idx] = True
                pos += 1
    else:
        chunk_size = 0
    for idx in range(total_docs):
        if processed[idx] and idx not in deferred_docs:
            continue
        ds, de = offsets[idx]
        if ds == de:
            for seed in seed_by_doc[idx]:
                if seed and seed not in seed_seen:
                    word_candidates.append(seed)
                    seed_seen.add(seed)
            processed[idx] = True
            continue
        local_texts = flat_cands[ds:de]
        local_vecs = encode_gpu(st_model, local_texts, bs_try=4096)
        if doc_embs.device.type != local_vecs.device.type:
            local_vecs = local_vecs.to(doc_embs.device)
            if local_vecs.device.type == "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        doc_vec = doc_embs[idx:idx + 1]
        if doc_vec.device != local_vecs.device:
            doc_vec = doc_vec.to(local_vecs.device)
        if use_mmr:
            picked = mmr_select(doc_vec, local_vecs, local_texts, TOP_HARD, diversity)
        else:
            ctx = _amp_autocast(doc_vec)
            with torch.inference_mode():
                with ctx:
                    sims = util.cos_sim(doc_vec, local_vecs)[0]
                    topk = torch.topk(sims, k=min(TOP_HARD, len(local_texts))).indices.tolist()
            picked = [local_texts[j] for j in topk]
        seeds = [s for s in seed_by_doc[idx] if _keep_phrase(s)]
        if seeds:
            picked = list(dict.fromkeys(seeds + picked))
        doc_text = _norm_detect(docs[idx])
        kinds_present = list(set(dc.soft_evidence_kinds(doc_text) or []))
        if kinds_present:
            idx_map = {t: i for i, t in enumerate(local_texts)}
            remain = [t for t in local_texts if t not in picked]
            if remain:
                remain_idx = torch.tensor([idx_map[t] for t in remain], device=local_vecs.device)
                remain_vecs = local_vecs.index_select(0, remain_idx)
                s_bank = _bank_sim_scores_for(remain_vecs, kinds_present)
                if s_bank is not None:
                    topk = min(len(remain), max(1, len(kinds_present)))
                    cand_idx = torch.topk(s_bank, k=topk).indices.tolist()
                    extras = [remain[i] for i in cand_idx]
                    picked = list(dict.fromkeys(picked + extras))
        # ---- C. ??????? ----
        idx_map = {t: i for i, t in enumerate(local_texts)}
        sel_idx = torch.tensor([idx_map[t] for t in picked if t in idx_map], device=local_vecs.device)
        if sel_idx.numel():
            sel_vecs = local_vecs.index_select(0, sel_idx)
            picked = _fused_rescore(
                doc_vec,
                sel_vecs,
                [t for t in picked if t in idx_map],
                kinds_present,
                gamma=0.35,
            )
        picked = picked[:TOP_HARD]
        picked_set = set(picked)
        for p in picked:
            if _has_actionish_pos(p):
                word_candidates.append(p)
            else:
                char_candidates.append(p)
        for seed in seed_by_doc[idx]:
            if not seed:
                continue
            if seed in picked_set:
                seed_seen.add(seed)
                continue
            if seed not in seed_seen:
                word_candidates.append(seed)
                seed_seen.add(seed)
        processed[idx] = True
        del local_vecs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del doc_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[stage] keybert_candidates done | docs={len(docs)} | flat_cands={len(flat_cands)}", flush=True)
    return raw_all, word_candidates, char_candidates


def _keybert_worker_job(payload):
    docs, model_name, device_str, params, seed = payload
    top_n, ngram_word, ngram_char, use_mmr, diversity, min_cjk = params
    resolved_device = device_str
    if isinstance(resolved_device, str) and resolved_device.startswith('cuda') and not torch.cuda.is_available():
        resolved_device = 'cpu'
    with _seed_ctx('keybert_candidates', seed):
        st_model = SentenceTransformer(model_name, device=resolved_device)
        st_model.eval()
        rebuild_bank_sketches()
        rebuild_bank_embeddings(st_model, bs_try=4096)
        print(f"[stage] keybert_spawn model={model_name} device={st_model.device}", flush=True)
        return keybert_candidates(
            docs,
            st_model,
            top_n=top_n,
            ngram_word=ngram_word,
            ngram_char=ngram_char,
            use_mmr=use_mmr,
            diversity=diversity,
            min_cjk=min_cjk,
        )

def _generate_keybert_candidates(docs, *, model_name: str, device: str, params: dict, seed: int) -> tuple[list[str], list[str], list[str]]:
    if not docs:
        return [], [], []
    payload = (docs, model_name, device, (
        int(params.get('top_n', 20)),
        tuple(params.get('ngram_word', (2, 6))),
        tuple(params.get('ngram_char', (3, 6))),
        bool(params.get('use_mmr', True)),
        float(params.get('diversity', 0.5)),
        float(params.get('min_cjk', 0.3)),
    ), int(seed))
    try:
        ctx = multiprocessing.get_context('spawn')
    except Exception:
        ctx = multiprocessing.get_context()
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        future = ex.submit(_keybert_worker_job, payload)
        return future.result()

def strong_dedupe(phrases, simhash_bits=64, simhash_thresh=1,
                  k=5, n_hash=64, bands=16, jaccard_thresh=0.90,
                  vec_dim=1024, cosine_thresh=0.92):
    cfg = get_default_deduper_kwargs(sim_bits=simhash_bits,
                                     sim_thresh=simhash_thresh,
                                     k=k, n_hash=n_hash, bands=bands,
                                     jaccard_thresh=jaccard_thresh,
                                     vec_dim=vec_dim, cosine_thresh=cosine_thresh)
    dd = Deduper(**cfg)
    uniq: list[str] = []
    dd_probe = dd.probe
    dd_add = dd.add_record
    u_append = uniq.append
    total = len(phrases)
    for i, ph in enumerate(phrases):
        try:
            ok, _, rec = dd_probe(ph)
        except Exception:
            # [dedupe][warn] Optional diagnostic: print the offending sample when needed.
            continue
        if ok:
            dd_add(rec)
            u_append(ph)
        if (i + 1) % 200000 == 0:
            kept = len(uniq)
            print(f"[dedupe] seen={i+1:,} kept={kept:,} keep_rate={kept/(i+1):.3f}", flush=True)
    kept = len(uniq)
    if total:
        print(f"[dedupe] seen={total:,} kept={kept:,} keep_rate={kept/total:.3f}", flush=True)
    else:
        print("[dedupe] seen=0 kept=0 keep_rate=0.000", flush=True)
    return uniq

def _mp_init(numba_threads: int, omp_threads: int, phrases_ref=None) -> None:
    """Initializer used by multiprocessing workers to bound intra-process threading."""
    import os
    global _PHRASES_VIEW
    numba_threads = max(1, int(numba_threads))
    omp_threads = max(1, int(omp_threads))
    os.environ["NUMBA_NUM_THREADS"] = str(numba_threads)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(omp_threads)
    try:
        import numba as _nb
        _nb.set_num_threads(numba_threads)
    except Exception:
        pass
    try:
        import numexpr as _ne
        _ne.set_num_threads(omp_threads)
    except Exception:
        pass
    if phrases_ref is not None:
        _PHRASES_VIEW = phrases_ref

def _dedupe_shard_worker_range(payload):
    """按 [start, end) 区间从全局视图取文本，严格顺序去重。"""
    if len(payload) == 2:
        (start, end), cfg = payload
        chunk = None
    else:
        (start, end), cfg, chunk = payload
    if chunk is None:
        if _PHRASES_VIEW is None:
            raise RuntimeError("[dedupe][mp] phrases view not initialized in worker")
        iterator = ((idx, _PHRASES_VIEW[idx]) for idx in range(start, end))
    else:
        iterator = ((start + offset, text) for offset, text in enumerate(chunk))
    dd = Deduper(**cfg)
    kept: list[tuple[int, str]] = []
    dd_probe = dd.probe
    dd_add = dd.add_record
    for idx, ph in iterator:
        try:
            ok, _, rec = dd_probe(ph)
        except Exception:
            continue
        if ok:
            dd_add(rec)
            kept.append((idx, ph))
    return kept

def _passes_soft_gate_threshold(phrase: str, thresh: float) -> bool:
    if _is_harvest_prototype(phrase):
        return True
    score = _softish_score(phrase)
    return (score >= thresh) or (score >= SOFT_GATE_VERB_FALLBACK and _has_actionish_pos(phrase))

def _soft_gate_worker_range(payload):
    """payload may include inline phrase data to avoid copying the full list under spawn."""
    if len(payload) == 2:
        (start, end), thresh = payload
        chunk = None
    else:
        (start, end), thresh, chunk = payload
    if chunk is None:
        if _PHRASES_VIEW is None:
            raise RuntimeError("[gate][mp] phrases view not initialized in worker")
        phrases = _PHRASES_VIEW[start:end]
    else:
        phrases = chunk
    _ensure_jieba_initialized()  # 每个进程初始�?jieba
    passed = []
    for offset, s in enumerate(phrases):
        idx = start + offset
        try:
            if _passes_soft_gate_threshold(s, thresh):
                passed.append((idx, s))
        except Exception:
            continue
    return passed

def _resolve_parallel_workers(workers_spec,
                              *,
                              cap=None,
                              reserve_cores: int = 2) -> tuple[int, int, int, int]:
    cores = os.cpu_count() or 4
    max_reserve = max(cores - 1, 0)
    reserve = min(max(reserve_cores, 0), max_reserve)
    usable = max(1, cores - reserve)
    if cap is not None:
        try:
            cap_val = int(cap)
        except Exception:
            cap_val = None
        else:
            if cap_val and cap_val > 0:
                usable = max(1, min(usable, cap_val))
    if isinstance(workers_spec, str):
        if workers_spec.lower() == "auto":
            resolved = usable
        else:
            try:
                resolved = int(workers_spec)
            except Exception:
                resolved = 1
    else:
        try:
            resolved = int(workers_spec)
        except Exception:
            resolved = 1
    resolved = max(1, min(usable, resolved))
    per_proc_numba = max(1, usable // resolved)
    per_proc_omp = 1
    return resolved, usable, per_proc_numba, per_proc_omp

def soft_gate_parallel(candidates: list[str],
                       *,
                       thresh: float,
                       workers="auto",
                       chunk=200_000,
                       log_every=1_000_000,
                       cap=None) -> list[str]:
    """Parallel variant of _passes_soft_gate that boosts throughput only."""
    n = len(candidates)
    if n == 0:
        return []
    chunk = max(1, int(chunk))
    workers, _, per_proc_numba, per_proc_omp = _resolve_parallel_workers(
        workers,
        cap=cap,
    )
    if workers <= 1 or chunk >= n:
        return [phrase for phrase in candidates if _passes_soft_gate_threshold(phrase, thresh)]
    try:
        w = int(workers)
    except Exception:
        w = 1
    if n > 0 and w > 0:
        target_shards = max(w * 2, w + 2)     # 你要更激进可设 w*3
        est_shards = max(1, math.ceil(n / max(1, chunk)))
        if est_shards < target_shards:
            chunk = max(50_000, int(math.ceil(n / target_shards)))
            print(f"[gate][mp] autotune chunk→{chunk} (n={n}, workers={w}, target_shards={target_shards})",
                  flush=True)
    ranges = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    kept = []
    next_log = int(log_every) if log_every else 0
    try:
        try:
            ctx = multiprocessing.get_context("fork")
        except Exception:
            ctx = multiprocessing.get_context()
        try:
            start_method = ctx.get_start_method()
        except AttributeError:
            start_method = getattr(ctx, "_name", "fork")
        start_method = (start_method or "fork").lower()
        share_via_global = start_method == "fork"
        global _PHRASES_VIEW
        _PHRASES_VIEW = candidates if share_via_global else None
        task_payloads = []
        for r in ranges:
            if share_via_global:
                task_payloads.append((r, thresh))
            else:
                start_idx, end_idx = r
                task_payloads.append((r, thresh, candidates[start_idx:end_idx]))
        print(f"[gate][mp] start {len(ranges)} shards, workers={workers}, "
              f"chunk≈{chunk}, thresh={thresh}", flush=True)
        print(f"[gate][mp] workers={workers} | per_proc_numba={per_proc_numba} | per_proc_omp={per_proc_omp}",
              flush=True)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_mp_init,
            initargs=(per_proc_numba, per_proc_omp, None),
        ) as ex:
            futs = [ex.submit(_soft_gate_worker_range, payload) for payload in task_payloads]
            total = len(futs)
            done = 0
            for fut in as_completed(futs):
                part = fut.result()
                kept.extend(part)
                done += 1
                if (next_log and len(kept) >= next_log) or (done % 2 == 0) or done == total:
                    print(f"[gate][mp] shard done {done}/{total} | pass_so_far={len(kept):,}", flush=True)
    finally:
        _PHRASES_VIEW = None
    kept.sort(key=lambda t: t[0])
    out = [s for _, s in kept]
    return out

def strong_dedupe_parallel(phrases: list[str],
                           *,
                           simhash_bits=64, simhash_thresh=1,
                           k=5, n_hash=64, bands=16, jaccard_thresh=0.90,
                           vec_dim=1024, cosine_thresh=0.92,
                           workers="auto",
                           worker_cap=None,
                           shard_size=500_000, log_every=200_000) -> list[str]:
    """Map-Reduce 并行强去重；分片去重 + 汇总再做一次全局强去重。"""
    n = len(phrases)
    if n == 0:
        return []
    shard_size = max(1, int(shard_size))
    log_every = max(0, int(log_every)) if log_every is not None else 0
    workers, _, per_proc_numba, per_proc_omp = _resolve_parallel_workers(
        workers,
        cap=worker_cap,
    )
    if workers <= 1 or n <= shard_size * 1.2:
        return strong_dedupe(phrases,
                             simhash_bits=simhash_bits, simhash_thresh=simhash_thresh,
                             k=k, n_hash=n_hash, bands=bands, jaccard_thresh=jaccard_thresh,
                             vec_dim=vec_dim, cosine_thresh=cosine_thresh)
    cfg = get_default_deduper_kwargs(sim_bits=simhash_bits,
                                     sim_thresh=simhash_thresh,
                                     k=k, n_hash=n_hash, bands=bands,
                                     jaccard_thresh=jaccard_thresh,
                                     vec_dim=vec_dim, cosine_thresh=cosine_thresh)
    try:
        w = int(workers)
    except Exception:
        w = 1
    if n > 0 and w > 0:
        target_shards = max(w * 3, w + 4)
        est_shards = max(1, math.ceil(n / max(1, shard_size)))
        if est_shards < target_shards:
            shard_size = max(100_000, int(math.ceil(n / target_shards)))
            print(f"[dedupe][mp] autotune shard_size→{shard_size} (n={n}, workers={w}, target_shards={target_shards})",
                  flush=True)
    ranges = [(i, min(i + shard_size, n)) for i in range(0, n, shard_size)]
    kept_all: list[tuple[int, str]] = []
    next_log = log_every
    try:
        try:
            ctx = multiprocessing.get_context("fork")
        except Exception:
            ctx = multiprocessing.get_context()
        try:
            start_method = ctx.get_start_method()
        except AttributeError:
            start_method = getattr(ctx, "_name", "fork")
        start_method = (start_method or "fork").lower()
        share_via_global = start_method == "fork"
        global _PHRASES_VIEW
        _PHRASES_VIEW = phrases if share_via_global else None
        shard_payloads = []
        for r in ranges:
            if share_via_global:
                shard_payloads.append((r, cfg))
            else:
                start_idx, end_idx = r
                shard_payloads.append((r, cfg, phrases[start_idx:end_idx]))
        print(f"[dedupe][mp] start {len(ranges)} shards, workers={workers}, shard_size≈{shard_size}", flush=True)
        print(f"[dedupe][mp] workers={workers} | per_proc_numba={per_proc_numba} | per_proc_omp={per_proc_omp}", flush=True)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_mp_init,
            initargs=(per_proc_numba, per_proc_omp, None),
        ) as ex:
            futures = [ex.submit(_dedupe_shard_worker_range, payload) for payload in shard_payloads]
            total = len(futures)
            completed = 0
            for fut in as_completed(futures):
                part = fut.result()
                kept_all.extend(part)
                completed += 1
                kept_count = len(kept_all)
                should_log = False
                if log_every and kept_count >= next_log:
                    should_log = True
                    next_log += log_every
                elif completed % 2 == 0 or completed == total:
                    should_log = True
                if should_log:
                    print(f"[dedupe][mp] shard done {completed}/{total} | kept_so_far={kept_count:,}", flush=True)
    finally:
        _PHRASES_VIEW = None
    kept_all.sort(key=lambda t: t[0])
    first_pass = [ph for _, ph in kept_all]
    print(f"[dedupe][mp] first_pass={len(first_pass):,}, now global strong_dedupe...", flush=True)
    final = strong_dedupe(first_pass,
                          simhash_bits=simhash_bits, simhash_thresh=simhash_thresh,
                          k=k, n_hash=n_hash, bands=bands, jaccard_thresh=jaccard_thresh,
                          vec_dim=vec_dim, cosine_thresh=cosine_thresh)
    print(f"[dedupe][mp] final={len(final):,}", flush=True)
    return final

def cluster_by_semantics(phrases, st_model, thr=0.85, min_size=2, precomputed_embs: Optional[torch.Tensor] = None):
    if not phrases:
        return []
    embs = precomputed_embs if precomputed_embs is not None else encode_gpu(st_model, phrases)
    if isinstance(embs, torch.Tensor):
        embs_tensor = embs.detach().float()
        X = embs_tensor.cpu().numpy()
    else:
        embs_tensor = torch.as_tensor(embs)
        X = embs_tensor.cpu().numpy()
    use_hdbscan = (hdbscan is not None) and (umap is not None)
    if use_hdbscan:
        try:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X_norm = X / norms
            X_2d = umap.UMAP(n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42).fit_transform(X_norm)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, min_size), metric="euclidean")
            labels = clusterer.fit_predict(X_2d)
        except Exception:
            use_hdbscan = False
    if not use_hdbscan:
        embs_cpu = embs_tensor.to("cpu")
        if not isinstance(embs_cpu, torch.Tensor):
            embs_cpu = torch.as_tensor(embs_cpu)
        with torch.inference_mode():
            clusters = util.community_detection(embs_cpu, threshold=thr, min_community_size=min_size)
        results = []
        with torch.inference_mode():
            for idxs in clusters:
                members = [phrases[i] for i in idxs]
                sub = embs_cpu[idxs].float()
                sim = util.cos_sim(sub, sub).mean(dim=1)
                rep_idx = int(sim.argmax())
                rep = members[rep_idx]
                rep_vec = sub[rep_idx].clone()
                results.append((rep, members, rep_vec))
        if min_size > 1:
            clustered = set(i for c in clusters for i in c)
            for i, ph in enumerate(phrases):
                if i not in clustered:
                    results.append((ph, [ph], embs_cpu[i].float().clone()))
        return results
    cluster_map: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if int(label) == -1:
            continue
        cluster_map[int(label)].append(idx)
    results = []
    for idxs in cluster_map.values():
        members = [phrases[i] for i in idxs]
        sub = embs_tensor[idxs]
        sim = util.cos_sim(sub, sub).mean(dim=1)
        rep_idx = int(sim.argmax())
        rep = members[rep_idx]
        rep_vec = sub[rep_idx].clone()
        results.append((rep, members, rep_vec))
    noise = [i for i, label in enumerate(labels) if int(label) == -1]
    for i in noise:
        results.append((phrases[i], [phrases[i]], embs_tensor[i].clone()))
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
SOFT_GATE_VERB_FALLBACK = 0.45
MIN_BANK_SIM_ROUTE = 0.62
MIN_BANK_SIM_SNAPSHOT = 0.68
BANK_SKETCHES: dict[str, list] = {}
BANK_EMBS: dict[str, torch.Tensor] = {}
CAND_CACHE = "chk_candidates.json"
PH_EMB_CACHE = "chk_phrase_emb.pt"
BANK_SNAPSHOT = "bank_snapshot.json"
_RERANKER = None
_RERANKER_LOCK = Lock()
_HANLP = None
_HANLP_LOCK = Lock()
_HANLP_MODULE_LOGGED = False
_HANLP_LOAD_LOGGED = False
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
    if not ALLOW_EXTERNAL_CACHE:
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
        # suffix_combo 可能�?".pkl.gz"；endswith('.gz') 会命�?gzip
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

def _list_fingerprint(lst: list[str]) -> str:
    h = hashlib.sha1()
    items = [s if isinstance(s, str) else str(s) for s in lst]
    for entry in sorted(items):
        h.update(entry.encode('utf-8', 'ignore'))
        h.update(b"\0")
    return h.hexdigest()

def _embedding_cache_meta(st_model, *, normalize_embeddings: bool, emb_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    return {
        'model': _model_signature(st_model),
        'dim': _infer_embedding_dim(st_model, emb_tensor),
        'normalize_embeddings': bool(normalize_embeddings),
    }
@contextmanager

def _amp_autocast(device_hint: Any):
    if not AMP_ENABLED:
        yield
        return
    device_type = None
    try:
        if isinstance(device_hint, torch.Tensor):
            device_type = device_hint.device.type
        elif isinstance(device_hint, torch.device):
            device_type = device_hint.type
        elif isinstance(device_hint, str):
            device_type = device_hint
        else:
            dev = getattr(device_hint, 'device', None)
            if isinstance(dev, torch.device):
                device_type = dev.type
    except Exception:
        device_type = None
    if device_type != 'cuda' or not torch.cuda.is_available():
        yield
        return
    dtype = torch.bfloat16 if getattr(torch.cuda, 'is_bf16_supported', lambda: False)() else torch.float16
    with torch.autocast(device_type='cuda', dtype=dtype):
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

def best_kind_by_bank_sim(phrase: str, candidate_kinds: list[str], *, min_sim: float | None = None) -> Optional[str]:
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
        thr = MIN_BANK_SIM_ROUTE if min_sim is None else float(min_sim)
        return best_kind if best_kind is not None and best_score >= thr else None
    for kind in PRIORITY:
        if kind in candidate_kinds:
            return kind
    return candidate_kinds[0]

def best_kind_by_bank_sim_gpu(phrase: str, rep_emb: torch.Tensor, candidate_kinds: list[str], *, min_sim: float | None = None) -> Optional[str]:
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
    thr = MIN_BANK_SIM_ROUTE if min_sim is None else float(min_sim)
    if matched and best_kind is not None and best_score >= thr:
        return best_kind
    if matched:
        return None
    return best_kind_by_bank_sim(phrase, candidate_kinds, min_sim=min_sim)

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
    if not ordered:
        return None
    if len(ordered) > 1:
        return best_kind_by_bank_sim(phrase, ordered)
    single_kind = ordered[0]
    routed = best_kind_by_bank_sim(phrase, [single_kind])
    return routed

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
        priority_candidates = [k for k in PRIORITY if k in BANK_EMBS]
        candidate_kinds = priority_candidates or list(BANK_EMBS.keys())
        guess = best_kind_by_bank_sim_gpu(rep, rep_emb, candidate_kinds, min_sim=MIN_BANK_SIM_ROUTE)
        if guess:
            return guess
        reranked = _rerank_kind_by_ce(rep, candidate_kinds[:8])
        if reranked:
            return reranked
        return best_kind_by_bank_sim(rep, candidate_kinds, min_sim=MIN_BANK_SIM_ROUTE)
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], _PRIORITY_RANK.get(kv[0], 10**9)))
    top_count = ranked[0][1]
    tied = [k for k, count in counter.items() if count == top_count]
    if len(tied) == 1:
        return tied[0]
    reranked = _rerank_kind_by_ce(rep, tied[:8])
    if reranked:
        return reranked
    return best_kind_by_bank_sim_gpu(rep, rep_emb, tied, min_sim=MIN_BANK_SIM_ROUTE)

def _main_impl(args):
    kind_override = (args.kind or "").strip() or None
    if kind_override and kind_override.lower() == "auto":
        kind_override = None
    args.kind = kind_override
    # --- set perf knobs from args ---
    global AMP_ENABLED, ENCODE_NUM_WORKERS, GPU_RESERVE_BYTES
    global CPU_MEM_RESERVE_PCT, CAND_CHUNK_MAX_CAP, ALLOW_EXTERNAL_CACHE
    AMP_ENABLED = not args.no_amp
    ALLOW_EXTERNAL_CACHE = bool(args.allow_external_cache)
    try:
        import psutil
        n_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 4
    except Exception:
        n_cores = os.cpu_count() or 4
    reserve_cores = max(int(args.cpu_core_reserve), int(math.ceil(0.10 * n_cores)))
    use_cores = max(1, n_cores - reserve_cores)
    try:
        torch.set_num_threads(use_cores)
        torch.set_num_interop_threads(max(1, use_cores // 2))
    except Exception:
        pass
    for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_env_key] = str(use_cores)
    print(f"[perf] final threads: num_threads={use_cores}, interop={max(1, use_cores // 2)}", flush=True)
    def _auto_encode_workers_from_system():
        try:
            import psutil
            cores = psutil.cpu_count(logical=True) or os.cpu_count() or 4
            keep_idle = max(args.cpu_core_reserve, int(math.ceil(0.20 * cores)))
            return max(0, min(8, cores - keep_idle))
        except Exception:
            cores = os.cpu_count() or 4
            return max(0, min(4, cores - args.cpu_core_reserve))
    if isinstance(args.encode_workers, str) and args.encode_workers.lower() == "auto":
        ENCODE_NUM_WORKERS = _auto_encode_workers_from_system()
    else:
        try:
            ENCODE_NUM_WORKERS = max(0, int(args.encode_workers))
        except Exception:
            ENCODE_NUM_WORKERS = 0
    GPU_RESERVE_BYTES = int(max(0.5, float(args.gpu_mem_reserve_gb)) * (1024 ** 3))
    CPU_MEM_RESERVE_PCT = float(args.cpu_mem_reserve_pct)
    CAND_CHUNK_MAX_CAP = int(args.cand_chunk_max) if args.cand_chunk_max is not None else None
    print(f"[perf] cpu cores={n_cores}, use={use_cores}, encode_workers={ENCODE_NUM_WORKERS}", flush=True)
    print(f"[perf] gpu reserve={GPU_RESERVE_BYTES/(1024**3):.1f}GB, ram reserve={CPU_MEM_RESERVE_PCT*100:.0f}%, chunk_cap={CAND_CHUNK_MAX_CAP}", flush=True)
    base = os.path.splitext(os.path.basename(args.input or ""))[0] or "input"
    global CAND_CACHE, PH_EMB_CACHE, BANK_SNAPSHOT
    CAND_CACHE = args.cand_cache or f"chk_candidates_{base}.pkl"
    PH_EMB_CACHE = args.emb_cache or f"chk_phrase_emb_{base}.pt"
    BANK_SNAPSHOT = args.bank_snapshot or f"bank_snapshot_{base}.json"
    prewarm_keys = PREWARM_DEFAULT_KEYS
    if args.prewarm_keys:
        prewarm_keys = [k.strip() for k in args.prewarm_keys.split(",") if k.strip()]

    reuse_bank = False
    snap_path = _safe_cache_path(BANK_SNAPSHOT) if BANK_SNAPSHOT else None
    if args.reuse_bank_snapshot and snap_path and snap_path.exists() and not args.force_build_bank:
        try:
            with snap_path.open("r", encoding="utf-8") as f:
                snap = json.load(f) or {}
            for k, v in (snap.items() if isinstance(snap, dict) else []):
                if isinstance(v, list):
                    dc.SOFT_PARAPHRASE_BANK.setdefault(k, []).extend(s for s in v if isinstance(s, str))
            # 每类去重
            for k in list(dc.SOFT_PARAPHRASE_BANK):
                dc.SOFT_PARAPHRASE_BANK[k] = list(dict.fromkeys(dc.SOFT_PARAPHRASE_BANK[k]))
            # 基于快照重建索引/检查
            rebuild_soft_check()
            rebuild_bank_sketches()
            print(f"[bank] reused snapshot from {snap_path}; skip prewarm/hydrate", flush=True)
            reuse_bank = True
        except Exception as e:
            print(f"[bank][warn] reuse snapshot failed: {e}; fallback to build.", flush=True)

    # ---- 仅当未复用快照时，才执行 prewarm / hydrate ----
    prewarm_stats = None
    if (not reuse_bank) and (args.prewarm_per_kind > 0):
        prewarm_stats = attach_to_dsl_core(
            dc,
            keys=prewarm_keys,
            per_kind=args.prewarm_per_kind,
            seed=args.prewarm_seed,
            dedupe=not args.prewarm_allow_dupes,
        )

    hydration_stats = None
    if (not reuse_bank) and args.hydrate_microgrammars:
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
    def chunk_cn(text: str, size=KEYBERT_MAX_DOC_CHARS, stride=None):
        s = (text or "").strip()
        if not s:
            return []
        if stride is None:
            stride = max(128, size // 2)
        stride = max(1, stride)
        out: list[str] = []
        idx = 0
        while idx < len(s):
            seg = s[idx:idx + size]
            if not seg:
                break
            out.append(seg)
            if idx + size >= len(s):
                break
            idx += stride
        return out
    docs = []
    for doc in docs_raw:
        stripped = dc.strip_anchors(doc)
        if not stripped:
            continue
        for seg in chunk_cn(stripped, size=KEYBERT_MAX_DOC_CHARS, stride=KEYBERT_MAX_DOC_CHARS // 2):
            docs.append(seg)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model_name = ST_MODEL_NAME
    rebuild_bank_sketches()
    ngram_word = (args.ngram_min, args.ngram_max)
    ngram_char = (max(3, args.ngram_min), max(6, args.ngram_max))
    cand_cache_hit = False
    cached_candidates = _load_candidate_cache(CAND_CACHE)
    if cached_candidates is not None:
        cand_raw, cand_word, cand_char = cached_candidates
        cand_cache_hit = True
    if not cand_cache_hit:
        cand_raw, cand_word, cand_char = _generate_keybert_candidates(
            docs,
            model_name=model_name,
            device=device,
            params={
                'top_n': args.top_n,
                'ngram_word': ngram_word,
                'ngram_char': ngram_char,
                'use_mmr': True,
                'diversity': args.diversity,
                'min_cjk': args.min_cjk,
            },
            seed=args.seed,
        )
        _save_candidate_cache(CAND_CACHE, cand_raw, cand_word, cand_char)
    cand_word_filtered = cand_word
    # Refresh soft-evidence checks so hydrated banks influence gating early.
    rebuild_soft_check()
    combined_candidates = cand_word + cand_char
    try:
        _ensure_hanlp()
    except Exception:
        pass
    cand_after_gate = soft_gate_parallel(
        combined_candidates,
        thresh=args.soft_gate_thresh,
        workers=args.soft_gate_workers,
        chunk=args.soft_gate_chunk,
        cap=args.soft_gate_cap,
    )
    # 在强去重之前
    cand_after_gate = list(dict.fromkeys(cand_after_gate))
    phrase_embs = None
    cand = strong_dedupe_parallel(
        cand_after_gate,
        simhash_bits=64, simhash_thresh=1,
        k=5, n_hash=64, bands=16, jaccard_thresh=0.90,
        vec_dim=1024, cosine_thresh=0.92,
        workers=args.dedupe_workers,
        worker_cap=args.dedupe_worker_cap,
        shard_size=args.dedupe_shard
    )
    st_model = SentenceTransformer(model_name, device=device)
    st_model.eval()
    print(f"[info] model={model_name} device={st_model.device}", flush=True)
    rebuild_bank_embeddings(st_model)
    phrases_for_cluster = cand
    phrases_fp = _list_fingerprint(phrases_for_cluster)
    expected_meta = _embedding_cache_meta(st_model, normalize_embeddings=True)
    emb_path = _safe_cache_path(PH_EMB_CACHE)
    if emb_path and emb_path.exists():
        try:
            data = torch.load(emb_path, map_location=st_model.device)
            meta = data.get('meta') if isinstance(data, dict) else None
            cached_phrases = data.get('phrases') if isinstance(data, dict) else None
            cached_embs = data.get('embs') if isinstance(data, dict) else None
            cached_fp = None
            if isinstance(meta, dict):
                cached_fp = meta.get('phrases_fp')
            if cached_fp is None and isinstance(data, dict):
                cached_fp = data.get('phrases_fp')
            meta_ok = (
                meta
                and isinstance(meta, dict)
                and meta.get('model') == expected_meta['model']
                and int(meta.get('dim', 0)) == expected_meta['dim']
                and bool(meta.get('normalize_embeddings', True)) == expected_meta['normalize_embeddings']
            )
            if meta_ok and isinstance(cached_embs, torch.Tensor):
                exact_match = isinstance(cached_phrases, list) and cached_phrases == phrases_for_cluster
                fp_match = bool(cached_fp) and cached_fp == phrases_fp
                if exact_match or fp_match:
                    phrase_embs = cached_embs.to(st_model.device)
                    reason = 'exact' if exact_match else 'fingerprint'
                    print(f"[cache] loaded phrase embeddings from {emb_path} (match={reason})", flush=True)
                else:
                    print(f"[cache][skip] phrase embedding cache at {emb_path} ignored due to phrase mismatch", flush=True)
            else:
                print(f"[cache][skip] ignored phrase embedding cache at {emb_path} due to metadata mismatch", flush=True)
        except Exception:
            phrase_embs = None
    elif PH_EMB_CACHE:
        print(f"[cache][skip] {PH_EMB_CACHE} rejected by cache policy", flush=True)
    if phrase_embs is None:
        phrase_embs = encode_gpu(st_model, phrases_for_cluster, bs_try=4096)
        emb_path = _safe_cache_path(PH_EMB_CACHE)
        if emb_path:
            try:
                cache_meta = _embedding_cache_meta(st_model, normalize_embeddings=True, emb_tensor=phrase_embs)
                cache_meta['phrases_fp'] = phrases_fp
                cache_payload = {
                    'phrases': phrases_for_cluster,
                    'phrases_fp': phrases_fp,
                    'embs': phrase_embs.cpu(),
                    'meta': cache_meta,
                }
                torch.save(cache_payload, emb_path)
                print(f"[cache] saved phrase embeddings to {emb_path}", flush=True)
            except Exception:
                pass
        elif PH_EMB_CACHE:
            print(f"[cache][skip] {PH_EMB_CACHE} rejected by cache policy", flush=True)
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
        refill_bank(args.kind, reps, max_add=None)
        if args.bank_all:
            all_members = []
            for _, members, _ in clusters:
                all_members.extend(members)
            refill_bank(args.kind, list(dict.fromkeys(all_members)), max_add=None)
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
            refill_bank(kind, unique_reps, max_add=None)
            coverage_inputs.extend(dc.Proto(text=rep, label=kind) for rep in unique_reps)
        if args.bank_all:
            for kind, members in bucketed_members.items():
                refill_bank(kind, list(dict.fromkeys(members)), max_add=None)
        routing_summary = {
            "routed_kinds": {kind: len(list(dict.fromkeys(reps))) for kind, reps in bucketed_reps.items()},
            "skipped_reps": len(skipped_reps),
            "bank_all": bool(args.bank_all),
        }
    rebuild_soft_check()
    rebuild_bank_embeddings(st_model)
    rebuild_bank_sketches()
    coverage_report = None
    if coverage_inputs:
        coverage_report = dc.probe_soft_coverage(coverage_inputs, seed=args.seed, topk_show=3)
    singletons = sum(1 for _, members, _ in clusters if len(members) == 1)
    snapshot_cap = args.snapshot_max_per_kind if args.snapshot_max_per_kind is not None else SNAPSHOT_MAX_PER_KIND_DEFAULT
    snapshot_bank = _sanitize_bank_for_snapshot(
        getattr(dc, 'SOFT_PARAPHRASE_BANK', {}),
        thresh=float(args.soft_gate_thresh),
        max_per_kind=snapshot_cap,
        seed=args.seed,
    )
    def _json_safe(o):
        try:
            import numpy as _np
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, (_np.ndarray,)):
                return o.tolist()
        except Exception:
            pass
        if isinstance(o, (set, tuple)):
            return list(o)
        if isinstance(o, dict):
            return {str(k): _json_safe(v) for k, v in o.items()}
        return str(o)
    print(json.dumps({
        "prewarm_stats": {k: len(v) for k, v in prewarm_stats.items()} if isinstance(prewarm_stats, dict) else prewarm_stats,
        "hydration_stats": hydration_stats,
        "docs_raw": len(docs_raw),
        "docs_after_strip": len(docs),
        "candidates_from_keybert": len(cand_raw),
        "candidates_word_route": len(cand_word_filtered),
        "candidates_char_route": len(cand_char),
        "candidates_before_soft_gate": len(combined_candidates),
        "candidates_after_soft_gate": len(cand_after_gate),
        "soft_gate_threshold": args.soft_gate_thresh,
        "candidates_after_dedupe": len(cand),
        "clusters": len(clusters),
        "cluster_singleton_ratio": singletons / max(1, len(clusters)),
        "routing_mode": routing_mode,
        "routing_summary": routing_summary,
        "coverage_report": coverage_report,
        "snapshot_bank_counts": {k: len(snapshot_bank.get(k, [])) for k in PRIORITY},
        "snapshot_max_per_kind": snapshot_cap,
    }, ensure_ascii=False, indent=2, default=_json_safe))
    snap_path = _safe_cache_path(BANK_SNAPSHOT) if BANK_SNAPSHOT else None
    if snap_path:
        try:
            with snap_path.open("w", encoding="utf-8") as f:
                json.dump(snapshot_bank, f, ensure_ascii=False, indent=2)
            print(f"[snapshot] saved bank to {snap_path}", flush=True)
        except Exception:
            pass
    else:
        if BANK_SNAPSHOT:
            print(f"[snapshot][skip] {BANK_SNAPSHOT} rejected by cache policy", flush=True)

def main(args):
    with _seed_ctx("extract_and_bank", args.seed):
        return _main_impl(args)
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="zh_lines.txt")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--top_n", type=int, default=25)
    p.add_argument("--ngram_min", type=int, default=2)
    p.add_argument("--ngram_max", type=int, default=6)
    p.add_argument("--diversity", type=float, default=0.5, help="KeyBERT MMR diversity; 0.4-0.6 suits most cases, bump to ~0.65 for forum/colloquial corpora.")
    p.add_argument("--min_cjk", type=float, default=0.6)
    p.add_argument("--cluster_thr", type=float, default=0.85)
    p.add_argument("--cluster_min", type=int, default=2)
    p.add_argument("--prewarm_keys", type=str, default=None,
               help="Comma-separated soft evidence kinds to prewarm; defaults to the priority list.")
    p.add_argument("--soft_gate_thresh", type=float, default=0.74,
                   help="Minimum softish score for KeyBERT phrases to pass into clustering (default 0.74).")
    p.add_argument("--soft_gate_workers", type=str, default="auto",
                   help="'auto' 或整数：软门控并行进程数（默认自动吃满 CPU，仅保留 2 个核心；受 --soft_gate_cap 限制）。")
    p.add_argument("--soft_gate_cap", type=int, default=None,
                   help="软门控并行进程数上限（仅在 --soft_gate_workers=auto 时生效，默认不设上限）")
    p.add_argument("--soft_gate_chunk", type=int, default=300000,
                   help="软门控每分片条数（默认300k）")
    p.add_argument(
        "--kind",
        type=str,
        default=None,
        help="Force all phrases into the specified soft-evidence bucket; omit or set to 'auto' for auto-routing."
    )
    p.add_argument("--bank_all", action="store_true")
    p.add_argument("--prewarm_per_kind", type=int, default=300)
    p.add_argument("--prewarm_seed", type=int, default=2025)
    p.add_argument("--prewarm_allow_dupes", action="store_true",
                   help="Disable dedupe when prewarming (dedupe is on by default).")
    hydrate_group = p.add_mutually_exclusive_group()
    hydrate_group.add_argument(
        "--hydrate_microgrammars",
        action="store_true",
        dest="hydrate_microgrammars",
        default=True,
        help="Pre-expand soft evidence prototypes via micro-grammars before routing (default: enabled).",
    )
    hydrate_group.add_argument(
        "--no_hydrate_microgrammars",
        action="store_false",
        dest="hydrate_microgrammars",
        help="Skip micro-grammar hydration (default: enabled).",
    )
    p.add_argument("--hydrate_keys", type=str, default=None,
                   help="Comma-separated soft evidence kinds to hydrate; defaults to core contract/routing kinds.")
    p.add_argument("--hydrate_per_axis", type=int, default=3)
    p.add_argument("--hydrate_max_per_ev", type=int, default=400)
    p.add_argument("--hydrate_seed", type=int, default=20250924)
    p.add_argument("--gpu_mem_reserve_gb", type=float, default=1,
                   help="GPU memory reserve in GB to avoid exhausting the device (default: 1).")
    p.add_argument("--cpu_mem_reserve_pct", type=float, default=0.1,
                   help="Fraction of system RAM to keep as reserve (default: 0.1).")
    p.add_argument("--cpu_core_reserve", type=int, default=2,
                   help="Logical CPU cores to keep idle for the system (default: 2).")
    p.add_argument("--encode_workers", type=str, default="auto",
                   help="'auto' or an integer; tokenizer DataLoader workers (default: auto).")
    p.add_argument("--cand_chunk_max", type=int, default=None,
                   help="Maximum candidate encode chunk size; defaults to 50k (GPU) or 10k (CPU).")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable autocast/AMP during encoding.")
    p.add_argument("--allow_external_cache", action="store_true",
                   help="Allow caches to be written outside the script directory.")
    p.add_argument("--cand_cache", type=str, default=None,
                   help="Path to cache KeyBERT candidates; defaults to per-input name.")
    p.add_argument("--emb_cache", type=str, default=None,
                   help="Path to cache phrase embeddings; defaults to per-input name.")
    p.add_argument("--bank_snapshot", type=str, default=None,
                   help="Path to write a snapshot of the soft paraphrase bank; defaults to per-input name.")
    p.add_argument("--snapshot_max_per_kind", type=int, default=SNAPSHOT_MAX_PER_KIND_DEFAULT,
                   help="每个优先软证据类型在快照中保留的最大条数（默认 400，设为 0 表示不限制）。")
    p.add_argument("--dedupe_workers", type=str, default="auto",
                   help="'auto' or integer: dedupe worker processes (auto uses all CPU cores minus two; limited by --dedupe_worker_cap).")
    p.add_argument("--dedupe_worker_cap", type=int, default=None,
                   help="Upper bound for auto dedupe workers (None uses all available cores minus two).")
    p.add_argument("--dedupe_shard", type=int, default=1000000,
                   help="每个进程处理的分片目标条数。")
    p.add_argument("--reuse_bank_snapshot", action="store_true",
               help="若设置且 --bank_snapshot 文件已存在，则直接加载该快照到内存并跳过 prewarm/hydrate。")
    p.add_argument("--force_build_bank", action="store_true",
               help="忽略快照强制重新执行 prewarm/hydrate。")

    main(p.parse_args())
