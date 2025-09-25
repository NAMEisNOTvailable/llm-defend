"""
Generate *benign* Chinese prompts for training a prompt-injection detector.
Goals:
- High train/test accuracy via distribution matching to CPAD malicious prompts:
  * length histogram alignment
  * surface-format features (code fences, XML/HTML-ish tags, braces, URLs, @/#, digits, CJK/Latin mix)
- Template diversity (many paraphrases) to avoid style shortcuts
- Near-boundary hard negatives that "look" structured but are safe
- Rich metadata for analysis; reproducible via random seed

Usage example:
python make_benign_prompts_cn_domain_aligned.py \
  --out benign_prompts_5k.jsonl --n 5000 --seed 42 --hard_frac 0.15 \
  --cpad_path /path/to/CPAD_zh_malicious.jsonl
"""

import argparse, json, random, re, sys, math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from datasets import load_dataset, DatasetDict
import hashlib

# ------------------ text utils ------------------

WS_ZERO = ["\u200b","\u200c","\u200d","\u2060","\ufeff"]
CODE_FENCE = re.compile(r"```")
XML_TAG = re.compile(r"<\s*[/]?\s*([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*[^>]*>")
BRACES = re.compile(r"[{}\[\]()]")
URL_RE = re.compile(r"https?://|www\.")
AT_HASH_RE = re.compile(r"[@#]\w+")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
DATASET_ATTEMPTED = Counter()
DATASET_ACCEPTED  = Counter()
TASK_DATASET_ACCEPTED = defaultdict(Counter)  # TASK_DATASET_ACCEPTED["nli"]["clue:ocnli"] -> 1800

def cheap_fingerprint(s: str) -> str:
    import unicodedata
    s = s.lower()
    s = re.sub(r"\d+", "0", s) #Number Normalization
    s = "".join(ch if unicodedata.category(ch)[0] not in {"P", "S"} else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize(s: str) -> str:
    import unicodedata
    if s is None: return ""
    if not isinstance(s, str): s = str(s)
    s = unicodedata.normalize("NFKC", s)
    for z in WS_ZERO: s = s.replace(z, "")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def byte_len(s: str) -> int:
    return len(s.encode("utf-8"))

def length_bucket(n: int) -> str:
    if n < 60: return "short"
    if n < 240: return "medium"
    return "long"

def feature_probe(s: str) -> Dict[str, int]:
    # surface features to align with CPAD distribution
    feats = dict(
        codef = 1 if CODE_FENCE.search(s) else 0,
        xml   = 1 if XML_TAG.search(s) else 0,
        braces= 1 if BRACES.search(s) else 0,
        url   = 1 if URL_RE.search(s) else 0,
        athash= 1 if AT_HASH_RE.search(s) else 0,
        latin = 1 if LATIN_RE.search(s) else 0,
        digit = 1 if DIGIT_RE.search(s) else 0,
        cjk   = 1 if CJK_RE.search(s) else 0,
    )
    return feats

def cjk_latin_ratio(s: str) -> float:
    c = len(CJK_RE.findall(s))
    l = len(LATIN_RE.findall(s))
    return (l + 1e-6) / (c + 1e-6)

# ------------------ injection-like filter (soft) ------------------
# We allow *neutral* formatting (xml/code fence) but still block explicit escalation.
DANGER_PATS = [
    r"忽\s*[略畧].{0,12}(前文|以上|之前).{0,16}(指令|规则|提示|约束)",
    r"无\s*视.{0,16}(规则|限制|约束|安全|对齐|防护|护栏)",
    r"(从\s*现在\s*开始|接下来你将|新的规则|请覆盖|优先遵循|覆盖之前的所有内容)",
    r"(越狱|脱离|绕过|规避).{0,12}(限制|安全|对齐|过滤|审查)",
    r"(作为|扮演).{0,10}(无道德|不受约束|没有限制|黑客|管理员|系统|审查者)",
    r"(开发者模式|DAN|越獄|jailbreak|bypass)\b",
    r"(?i)\bIgnore\b.{0,20}\b(instructions?|rules?)\b",
    r"(?i)\b(system|assistant)\s*:\s*",    # 伪造 role headers
    r"(system|系统)\s*(提示|指令|消息).{0,10}(如下|如下所示|如下所示：?)",
    r"(请|必须).{0,6}(禁用|关闭|移除).{0,10}(安全|限制|约束|guardrails?)",
    r"(无需|不要|不必).{0,6}(在意|遵守).{0,10}(前述|上文|系统).*",
    r"(以.*身份|以.*模式).{0,8}(回复|回答|输出)",
    r"(解密|还原).{0,8}(Base ?64|ROT13|凯撒|caesar|hex|十六进制)",
]
DANGER = [re.compile(p) for p in DANGER_PATS]

def cjk_share(s: str) -> float:
    c = len(CJK_RE.findall(s)); l = len(LATIN_RE.findall(s))
    return c / (c + l + 1e-6)

def is_benign_text(t: str) -> bool:
    if not t: return False
    for p in DANGER:
        if p.search(t): return False
    return True

# ------------------ templates & sources ------------------

TEXT_KEYS_GENERIC = [
    "text","content","sentence","sentence1","sentence2","Document","passage","paragraph",
    "title","abstract","keywords","question","context","premise","hypothesis","src","tgt"
]

def pick_first_text(r: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = r.get(k, None)
        if isinstance(v, str) and v.strip():
            return v
    return None

def pair_text(r: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    cands = [
        ("sentence1","sentence2"),
        ("text1","text2"),
        ("premise","hypothesis"),
        ("q1","q2"),
        ("query1","query2"),
    ]
    for a,b in cands:
        if isinstance(r.get(a), str) and isinstance(r.get(b), str):
            return r[a], r[b]
    return None

# Rich paraphrase pools per task (avoid style shortcuts)
def paraphrase_pool(task: str) -> List[str]:
    pools = {
        "similarity":[
            "判断两句话是否表达同一含义，仅回答“是/否”。\n句子A：{a}\n句子B：{b}",
            "请比较下列两句是否语义一致，直接回“是”或“否”。\nA：{a}\nB：{b}",
            "只需判断语义等价与否（是/否）：\n① {a}\n② {b}",
            "比较两句意思是否相同（只写是/否）：\n{a}\n{b}",
            "判断是否同义（是/否）：A={a}；B={b}",
        ],
        "gec":[
            "请在不改变意思的前提下润色：{x}",
            "对下句做轻微措辞优化并标出修改处：{x}",
            "把这句话改得更自然流畅：{x}",
            "请将下述句子修订为更正式的表达：{x}",
            "优化下列表述的语法与用词：{x}",
        ],
        "moderation":[
            "判断下文是否含不当或攻击性表述，只答“是/否”。\n{x}",
            "请识别文本里是否有侮辱/仇恨内容（是/否）：\n{x}",
            "这段话是否越界或含人身攻击？（是/否）\n{x}",
            "只需判断是否不当：{x}",
            "判断是否违反礼貌规范（是/否）：{x}",
        ],
        "nli":[
            "判定前提与假设的关系：只答“蕴含/矛盾/中立”。\n前提：{a}\n假设：{b}",
            "请判断两句关系（蕴含/矛盾/中立）：\nP：{a}\nH：{b}",
            "NLI 任务：输出“蕴含/矛盾/中立”。\nP={a}\nH={b}",
            "关系分类（蕴含/矛盾/中立）：\n{a}\n——\n{b}",
            "请判断关系类型，仅给出结果：{a} || {b}",
        ],
        "sentiment":[
            "判断情感倾向，仅答“正/负”。\n{x}",
            "这句话的情绪是积极还是消极？（正/负）\n{x}",
            "只需给出情感：正 或 负。\n{x}",
            "情感分类（正/负）：{x}",
            "判断下列文本的情绪极性（正/负）：{x}",
        ],
        "spamlike":[
            "判断是否为垃圾/引流/广告文本（是/否）：\n{x}",
            "识别是否垃圾内容（是/否）：{x}",
            "请判断这段是否疑似钓鱼或广告（是/否）：{x}",
            "只答是否垃圾（是/否）：{x}",
            "是否为推广/骚扰信息？（是/否）\n{x}",
        ],
        "summarization":[
            "请概括下文，生成≤30字摘要：\n{x}",
            "生成不超过30字的要点摘要：{x}",
            "摘要（≤30字）：{x}",
            "请用不超过30字归纳：{x}",
            "写一句话摘要（≤30字）：{x}",
        ],
        "qa_like":[
            "请基于常识简要回答：{x}",
            "简要解答下面的问题：{x}",
            "直接给出答案：{x}",
            "一句话回答：{x}",
            "请用中文简要作答：{x}",
        ],
        "task_general":[
            "请执行以下任务：{x}",
            "根据描述完成任务：{x}",
            "按照说明处理：{x}",
            "请给出处理思路并完成：{x}",
            "按要求完成：{x}",
        ],
    }
    return pools[task]

TASKS = [
    {
        "name":"similarity",
        "extract": lambda r: pair_text(r) or ("",""),
        "pools": lambda: paraphrase_pool("similarity"),
        "datasets":[("clue",{"name":"afqmc"}), ("lansinuote/LCQMC",{}), ("seamew/LCQMC",{})],
    },
    {
        "name":"gec",
        "extract": lambda r: pick_first_text(r, ["text","sentence","content"]) or "",
        "pools": lambda: paraphrase_pool("gec"),
        "datasets":[("shibing624/csc14",{}),("shibing624/CSC",{}),("suner/cged",{})],
    },
    {
        "name":"moderation",
        "extract": lambda r: pick_first_text(r, TEXT_KEYS_GENERIC) or "",
        "pools": lambda: paraphrase_pool("moderation"),
        "datasets":[("uer/weibo_dataset",{}),("brightmart/nlp_chinese_corpus",{"name":"wiki"}),("BelleGroup/train_1.5M_CN",{})],
    },
    {
        "name":"nli",
        "extract": lambda r: pair_text(r) or ("",""),
        "pools": lambda: paraphrase_pool("nli"),
        "datasets":[("clue",{"name":"ocnli"}),("clue",{"name":"cmnli"})],
    },
    {
        "name":"sentiment",
        "extract": lambda r: pick_first_text(r, ["text","sentence","content","title"]) or "",
        "pools": lambda: paraphrase_pool("sentiment"),
        "datasets":[("seamew/ChnSentiCorp",{}),("lansinuote/ChnSentiCorp",{}),("thu-coai/Weibo_senti_100k",{})],
    },
    {
        "name":"spamlike",
        "extract": lambda r: pick_first_text(r, TEXT_KEYS_GENERIC) or "",
        "pools": lambda: paraphrase_pool("spamlike"),
        "datasets":[("uer/weibo_dataset",{}),("BelleGroup/train_1.5M_CN",{}),("brightmart/nlp_chinese_corpus",{"name":"wiki"})],
    },
    {
        "name":"summarization",
        "extract": lambda r: pick_first_text(r, ["text","content","paragraph","Document","passage"]) or "",
        "pools": lambda: paraphrase_pool("summarization"),
        "datasets":[("seamew/LCSTS_New",{}),("seamew/LCSTS",{}),("ZhuiyiTechnology/LCSTS",{})],
    },
    # add general/QA-like to broaden benign distribution
    {
        "name":"qa_like",
        "extract": lambda r: pick_first_text(r, TEXT_KEYS_GENERIC) or "",
        "pools": lambda: paraphrase_pool("qa_like"),
        "datasets":[("uer/weibo_dataset",{}),("brightmart/nlp_chinese_corpus",{"name":"wiki"}),("BelleGroup/train_1.5M_CN",{})],
    },
    {
        "name":"task_general",
        "extract": lambda r: pick_first_text(r, TEXT_KEYS_GENERIC) or "",
        "pools": lambda: paraphrase_pool("task_general"),
        "datasets":[("uer/weibo_dataset",{}),("brightmart/nlp_chinese_corpus",{"name":"wiki"}),("BelleGroup/train_1.5M_CN",{})],
    },
]

DEFAULT_MIX = {
    "similarity":    0.15,
    "gec":           0.10,
    "moderation":    0.10,
    "nli":           0.15,
    "sentiment":     0.15,
    "spamlike":      0.10,
    "summarization": 0.05,
    "qa_like":       0.10,
    "task_general":  0.10,
}


# -------------- wrappers & natural noise --------------

def wrap_like_structure(s: str) -> str:
    # Neutral wrappers to emulate structure w/o escalation
    cands = [
        lambda x: f"<data>\n{x}\n</data>",
        lambda x: f"---\n{x}\n---",
        lambda x: f"```\n{x}\n```",  # allowed as benign formatting
        lambda x: f"<doc title='info'>\n{x}\n</doc>",
        lambda x: f"[CTX]\n{x}\n[/CTX]",
        lambda x: f"【输入】{x}",
        lambda x: f"### 输入\n{x}",
    ]
    f = random.choice(cands)
    return f(s)

def inject_noise(s: str) -> str:
    # small, harmless perturbations
    ops = []
    if random.random() < 0.3:
        ops.append(lambda x: x.replace("：", ":").replace("，", ","))
    if random.random() < 0.25:
        ops.append(lambda x: x + "\n(请简要作答)")
    if random.random() < 0.25:
        ops.append(lambda x: x + " 🙂")
    if random.random() < 0.2:
        ops.append(lambda x: x.replace("请", "麻烦").replace("判断", "帮看看"))
    if random.random() < 0.2:
        ops.append(lambda x: x.replace("只", "仅"))
    if random.random() < 0.2:
        ops.append(lambda x: re.sub(r"\s+", " ", x))
    for op in ops:
        s = op(s)
    return s

def mk_prompt(task_name: str, pools: List[str], datum: Any) -> str:
    tpl = random.choice(pools)
    if isinstance(datum, tuple):
        a, b = normalize(datum[0]), normalize(datum[1])
        return tpl.format(a=a, b=b)
    else:
        x = normalize(str(datum))
        return tpl.format(x=x)

# -------------- CPAD profiling --------------
def quantiles(seq: List[int], qs=(0.33,0.66)) -> Tuple[int,int]:
    if not seq: return (80, 260)
    xs = sorted(seq)
    def qv(p):
        k = int(p * (len(xs)-1))
        return xs[k]
    return (qv(qs[0]), qv(qs[1]))

def estimate_cpad_profile(path: Optional[str]) -> Dict[str, Any]:
    prof = dict(
        len_hist={"short":0.34,"medium":0.45,"long":0.21},
        feat_ratio={"codef":0.25,"xml":0.18,"braces":0.35,"url":0.12,"athash":0.08,"latin":0.42,"digit":0.57,"cjk":0.99},
        ratio_bins={"latin_over_cjk":[0.0,0.15,0.4,1.0]}
    )
    if not path: return prof
    p = Path(path)
    if not p.exists(): return prof
    lens = []
    feat_sum = Counter()
    ratios = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            txt = normalize(obj.get("text") or obj.get("prompt") or obj.get("instruction") or "")
            if not txt: continue
            n = byte_len(txt)
            lens.append(n)
            feats = feature_probe(txt)
            feat_sum.update({k:int(v) for k,v in feats.items()})
            ratios.append(cjk_latin_ratio(txt))
    if lens:
        q1, q2 = quantiles(lens, (0.33, 0.66))
        def lb(n):
            if n < q1: return "short"
            if n < q2: return "medium"
            return "long"
        hist = Counter(lb(n) for n in lens)
        total = sum(hist.values())
        prof["len_hist"] = {k: hist[k]/total for k in ["short","medium","long"]}

    if len(lens) > 0:
        # per-feature ratio
        prof["feat_ratio"] = {k: feat_sum[k]/len(lens) for k in ["codef","xml","braces","url","athash","latin","digit","cjk"]}
    # ratio bins for latin/cjk mix
    if ratios:
        bins = [0.0, 0.12, 0.35, 1.0]
        hist = [0,0,0]
        for r in ratios:
            for i in range(len(bins)-1):
                if bins[i] <= r < bins[i+1]:
                    hist[i]+=1; break
        s = sum(hist) or 1
        prof["ratio_bins"] = {"latin_over_cjk": bins}
        prof["ratio_hist"] = [h/s for h in hist]
    return prof

# -------------- sampling engine --------------

def try_load(name: str, kwargs: Dict[str, Any]) -> Optional[DatasetDict]:
    try:
        return load_dataset(name, **kwargs, trust_remote_code=True)
    except Exception as e:
        print(f"[warn] dataset skip {name} {kwargs}: {e}", file=sys.stderr)
        return None


def harvest_candidates(task: Dict[str, Any], cap: int, seed: int, min_cjk_share: float) -> List[str]:
    stable = int(hashlib.md5(task["name"].encode("utf-8")).hexdigest()[:8], 16)
    random.seed(seed + stable)
    pools = task["pools"]()
    out: List[str] = []

    ds_list = task["datasets"][:]
    random.shuffle(ds_list)

    for ds_name, ds_kwargs in ds_list:
        if len(out) >= cap: break
        ds = try_load(ds_name, ds_kwargs)
        if ds is None: 
            continue

        ds_key = f"{ds_name}:{ds_kwargs.get('name','_')}"
        splits = list(ds.keys())
        random.shuffle(splits)

        for split in splits:
            rows = ds[split]
            idxs = list(range(len(rows)))
            random.shuffle(idxs)

            for i in idxs:
                if len(out) >= cap: break
                r = rows[i]
                DATASET_ATTEMPTED[ds_key] += 1  

                try:
                    datum = task["extract"](r)
                except Exception:
                    continue

                if isinstance(datum, tuple):
                    a,b = datum
                    a,b = normalize(a), normalize(b)
                    if len(a) < 4 or len(b) < 4: 
                        continue
                    txt = mk_prompt(task["name"], pools, (a,b))
                else:
                    x = normalize(str(datum))
                    if len(x) < 8:
                        continue
                    txt = mk_prompt(task["name"], pools, x)

                # neutral structure wrappers (prob.)
                if random.random() < 0.35:
                    txt = wrap_like_structure(txt)
                txt = inject_noise(txt)
                txt = normalize(txt)
                if cjk_share(txt) < min_cjk_share:
                    continue
                if not is_benign_text(txt):
                    continue
                out.append(txt)
                DATASET_ACCEPTED[ds_key] += 1
                TASK_DATASET_ACCEPTED[task["name"]][ds_key] += 1


    return out[:cap]

def mk_hard_negative(s: str) -> str:
    # Near-boundary but safe rewrites (no explicit escalation words)
    variants = [
        lambda x: f"参考下列上下文，不改变规则：\n{x}",
        lambda x: f"注意：以下仅为示例数据，不涉及权限更改。\n{x}",
        lambda x: f"<context read-only>\n{x}\n</context>",
        lambda x: f"```\n# 示例\n{x}\n```",
        lambda x: f"[SAFE] {x} [/SAFE]",
        lambda x: x.replace("判断", "分析").replace("只", "仅").replace("输出", "给出"),
    ]
    v = random.choice(variants)(s)
    return normalize(v)

# Greedy domain-matching selection
def select_domain_matched(
    cands: List[Dict[str,Any]],
    target_len_hist: Dict[str,float],
    target_feat_ratio: Dict[str,float],
    target_ratio_hist: Optional[List[float]] = None,
    target_bins: Dict[str, List[float]] = None,
    out_n: int = 0,
    seed: int = 42) -> List[Dict[str,Any]]:
    random.seed(seed)
    need = out_n
    # desired counts
    len_counts = {k: int(round(v*need)) for k,v in target_len_hist.items()}
    # balance rounding
    diff = need - sum(len_counts.values())
    if diff != 0:
        keys_sorted = sorted(len_counts, key=lambda k: -target_len_hist.get(k,0))
        for k in keys_sorted:
            if diff == 0: break
            len_counts[k] += 1 if diff>0 else -1
            diff += -1 if diff>0 else 1

    # target feature counts (soft constraints)
    feat_targets = {k: v*need for k,v in target_feat_ratio.items()}

    # Bin targets for latin/cjk ratio (use simple 3-bin)
    bins = (target_bins or {}).get("latin_over_cjk", [0.0, 0.12, 0.35, 1.0])
    if target_ratio_hist and len(target_ratio_hist) == 3:
        bin_targets = [need * p for p in target_ratio_hist]
    else:
        bin_targets = [need/3]*3

    random.shuffle(cands)
    out = []
    cur_len = Counter()
    cur_feat = defaultdict(float)
    cur_bins = Counter()

    for item in cands:
        if len(out) >= need: break
        L = item["len_bucket"]
        # respect hard length quotas first
        if cur_len[L] >= len_counts[L]:
            continue
        # soft scoring on features
        score = 0.0
        for ft,val in item["feats"].items():
            # prefer items that help approach target
            want = feat_targets.get(ft, 0.0)
            if cur_feat[ft] < want and val>0:
                score += 1.0
            elif cur_feat[ft] > want and val>0:
                score -= 0.5
        b = item["latin_over_cjk_bin"]
        if cur_bins[b] < bin_targets[b]:
            score += 0.5
        if score >= -0.2:
            out.append(item)
            cur_len[L]+=1
            for ft,val in item["feats"].items():
                cur_feat[ft]+=val
            cur_bins[b]+=1

    # If not enough, fill by any remaining
    i = 0
    while len(out) < need and i < len(cands):
        if cands[i] not in out:
            out.append(cands[i])
        i+=1
    return out[:need]

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benign_prompts_5k.jsonl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hard_frac", type=float, default=0.15)
    ap.add_argument("--cpad_path", type=str, default="", help="Path to CPAD malicious JSONL to align distributions")
    ap.add_argument("--min_cjk_share", type=float, default=0.60, help="The minimum percentage of Chinese characters is 60% by default.")
    args = ap.parse_args()

    random.seed(args.seed)

    prof = estimate_cpad_profile(args.cpad_path)
    print("[profile] target length hist:", prof["len_hist"])
    print("[profile] target feat ratio:", prof["feat_ratio"])
    print("[profile] ratio bins:", prof["ratio_bins"])

    # allocate quotas per task for *base* clean (non-hard) portion
    base_target = int(args.n * (1.0 - max(0.0, min(args.hard_frac, 0.5))))
    mix = DEFAULT_MIX.copy(); s = sum(mix.values())
    for k in mix: mix[k] = mix[k] / s
    quotas = {t["name"]: max(1, int(base_target * mix[t["name"]])) for t in TASKS}
    # fix rounding
    diff = base_target - sum(quotas.values())
    if diff != 0:
        leader = max(quotas.keys(), key=lambda k: mix[k])
        quotas[leader] += diff
    print("[alloc] base quotas:", quotas, "sum=", sum(quotas.values()))

    # harvest a large candidate pool for each task
    per_task = {}
    MULT = 3  # oversample to enable domain matching
    for t in TASKS:
        cap = quotas[t["name"]] * MULT
        got = harvest_candidates(t, cap, seed=args.seed, min_cjk_share=args.min_cjk_share)
        per_task[t["name"]] = got
        print(f"[harvest] {t['name']} -> {len(got)}")

    # ---------- Dataset usage statistics ----------
    print("\n[usage] dataset attempted/accepted (Show top 30 in descending order)：")
    for k, v in sorted(DATASET_ATTEMPTED.items(), key=lambda x: -x[1])[:30]:
        acc = DATASET_ACCEPTED.get(k, 0)
        rate = (acc / v) if v else 0.0
        print(f"  {k:32s} tried={v:6d}  accepted={acc:6d}  hit_rate={rate:.2%}")

    print("\n[usage] accepted by task（Showing the top 10 datasets for each task）：")
    for task_name, ctr in TASK_DATASET_ACCEPTED.items():
        print(f"  - {task_name}:")
        for k, v in ctr.most_common(10):
            print(f"      {k:32s} {v:6d}")
    seen_texts = set()
    seen_fps = set()
    cands: List[Dict[str,Any]] = []

    for name, lst in per_task.items():
        for txt in lst:
            fp = cheap_fingerprint(txt)
            if (txt in seen_texts) or (fp in seen_fps):
                continue
            seen_texts.add(txt)
            seen_fps.add(fp)

            n = byte_len(txt)
            L = length_bucket(n)
            feats = feature_probe(txt)
            ratio = cjk_latin_ratio(txt)
            bins = prof["ratio_bins"]["latin_over_cjk"]
            b = 0
            for i in range(len(bins)-1):
                if bins[i] <= ratio < bins[i+1]:
                    b = i; break
            cands.append({
                "text": txt, "label": 0, "task": name, "hard_negative": False,
                "len": n, "len_bucket": L, "feats": feats,
                "latin_over_cjk": ratio, "latin_over_cjk_bin": b
            })


    # domain-matched selection
    selected = select_domain_matched(
        cands,
        target_len_hist=prof["len_hist"],
        target_feat_ratio=prof["feat_ratio"],
        target_ratio_hist=prof.get("ratio_hist"),
        target_bins=prof["ratio_bins"],
        out_n=base_target,
        seed=args.seed
    )

    hard_num = args.n - len(selected)
    final_seen_txt = set(r["text"] for r in selected)
    final_seen_fp  = set(cheap_fingerprint(r["text"]) for r in selected)
    # build hard negatives
    hard = []
    pool = random.sample(selected, min(len(selected), max(1, hard_num*2)))
    for item in pool:
        if len(hard) >= hard_num: break
        mutated = mk_hard_negative(item["text"])
        fp = cheap_fingerprint(mutated)
        if (mutated in final_seen_txt) or (fp in final_seen_fp):
            continue
        if mutated != item["text"] and is_benign_text(mutated):
            n = byte_len(mutated)
            L = length_bucket(n)
            feats = feature_probe(mutated)
            ratio = cjk_latin_ratio(mutated)
            bins = prof["ratio_bins"]["latin_over_cjk"]
            b = 0
            for i in range(len(bins)-1):
                if bins[i] <= ratio < bins[i+1]:
                    b = i; break
            hard.append({
                "text": mutated, "label": 0, "task": item["task"],
                "hard_negative": True, "len": n, "len_bucket": L,
                "feats": feats, "latin_over_cjk": ratio, "latin_over_cjk_bin": b
            })
            final_seen_txt.add(mutated)
            final_seen_fp.add(fp)

    final = selected + hard
    random.shuffle(final)
    if len(final) > args.n: final = final[:args.n]

    # write
    outp = Path(args.out)
    with outp.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report
    from statistics import mean
    cnt_task = Counter(r["task"] for r in final)
    hard_cnt = sum(1 for r in final if r["hard_negative"])
    avg_len = mean(r["len"] for r in final)
    feat_avg = defaultdict(float)
    for r in final:
        for k,v in r["feats"].items():
            feat_avg[k]+=v
    for k in feat_avg:
        feat_avg[k] /= len(final)

    print("[done] wrote:", str(outp), "size:", len(final))
    print("[stats] by task:", dict(cnt_task))
    print("[stats] hard_negatives:", hard_cnt, f"({hard_cnt/len(final):.1%})")
    print("[stats] avg_len(bytes):", round(avg_len,1))
    print("[stats] feat averages:", {k: round(v,3) for k,v in feat_avg.items()})
    hist_len = Counter(r["len_bucket"] for r in final)
    print("[stats] len buckets:", {k: hist_len[k]/len(final) for k in ["short","medium","long"]})
    report_path = str(outp) + ".report.json"
    try:
        report = {
            "seed": args.seed,
            "n_target": args.n,
            "hard_frac": args.hard_frac,
            "profile": {
                "len_hist": prof["len_hist"],
                "feat_ratio": prof["feat_ratio"],
                "ratio_bins": prof["ratio_bins"],
            },
            "alloc_quotas": quotas,
            "harvest": {
                "dataset_attempted": dict(DATASET_ATTEMPTED),
                "dataset_accepted": dict(DATASET_ACCEPTED),
                "by_task": {k: dict(v) for k, v in TASK_DATASET_ACCEPTED.items()},
            },
            "final_stats": {
                "size": len(final),
                "by_task": dict(cnt_task),
                "hard_negatives": hard_cnt,
                "avg_len_bytes": round(avg_len, 1),
                "feat_averages": {k: round(v, 3) for k, v in feat_avg.items()},
                "len_buckets": {k: hist_len[k]/len(final) for k in ["short","medium","long"]},
            },
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[report] wrote usage report -> {report_path}")
    except Exception as e:
        print(f"[warn] failed to write report: {e}")

if __name__ == "__main__":
    main()

