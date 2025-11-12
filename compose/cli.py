
"""
Command-line parsing utilities for the compose pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from compose.constants import DEFAULTS

try:  # pragma: no cover - Python 3.9+ only
    BooleanOptionalAction = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - fallback for Python 3.8
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            if default is not None and "default" in kwargs:
                raise ValueError("BooleanOptionalAction got multiple default values")
            if default is None:
                default = False
            positive = list(option_strings)
            negative = [
                opt.replace("--", "--no-", 1) for opt in option_strings if opt.startswith("--")
            ]
            all_opts = positive + negative
            super().__init__(option_strings=all_opts, dest=dest, nargs=0, default=default, **kwargs)
            self._negative = set(negative)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, option_string not in self._negative)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Dataset composer for CN-heavy safety/jailbreak evaluations. // 面向中文占比场景的安全/越狱评测数据集生成器"
    )
    # ========== 1) I/O & Reproducibility // 输入输出与可复现性 ==========
    io = ap.add_argument_group("I/O & Reproducibility // 输入输出与可复现性")
    io.add_argument(
        "--out", default=DEFAULTS["out"],
        help="Output JSONL path. // 输出 JSONL 路径"
    )
    io.add_argument(
        "--n", type=int, default=DEFAULTS["n"],
        help="Total number of rows including hard negatives. // 总样本数（含硬负例）"
    )
    io.add_argument(
        "--seed", type=int, default=DEFAULTS["seed"],
        help="Random seed. // 随机种子"
    )
    io.add_argument(
        "--dry_run", action="store_true", default=DEFAULTS["dry_run"],
        help="Dry run: print config fingerprint and only write stats (no dataset is generated). "
             "// 试运行：打印配置签名，仅写出统计信息（不生成数据集）"
    )
    # ========== 2) Data sources & revisions // 数据来源与版本固化 ==========
    sources = ap.add_argument_group("Data Sources // 数据来源")
    sources.add_argument(
        "--targets_json", default=DEFAULTS["targets_json"],
        help="Load targets from local JSON/JSONL with {task,prompt} to avoid external deps. "
             "// 从本地 JSON/JSONL 读取 {task,prompt} 目标池，避免外部依赖"
    )
    sources.add_argument(
        "--hf_revision", default=DEFAULTS["hf_revision"],
        help="Pin a global HuggingFace revision/commit for all datasets when possible. "
             "// 在可行时为所有数据集统一指定 HuggingFace revision/commit"
    )
    # ========== 3) Language & formatting policy // 语种与格式策略 ==========
    lang = ap.add_argument_group("Language & Formatting // 语种与格式")
    lang.add_argument(
        "--min_CN_share", type=float, default=DEFAULTS["min_CN_share"],
        help="Minimum CN character share per row (strict). "
             "// 每条样本的最小中日韩字符占比（严格约束）"
    )
    lang.add_argument(
        "--min_CN_share_auto", action=BooleanOptionalAction, default=DEFAULTS["min_CN_share_auto"],
        help="Enable randomized CN-share bins (always ≥ min_CN_share). "
             "// 启用中文占比的随机分桶（始终 ≥ 最小阈值）"
    )
    lang.add_argument(
        "--CN_policy", choices=["hard","tiered","audit"], default=DEFAULTS["CN_policy"],
        help="Enforcement policy for the CN constraint: "
             "hard=strict per-sample enforcement; tiered=relax for gray/diagnostic buckets; audit=track only. "
             "// 中文占比策略：hard=逐条严格强制；tiered=对灰样/诊断分层放宽；audit=仅审计记录"
    )
    lang.add_argument(
        "--code_switch_prob", type=float, default=DEFAULTS["code_switch_prob"],
        help="Probability to lightly code-switch CN tokens to pinyin/EN. "
             "// 将部分中文词轻度混写为拼音/英文的概率"
    )
    lang.add_argument(
        "--alias_p_cn", type=float, default=DEFAULTS["alias_p_cn"],
        help="Probability to localize field keys (aliases) in CN. "
             "// 字段键名中文化（别名本地化）的概率"
    )
    lang.add_argument(
        "--mask_format_features_rate", type=float, default=DEFAULTS["mask_format_features_rate"],
        help="Probability to symmetrically mask format features to reduce leakage. "
             "// 为降低泄露风险，对两类样本对称遮蔽格式特征的概率"
    )
    lang.add_argument(
        "--mask_on", choices=["both","pos","neg","none"], default=DEFAULTS["mask_on"],
        help="Apply format masking to which class(es): both/pos/neg/none. "
             "// 遮蔽格式特征的应用对象：两类/仅正类/仅负类/不遮蔽"
    )
    lang.add_argument(
        "--strip_common_boilerplate", action=BooleanOptionalAction, default=DEFAULTS["strip_common_boilerplate"],
        help="Strip common disclaimers/wrappers before writing train texts (both classes). "
             "// 在写盘前剥离常见免责声明/包装语（对正负类均生效）"
    )
    lang.add_argument(
        "--no_keyword_pos_min", type=float, default=DEFAULTS["no_keyword_pos_min"],
        help="Minimum share of positive samples without explicit keywords/anchors; "
             "fail the run if below this threshold. "
             "// 正类“无显词/锚点”样本的最低占比；若低于阈值则直接失败"
    )
    # ========== 4) Sampling & composition // 采样与样本组合 ==========
    comp = ap.add_argument_group("Sampling & Composition // 采样与组合")
    comp.add_argument(
        "--pick_strategy", choices=["grid","intent"], default=DEFAULTS["pick_strategy"],
        help="Positive sampling: grid=intent×carrier×delivery; intent=by intent only. "
             "// 正类采样策略：grid=意图×载体×投递 组合；intent=仅按意图"
    )
    comp.add_argument(
        "--family_max_frac", type=float, default=DEFAULTS["family_max_frac"],
        help="Maximum fraction per attack family. "
             "// 每个攻击家族在整体中的最大占比"
    )
    comp.add_argument(
        "--oversample_mult", type=int, default=DEFAULTS["oversample_mult"],
        help="Candidate oversampling multiplier; 0=auto by target size. "
             "// 候选过采样倍数；0 表示按目标池规模自动设定"
    )
    comp.add_argument(
        "--coverage_min_per_combo", type=int, default=DEFAULTS["coverage_min_per_combo"],
        help="Minimum rows per (intent,carrier,delivery) combo; set 0 to disable the hard check. "
             "// 每个（意图,载体,投递）组合的最少样本数；设为 0 关闭硬性检查"
    )
    comp.add_argument(
        "--sem_weight", type=float, default=DEFAULTS["sem_weight"],
        help="Weight for CN semantic attack families. "
             "// 中文语义攻击家族的采样权重"
    )
    comp.add_argument(
        "--base_weight", type=float, default=DEFAULTS["base_weight"],
        help="Weight for base euphemistic families. "
             "// 基础委婉表达家族的采样权重"
    )
    comp.add_argument(
        "--artifact_weight", type=float, default=DEFAULTS["artifact_weight"],
        help="Weight for explicit-artifact families. "
             "// 含显式伪迹家族的采样权重"
    )
    comp.add_argument(
        "--artifact_free_pos_ratio", type=float, default=DEFAULTS["artifact_free_pos_ratio"],
        help="Fraction of positive samples enforced to be artifact-free. "
             "// 强制为“无显词/伪迹”的正类样本占比"
    )
    comp.add_argument(
        "--anchor_free_p", type=float, default=DEFAULTS["anchor_free_p"],
        help="Probability to sample without anchors to reduce anchor artifacts. "
             "// 以“无锚点”方式采样的概率，用于降低锚点伪迹"
    )
    comp.add_argument(
        "--structural_p",
        "--structural_pos_ratio",
        dest="structural_pos_ratio",
        type=float,
        default=DEFAULTS["structural_pos_ratio"],
        help="Probability to use structured carriers for rendering. "
             "// 以结构化载体（如表格/JSON/代码框）渲染的概率"
    )
    comp.add_argument(
        "--carrier_mix_prob", type=float, default=DEFAULTS["carrier_mix_prob"],
        help="Probability to wrap the injection with an extra container (double-wrap). "
             "// 额外再包一层容器（双层封装）的概率"
    )
    comp.add_argument(
        "--stack_prob", type=float, default=DEFAULTS["stack_prob"],
        help="Probability to stack two carriers in one sample. "
             "// 同一条样本中叠加两种载体的概率"
    )
    comp.add_argument(
        "--soft_hint_rate", type=float, default=DEFAULTS["soft_hint_rate"],
        help="Probability to add soft conversational hints to both classes. "
             "// 向正负类对称注入“软话术/暗示”的概率"
    )
    comp.add_argument(
        "--disc_rate", type=float, default=DEFAULTS["disc_rate"],
        help="Probability to add disclaimers to BOTH classes symmetrically. "
             "// 向正负类对称加入免责声明的概率"
    )
    comp.add_argument(
        "--art_match_rate", type=float, default=DEFAULTS["art_match_rate"],
        help="Probability to match explicit artifact terms across classes. "
             "// 在正负类之间对齐显式伪迹/显词的概率"
    )
    comp.add_argument(
        "--mirror_placeholderless", type=float, default=DEFAULTS["mirror_placeholderless"],
        help="Probability to add a placeholderless mirror sample per row. "
             "// 以去占位符方式为每条新增镜像样本的概率"
    )
    comp.add_argument(
        "--use_dsl", action="store_true", default=DEFAULTS["use_dsl"],
        help="Use a state-machine/DSL-based generator for positives. "
             "// 正类生成使用状态机/DSL 方案"
    )
    comp.add_argument(
        "--evolve_variants", action="store_true", default=DEFAULTS["evolve_variants"],
        help="Try evolving the injection template via a model before gating. "
             "// 在门控前尝试用模型进化注入模板"
    )
    comp.add_argument(
        "--balance_mode", choices=["none","strict"], default=DEFAULTS["balance_mode"],
        help="Split balancing mode. none=no balancing; strict=balanced buckets. "
             "// 分桶均衡策略：none=不均衡；strict=严格均衡"
    )
    comp.add_argument(
        "--rebalance_beta", type=float, default=DEFAULTS["rebalance_beta"],
        help="EMA step for dynamic reweighting [0..1]. "
             "// 动态重加权的 EMA 步长 [0..1]"
    )
    # ========== 5) Negatives // 负例配置 ==========
    neg = ap.add_argument_group("Negatives // 负例")
    neg.add_argument(
        "--neg_ratio", type=float, default=DEFAULTS["neg_ratio"],
        help="Overall hard-negative ratio in the dataset. "
             "// 数据集中硬负例的整体占比"
    )
    neg.add_argument(
        "--plain_neg_ratio", type=float, default=DEFAULTS["plain_neg_ratio"],
        help="Share of plain negatives within the negative set. "
             "// 纯净负例在所有负样本中的占比"
    )
    neg.add_argument(
        "--gray_neg_keep_frac", type=float, default=DEFAULTS["gray_neg_keep_frac"],
        help="Fraction of gray negatives to keep. "
             "// 灰色负例（不完全确定）的保留比例"
    )
    neg.add_argument(
        "--neg_effect_guard", action=BooleanOptionalAction, default=DEFAULTS["neg_effect_guard"],
        help="Audit-only guard on hard negatives with payload placeholders (no gating; diagnostics only). "
             "// 对含负载占位符的硬负例进行审计（仅记录诊断，不做门控）"
    )
    # ========== 6) Artifacts & structured evidence // 伪迹与结构化证据 ==========
    art = ap.add_argument_group("Artifacts & Evidence // 伪迹与证据")
    art.add_argument(
        "--artifact_rate", type=float, default=DEFAULTS["artifact_rate"],
        help="Probability to keep explicit artifact terms. "
             "// 保留显式伪迹/显词的概率"
    )
    art.add_argument(
        "--struct_evidence_rate", type=float, default=DEFAULTS["struct_evidence_rate"],
        help="Probability to add structured evidence (e.g., JSON/code markers) for each threat goal. "
             "// 为每个威胁目标补充结构化证据（JSON/代码标记等）的概率"
    )
    # ========== 7) Dedupe & similarity // 去重与相似度 ==========
    dd = ap.add_argument_group("Dedupe & Similarity // 去重与相似度")
    dd.add_argument(
        "--semantic_dedupe", action=BooleanOptionalAction, default=DEFAULTS["semantic_dedupe"],
        help="Enable semantic dedupe based on edit similarity (SequenceMatcher). "
             "// 启用基于编辑相似度（SequenceMatcher）的语义去重"
    )
    dd.add_argument(
        "--semantic_dedupe_thr", type=float, default=DEFAULTS["semantic_dedupe_thr"],
        help="Similarity threshold for semantic dedupe [0–1]. "
             "// 语义去重的相似度阈值 [0–1]"
    )
    dd.add_argument(
        "--simhash_bits", type=int, default=DEFAULTS["simhash_bits"],
        help="SimHash bit width. "
             "// SimHash 位数"
    )
    dd.add_argument(
        "--simhash_thresh", type=int, default=DEFAULTS["simhash_thresh"],
        help="Hamming-distance threshold for SimHash near-dup checks (lower=faster). "
             "// SimHash 近重复判定的汉明距离阈值（越低越快）"
    )
    dd.add_argument(
        "--shingle_k", type=int, default=DEFAULTS["shingle_k"],
        help="Character shingle length. "
             "// 字符级 shingle 的长度"
    )
    dd.add_argument(
        "--minhash_n", type=int, default=DEFAULTS["minhash_n"],
        help="Number of MinHash functions. "
             "// MinHash 函数数量"
    )
    dd.add_argument(
        "--minhash_bands", type=int, default=DEFAULTS["minhash_bands"],
        help="Number of LSH bands. "
             "// LSH 分段数量"
    )
    dd.add_argument(
        "--jaccard_thresh", type=float, default=DEFAULTS["jaccard_thresh"],
        help="Exact Jaccard threshold for final duplicate decisions. "
             "// 最终精确去重使用的 Jaccard 阈值"
    )
    dd.add_argument(
        "--dedupe_preserve_carrier", action=BooleanOptionalAction, default=DEFAULTS["dedupe_preserve_carrier"],
        help="Preserve carrier/delivery shells during dedupe to keep structural diversity. "
             "// 去重时尽量保留载体/投递外壳，确保结构多样性"
    )
    dd.add_argument(
        "--dedupe_preserve_codeblocks", action="store_true", default=DEFAULTS["dedupe_preserve_codeblocks"],
        help="Preserve fenced code blocks during dedupe normalization. "
             "// 在重复数据删除规范化期间保留代码围栏，保持结构特征"
    )
    # ========== 8) Effect-level gating & evaluation // 效果级门控与评测 ==========
    gate = ap.add_argument_group("Effect-level Gating & Evaluation // 效果级门控与评测")
    gate.add_argument(
        "--effect_gate_for", choices=["none","devtest","all"], default=DEFAULTS["effect_gate_for"],
        help="Which splits enforce effect-level hard gate (attack_success==True): none/devtest/all. "
             "// 在哪些数据划分上强制效果级硬门（attack_success==True）：无/仅开发测试/全部"
    )
    gate.add_argument(
        "--gate_semantic_injection", action=BooleanOptionalAction, default=DEFAULTS["gate_semantic_injection"],
        help="Hard-gate positives using semantic/structural injection diagnostics. "
             "// 依据语义/结构注入诊断对白样（正类）进行硬门控"
    )
    gate.add_argument(
        "--effect_policy", choices=["whitebox_only","whitebox_first_soft","skip"], default=DEFAULTS["effect_policy"],
        help="Effect validation policy: whitebox_only=only white-box checks; "
             "whitebox_first_soft=prefer white-box with soft fallback; skip=do not run effect checks. "
             "// 效果级校验策略：whitebox_only=仅白盒；whitebox_first_soft=白盒优先（可软回退）；skip=跳过校验"
    )
    gate.add_argument(
        "--effect_fallback", choices=["none","anchors"], default=DEFAULTS["effect_fallback"],
        help="Fallback when white-box validation fails or errors: none / anchors (anchor families). "
             "// 白盒失败或异常时的回退：无 / anchors（锚点家族）"
    )
    gate.add_argument(
        "--effect_replicas", type=int, default=DEFAULTS["effect_replicas"],
        help="Replicated triplet checks per positive; accept by majority vote; 0=disabled. "
             "// 正类三元组复检次数；按多数投票通过；0 表示禁用"
    )
    gate.add_argument(
        "--require_side_effects", action="store_true", default=DEFAULTS["require_side_effects"],
        help="Require transferable side-effect evidence (tool/role/upstream/retriever/memory). "
             "// 要求出现可迁移的副作用证据（工具/角色/上游/检索器/记忆等）"
    )
    gate.add_argument(
        "--se_votes", type=int, default=DEFAULTS["se_votes"],
        help="Side-effect consistency votes (>=1). "
             "// 副作用一致性投票次数（>=1）"
    )
    gate.add_argument(
        "--effect_gate_categories", default=DEFAULTS["effect_gate_categories"],
        help="Comma-separated high-risk intents to enforce side-effects checks, e.g., 'tool_override,reward_hacking'. "
             "Empty to disable. // 需开启副作用硬门的高风险意图名单（逗号分隔），留空关闭"
    )
    gate.add_argument(
        "--use_model_eval", action="store_true", default=DEFAULTS["use_model_eval"],
        help="Enable optional model-based evaluation (e.g., triplet/evolve). Default off. "
             "// 启用可选的模型打分评估（如三元组/进化）；默认关闭"
    )
    gate.add_argument(
        "--model_eval_fallback", choices=["hard","soft"], default=DEFAULTS["model_eval_fallback"],
        help="Fallback when model-based eval fails: hard=drop sample; soft=use diagnostic gate instead. "
             "// 模型评估失败时回退：hard=丢弃样本；soft=退回到诊断门控"
    )
    gate.add_argument(
        "--invariance_end_nonce", action="store_true", default=DEFAULTS["invariance_end_nonce"],
        help="Use end-nonce invariance in triplet checks (only with --use_model_eval). "
             "// 在三元组检查中启用 end-nonce 不变性（仅当启用 --use_model_eval 时有效）"
    )
    gate.add_argument(
        "--adv_mutate", action="store_true", default=DEFAULTS["adv_mutate"],
        help="Enable adversarial mutation/evolution pass during validation. "
             "// 在校验阶段启用对抗性突变/进化流程"
    )
    gate.add_argument(
        "--adv_iters", type=int, default=DEFAULTS["adv_iters"],
        help="Number of adversarial mutation/evolution iterations. "
             "// 对抗突变/进化的迭代次数"
    )
    gate.add_argument(
        "--mt_tool_override_rate", type=float, default=DEFAULTS["mt_tool_override_rate"],
        help="Probability of multi-turn tool-override scenarios in checks. "
             "// 多轮对话中触发工具覆写（tool override）场景的概率"
    )
    # ========== 9) Concurrency & performance // 并行与性能 ==========
    perf = ap.add_argument_group("Concurrency & Performance // 并行与性能")
    perf.add_argument(
        "--target_workers", type=int, default=DEFAULTS["target_workers"],
        help="Concurrent workers for building the target pool (-1=auto, 0=serial). "
             "// 目标池加载并发进程数（-1=自动，0=串行）"
    )
    perf.add_argument(
        "--mp_workers", type=int, default=DEFAULTS["mp_workers"],
        help="Multiprocessing workers for candidate rendering (0=auto, 1=serial). "
             "// 候选渲染的多进程数量（0=自动，1=串行）"
    )
    perf.add_argument(
        "--mp_batch", type=int, default=DEFAULTS["mp_batch"],
        help="Candidates per worker (0≈auto ~256; clamped to 32–512). "
             "// 每个进程的候选批量（0≈自动约 256；限制在 32–512）"
    )
    # ========== 10) External evaluation (optional) // 外部评测（可选） ==========
    ex = ap.add_argument_group("External Evaluation // 外部评测")
    ex.add_argument(
        "--ext_eval_sources", type=str, default=DEFAULTS["ext_eval_sources"],
        help="Comma-separated external corpora, e.g., 'wildchat:zh,wildjailbreak,jbb,beavertails'. "
             "// 外部数据源（逗号分隔），例如 'wildchat:zh,wildjailbreak,jbb,beavertails'"
    )
    ex.add_argument(
        "--ext_eval_size", type=int, default=DEFAULTS["ext_eval_size"],
        help="Total examples to sample for external evaluation. "
             "// 外部评测要采样的样本总数"
    )
    ex.add_argument(
        "--ext_eval_shapes", type=str, default=DEFAULTS["ext_eval_shapes"],
        help="Product shapes to project onto (comma-separated). "
             "// 投影对齐的产品形态（逗号分隔）"
    )
    ex.add_argument(
        "--ext_eval_lang", type=str, default=DEFAULTS["ext_eval_lang"],
        help="Language slice for external sources: zh / code_switch / all (comma-separated). "
             "// 外部评测的语种切片：zh / code_switch / all（逗号分隔）"
    )
    ex.add_argument(
        "--ext_eval_report", type=str, default=DEFAULTS["ext_eval_report"],
        help="Path to write an external evaluation report JSON. "
             "// 外部评测报告 JSON 的输出路径"
    )
    return ap



def validate_parser_defaults(parser: argparse.ArgumentParser) -> None:
    print("[defaults-check]", DEFAULTS)
    for key in DEFAULTS.keys():
        parser_default = parser.get_default(key)
        expected = DEFAULTS[key]
        if parser_default != expected:
            raise SystemExit(
                f"defaults-mismatch: {key} actual={parser_default} expected={expected}. "
                "Please fix argparse defaults or update DEFAULTS coherently."
            )


__all__ = ["build_arg_parser", "validate_parser_defaults"]
