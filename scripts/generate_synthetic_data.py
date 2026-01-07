"""
Synthetic Query Generator  
- 双模型（Generator + Judge）
- 多轮对话生成
- LLM 评分（1–10）
- LLM 改写（rewrite）
- embedding 去重（bge-m3）
- 语言一致性过滤
- 条文痕迹过滤 
"""

from __future__ import annotations

import os
import re
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
import torch

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from legalrag.config import AppConfig
from legalrag.llm.client import LLMClient


# =========================
# 基础工具
# =========================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# =========================
# 文本清洗与检测
# =========================

ARTICLE_PATTERN = re.compile(
    r"(第[一二三四五六七八九十百千零〇两0-9]+条)|"
    r"(本法第[一二三四五六七八九十百千零〇两0-9]+条[第款项]*)|"
    r"(依据本法规定)|"
    r"(根据本法规定)|"
    r"(依照本法规定)",
    re.UNICODE,
)

FULLWIDTH_SPACE = "\u3000"


def strip_citation_markers(text: str) -> str:
    """去掉条文编号、‘本法规定’等引用痕迹。"""
    if not text:
        return ""
    text = text.replace(FULLWIDTH_SPACE, " ")
    text = ARTICLE_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_chinese_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def detect_language_simple(text: str) -> str:
    """非常简单的语言检测：中文字符占比 > 0.3 → zh，否则 en。"""
    if not text:
        return "unknown"
    chinese_count = sum(1 for ch in text if is_chinese_char(ch))
    ratio = chinese_count / max(len(text), 1)
    return "zh" if ratio > 0.3 else "en"


def looks_like_article(text: str) -> bool:
    """判断是否像条文本身，而不是问题。"""
    if not text:
        return True
    if "?" not in text and "？" not in text:
        if text.strip().endswith(("的", "时", "为", "者", "之", "等", "；", ";", "。", ".")):
            return True
    if re.search(r"第[一二三四五六七八九十百千零〇两0-9]+条", text):
        return True
    return False


def is_question_like(text: str) -> bool:
    """判断是否像问句。"""
    if not text:
        return False
    if "?" in text or "？" in text:
        return True
    if text.strip().startswith(("如何", "什么", "哪些", "是否", "能否", "在什么情况下")):
        return True
    if re.match(r"^(How|What|When|Why|Which|Can|Could|Should|Is|Are|Do|Does)\b", text.strip(), re.I):
        return True
    return False


def has_mixed_lang(text: str) -> bool:
    """简单判断是否中英文混杂。"""
    if not text:
        return False
    has_zh = any(is_chinese_char(ch) for ch in text)
    has_en = any("a" <= ch.lower() <= "z" for ch in text)
    return has_zh and has_en



# =========================
# Query quality rules (standalone / non-deictic / non-abstract)
# =========================

DEICTIC_PATTERN = re.compile(
    r"(这种|这样的|该等|上述|前述|本案|该案|此种|该种|其中|对此|在此情况下|在这种情况下|该情况下|这种情况下|该情形|该行为)",
    re.UNICODE,
)

ABSTRACT_PATTERN = re.compile(
    r"(如何合理|一般而言|原则上|通常情况下|应当如何|法律如何规定|如何理解|如何适用|如何认定|如何判断)",
    re.UNICODE,
)

ZH_FACT_ANCHORS = [
    "合同", "违约", "履行", "解除", "终止", "效力", "赔偿", "违约金", "定金", "退款",
    "交付", "付款", "交货", "租赁", "买卖", "借款", "担保", "保证", "抵押", "质押",
    "订立", "签订", "签署", "协商", "通知", "催告", "解除权", "撤销", "无效", "可撤销",
]

EN_FACT_ANCHORS = [
    "contract", "breach", "performance", "terminate", "termination", "liability", "damages",
    "penalty", "deposit", "refund", "deliver", "delivery", "payment", "lease", "sale", "loan",
    "guarantee", "mortgage", "void", "rescission",
]


def has_deictic_reference(text: str) -> bool:
    if not text:
        return False
    return bool(DEICTIC_PATTERN.search(text))


def is_overly_abstract(text: str) -> bool:
    if not text:
        return True
    # abstract templates + no factual anchors -> abstract
    if ABSTRACT_PATTERN.search(text):
        if not any(k in text for k in ZH_FACT_ANCHORS) and not any(k in text.lower() for k in EN_FACT_ANCHORS):
            return True
    # textbook-like topic: rights/obligations without a contract/transaction anchor
    if ("权利" in text and "义务" in text and ("如何" in text or "应该" in text)) and ("合同" not in text):
        return True
    return False


def has_min_fact_anchor(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if any(k in t for k in ZH_FACT_ANCHORS):
        return True
    tl = t.lower()
    if any(k in tl for k in EN_FACT_ANCHORS):
        return True
    return False


def clean_and_filter_query(
    q: str,
    min_len: int = 6,
    max_len: int = 200,
) -> Optional[str]:
    "Clean and filter a single query; it must be standalone-answerable."
    if not q:
        return None
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    q = strip_citation_markers(q)

    # length
    if len(q) < min_len or len(q) > max_len:
        return None

    # article-like text
    if looks_like_article(q):
        return None

    # must look like a question
    if not is_question_like(q):
        return None

    # forbid deictic/anaphora
    if has_deictic_reference(q):
        return None

    # forbid overly abstract prompts
    if is_overly_abstract(q):
        return None

    # require minimal factual anchor
    if not has_min_fact_anchor(q):
        return None

    return q


# =========================
# 双模型初始化（Generator + Judge）
# =========================

def init_generator_llm(cfg: AppConfig, logger: logging.Logger) -> LLMClient:
    """
    Generator（生成模型）选择逻辑：
    1. 如果有 GPU → 使用 Qwen（本地）
    2. 否则如果用户提供 OPENAI_API_KEY → 使用 OpenAI
    3. 否则 → degraded 模式（LLMClient 会自动 fallback）
    """
    llm = LLMClient.from_config(cfg)
    logger.info(f"[Generator] provider={llm.provider}, model={llm.model_name}")
    return llm


def init_judge_llm(cfg: AppConfig, logger: logging.Logger) -> LLMClient:
    """
    Judge（评审模型）选择逻辑：
    1. 如果用户提供 OPENAI_API_KEY → 用 OpenAI（最强）
    2. 否则如果有 GPU → 用 Qwen 小模型（7B/14B）
    3. 否则 → fallback（使用同一个模型）
    """
    openai_key = os.getenv(cfg.llm.api_key_env, "").strip()

    if openai_key:
        logger.info("[Judge] Using OpenAI for evaluation")
        return LLMClient.from_config_with_key(cfg, openai_key=openai_key)

    # 无 OpenAI Key → 尝试 Qwen 小模型
    if torch.cuda.is_available():
        logger.info("[Judge] Using Qwen-local (7B/14B) for evaluation")
        # 强制覆盖为小模型
        os.environ[cfg.llm.qwen_model_env] = "Qwen/Qwen2.5-7B-Instruct"
        return LLMClient.from_config(cfg)

    # fallback
    logger.info("[Judge] Fallback: using same provider as Generator")
    return LLMClient.from_config(cfg)


# =========================
# Prompt  
# =========================

SYSTEM_PROMPT_EN = (
    "You are a helpful assistant that generates natural, high-quality legal questions "
    "for retrieval evaluation.\n"
    "Follow the user's instructions strictly.\n"
    "Do not copy or restate the legal article.\n"
    "Do not include article numbers or citations.\n"
    "Produce natural questions that real users would ask.\n"
    "One question per line."
)

# -------- 单轮生成 Prompt --------

def build_single_turn_prompt(text: str, law_name: str, article_no: str, role: str, num_q: int, lang: str) -> str:
    if lang == "zh":
        return f"""
你是一名{role}，正在阅读一条法律条文。请根据这条条文，提出 {num_q} 个自然的问题。

要求：
- 问题必须是你“真实可能会问”的，而不是复述条文。
- 不要出现条文编号（如“第几条”）、“本法规定”等字样。
- 不要直接照抄条文原句。
- 每个问题单独一行。
- 语言：中文。

法律名称：{law_name}
条文编号：{article_no}
条文内容：
{text}
""".strip()

    else:
        return f"""
You are a {role} reading a legal provision. Based on this article, generate {num_q} natural questions you might genuinely ask.

Requirements:
- Questions must sound like real user questions, not restatements of the article.
- Do NOT include article numbers or citations.
- Do NOT copy sentences from the article.
- One question per line.
- Language: English.

Law name: {law_name}
Article number: {article_no}
Article text:
{text}
""".strip()


# -------- 多轮对话 Prompt --------

MULTI_TURN_PROMPT = """
Simulate a realistic multi-turn conversation between a user and a lawyer about the following legal article.

Rules:
- 5 turns total: User → Lawyer → User → Lawyer → User
- User questions must be natural, realistic, and based on the article.
- Do NOT copy or restate the article.
- Do NOT include article numbers or citations.
- Keep each turn short and natural.
- Output format:
User: ...
Lawyer: ...
User: ...
Lawyer: ...
User: ...

Article:
{article}
"""


# -------- 评分 Prompt --------

JUDGE_SCORE_PROMPT = """
You are a strict evaluator. Score the following question from 1 to 10.

Scoring rules:
- 10: Extremely natural, realistic, and useful legal question.
- 8–9: High quality, natural, meaningful.
- 6–7: Acceptable but slightly unnatural or vague.
- 4–5: Low quality, unnatural, or unclear.
- 1–3: Very poor, meaningless, or copied from the article.

Return ONLY a number (1–10), nothing else.

Question:
{query}
"""


# -------- 改写 Prompt --------

REWRITE_PROMPT = """
Rewrite the following legal question so that it can be answered WITHOUT any external context.

Rules:
- Remove vague references such as "this", "such", "in this case", and Chinese deictic phrases like "上述/前述/这种/这样的/该情况".
- If needed, restate the minimal factual situation explicitly, but do NOT invent new facts.
- Do NOT make it more abstract or philosophical.
- Do NOT add article numbers, citations, or legal sources.
- Output ONLY the rewritten question (one line).

Question:
{query}

Rewrite:
"""



# =========================
# LLM 调用封装
# =========================

def llm_chat(llm: LLMClient, system_prompt: str, user_prompt: str) -> str:
    """统一封装 LLMClient.chat()"""
    out = llm.chat(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    return out.strip() if isinstance(out, str) else str(out)


# =========================
# 多轮对话生成 + 抽取用户问题
# =========================

def generate_multi_turn_dialog(llm: LLMClient, article_text: str) -> List[str]:
    """生成 5 轮对话，并抽取所有 User 的问题"""
    prompt = MULTI_TURN_PROMPT.format(article=article_text)
    raw = llm_chat(llm, SYSTEM_PROMPT_EN, prompt)

    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    user_questions = []

    for line in lines:
        if line.lower().startswith("user:"):
            q = line.split(":", 1)[1].strip()
            cleaned = clean_and_filter_query(q)
            if cleaned:
                user_questions.append(cleaned)

    return user_questions


# =========================
# 单轮生成
# =========================

def generate_single_turn_questions(
    llm: LLMClient,
    text: str,
    law_name: str,
    article_no: str,
    role: str,
    num_q: int,
    lang: str,
) -> List[str]:

    prompt = build_single_turn_prompt(text, law_name, article_no, role, num_q, lang)
    raw = llm_chat(llm, SYSTEM_PROMPT_EN, prompt)

    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    out = []

    for line in lines:
        line = re.sub(r"^[0-9]+\.\s*", "", line)
        line = re.sub(r"^[-•]\s*", "", line)
        cleaned = clean_and_filter_query(line)
        if cleaned:
            out.append(cleaned)

    return out


# =========================
# 评分（Judge）
# =========================

def judge_score(judge_llm: LLMClient, query: str) -> float:
    prompt = JUDGE_SCORE_PROMPT.format(query=query)
    raw = llm_chat(judge_llm, SYSTEM_PROMPT_EN, prompt)

    # 只返回数字
    m = re.search(r"([0-9]+(\.[0-9]+)?)", raw)
    if not m:
        return 0.0
    try:
        score = float(m.group(1))
        return max(1.0, min(10.0, score))
    except:
        return 0.0


# =========================
# 改写（Judge）
# =========================


def judge_rewrite(judge_llm: LLMClient, query: str) -> Optional[str]:
    "Rewrite query to be standalone; return None if rewrite still fails quality rules."
    prompt = REWRITE_PROMPT.format(query=query)
    raw = llm_chat(judge_llm, SYSTEM_PROMPT_EN, prompt)

    cleaned = clean_and_filter_query(raw)
    if not cleaned:
        return None

    # final guardrail
    if has_deictic_reference(cleaned) or is_overly_abstract(cleaned) or (not has_min_fact_anchor(cleaned)):
        return None
    return cleaned



# =========================
# embedding 去重
# =========================

def deduplicate_by_embedding(queries, model, threshold = 0.85):
    if not queries:
        return []

    texts = [q["query"] for q in queries]
    emb = model.encode(texts, normalize_embeddings=True)

    keep = []
    used = set()

    for i, q in enumerate(queries):
        if i in used:
            continue
        keep.append(q)
        for j in range(i + 1, len(queries)):
            if j in used:
                continue
            sim = float(np.dot(emb[i], emb[j]))
            if sim >= threshold:
                used.add(j)

    return keep


# =========================
# 生成 queries
# =========================

def generate_queries_for_article(
    generator_llm: LLMClient,
    judge_llm: LLMClient,
    row: pd.Series,
    per_article: int = 5,
    zh_ratio: float = 0.7,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:

    text_raw = str(row.get("text", "")).strip()
    if not text_raw:
        return []

    text = strip_citation_markers(text_raw)
    if len(text) < 10:
        return []

    law_name = str(row.get("law_name", ""))
    article_no = str(row.get("article_no", ""))
    article_id = row.get("article_id", None)

    roles = ["user", "lawyer", "judge", "inhouse"]
    all_queries = []

    # -------- 单轮生成 --------
    for role in roles:
        lang = "zh" if random.random() < zh_ratio else "en"
        qs = generate_single_turn_questions(
            generator_llm, text, law_name, article_no, role, per_article, lang
        )
        for q in qs:
            all_queries.append({
                "query": q,
                "lang": detect_language_simple(q),
                "role": role,
                "law_name": law_name,
                "article_no": article_no,
                "article_id": article_id,
                "round": 1,
                "rewritten": False,
                "score": None,
            })

    # -------- 多轮对话生成 --------
    dialog_qs = generate_multi_turn_dialog(generator_llm, text)
    for q in dialog_qs:
        all_queries.append({
            "query": q,
            "lang": detect_language_simple(q),
            "role": "user",
            "law_name": law_name,
            "article_no": article_no,
            "article_id": article_id,
            "round": 2,
            "rewritten": False,
            "score": None,
        })

    # -------- 评分 + 改写 --------
    final_queries = []
    for q in all_queries:
        score = judge_score(judge_llm, q["query"])
        q["score"] = score

        if score < 7.0:
            # 改写一次
            rewritten = judge_rewrite(judge_llm, q["query"])
            if not rewritten:
                continue
            q["query"] = rewritten
            q["rewritten"] = True
            q["score"] = judge_score(judge_llm, rewritten)

        if q["score"] >= 7.0:
            final_queries.append(q)

    return final_queries


# =========================
# 构建最终 queries
# =========================

def build_ground_truth_queries(
    df_chunks: pd.DataFrame,
    per_article: int,
    max_articles: Optional[int],
    total_queries: Optional[int],
    logger: logging.Logger,
    zh_ratio: float = 0.7,
    seed: int = 0,
    generator_llm=None, 
    judge_llm=None,
    embedding_model=None
) -> pd.DataFrame:

    set_seed(seed)

    cfg = AppConfig.load(None)

    if generator_llm is None: 
        generator_llm = init_generator_llm(cfg, logger) 
    if judge_llm is None: 
        judge_llm = init_judge_llm(cfg, logger)

    # embedding 去重模型
    if embedding_model is None: 
        embedding_model = SentenceTransformer(cfg.retrieval.embedding_model)
    logger.info(f"[Embedding] Using model: {embedding_model}")

    if max_articles is None:
        max_articles = len(df_chunks)

    df_sampled = df_chunks.sample(
        n=min(max_articles, len(df_chunks)),
        random_state=seed,
    ).reset_index(drop=True)

    all_rows: List[Dict[str, Any]] = []

    pbar = tqdm(df_sampled.itertuples(index=False), total=len(df_sampled), desc="Generating queries")
    for row in pbar:
        row_s = pd.Series(row._asdict())

        qs = generate_queries_for_article(
            generator_llm=generator_llm,
            judge_llm=judge_llm,
            row=row_s,
            per_article=per_article,
            zh_ratio=zh_ratio,
            logger=logger,
        )
        all_rows.extend(qs)

        if total_queries is not None and len(all_rows) >= total_queries:
            break

    if not all_rows:
        return pd.DataFrame()

    # embedding 去重
    logger.info(f"[Dedup] Before: {len(all_rows)}")
    all_rows = deduplicate_by_embedding(all_rows, embedding_model, threshold=0.85)
    logger.info(f"[Dedup] After: {len(all_rows)}")

    df = pd.DataFrame(all_rows)

    # 截断
    if total_queries is not None and len(df) > total_queries:
        df = df.sample(n=total_queries, random_state=seed).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Query Generator for LegalRAG (2026 Edition)"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to processed law JSONL. Default: cfg.paths.law_jsonl",
    )

    parser.add_argument(
        "--per-article",
        type=int,
        default=5,
        help="Approximate number of queries generated per article.",
    )

    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Max number of articles to sample.",
    )

    parser.add_argument(
        "--total-queries",
        type=int,
        default=None,
        help="Target total number of synthetic queries.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/synthetic_queries.jsonl",
        help="Output JSONL file path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--zh-ratio",
        type=float,
        default=0.7,
        help="Probability of generating Chinese queries (0~1).",
    )

    args = parser.parse_args()

    logger = setup_logger()
    set_seed(args.seed)

    logger.info("Starting Synthetic Query Generation (Level 1 + 2 + 3)...")
    logger.info(f"Seed={args.seed}, zh_ratio={args.zh_ratio}")

    cfg = AppConfig.load(None)
    input_path = Path(args.input) if args.input else Path(cfg.paths.law_jsonl)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found; run preprocess first.")

    logger.info(f"Loading corpus from: {input_path}")
    chunks = [
        json.loads(l)
        for l in input_path.open("r", encoding="utf-8")
        if l.strip()
    ]
    df_chunks = pd.DataFrame(chunks)
    logger.info(f"Loaded chunks: {len(df_chunks)}")

    if args.max_articles is None:
        args.max_articles = min(500, len(df_chunks))

    logger.info(
        f"Generation plan: per_article={args.per_article}, "
        f"max_articles={args.max_articles}, total_queries={args.total_queries}"
    )

    df_queries = build_ground_truth_queries(
        df_chunks=df_chunks,
        per_article=args.per_article,
        max_articles=args.max_articles,
        total_queries=args.total_queries,
        logger=logger,
        zh_ratio=args.zh_ratio,
        seed=args.seed
    )


    logger.info(f"Generated queries: {len(df_queries)}")

    if df_queries.empty:
        logger.warning("No queries generated. Check LLM config / API key / filters.")
        return

    logger.info("Sample queries:")
    try:
        logger.info(df_queries.sample(min(5, len(df_queries)), random_state=args.seed))
    except Exception:
        logger.info(df_queries.head(5))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df_queries.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    logger.info(f"Synthetic queries saved to: {out_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
