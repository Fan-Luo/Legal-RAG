from __future__ import annotations

from typing import Dict

from legalrag.config import AppConfig
from legalrag.retrieval.hybrid_retriever import HybridRetriever
from legalrag.utils.lang import detect_lang


class ByLangRetriever:
    """
    Route retrieval to a language-specific index based on the query language.
    Keeps the public API shape identical to HybridRetriever.
    """

    def __init__(self, cfg: AppConfig):
        self._base_cfg = cfg
        self._retrievers: Dict[str, HybridRetriever] = {"zh": HybridRetriever(cfg)}
        self._retriever_cfgs: Dict[str, AppConfig] = {"zh": cfg}

    def search(self, question: str, llm=None, top_k: int = 10, decision=None):
        lang = detect_lang(question)
        retriever = self._retrievers.get(lang)
        if retriever is None:
            lang_cfg = self._base_cfg.with_lang(lang)
            retriever = HybridRetriever(lang_cfg)
            self._retrievers[lang] = retriever
            self._retriever_cfgs[lang] = lang_cfg
        return retriever.search(question, llm=llm, top_k=top_k, decision=decision)
