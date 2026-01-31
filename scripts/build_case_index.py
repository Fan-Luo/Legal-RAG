from __future__ import annotations

from legalrag.config import AppConfig
from legalrag.retrieval.case_retriever import CaseRetriever
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = AppConfig.load()
    retriever = CaseRetriever(cfg)
    if not retriever.cases:
        logger.warning("No cases loaded; check data/cases/case_law.jsonl")
        return
    retriever.vector_store.build(retriever.cases)
    logger.info("Case index built: %s", cfg.retrieval.case_index_file)


if __name__ == "__main__":
    main()
