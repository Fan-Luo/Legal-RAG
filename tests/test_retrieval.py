import json
from pathlib import Path

import pytest

from legalrag.retrieval.bm25_retriever import BM25Retriever
from legalrag.retrieval.vector_store import VectorStore
from legalrag.retrieval.hybrid_retriever import HybridRetriever
from legalrag.models import RetrievalHit


# -------------------------------------------------------------------
# 辅助配置对象 
# -------------------------------------------------------------------


class DummyRetrievalCfg:
    def __init__(self, base_dir: Path):
        self.bm25_index_file = str(base_dir / "bm25.pkl")
        self.processed_file = str(base_dir / "contract_law.jsonl")
        self.faiss_index_file = str(base_dir / "faiss.index")
        self.faiss_meta_file = str(base_dir / "faiss_meta.jsonl")

        # HybridRetriever 用到的参数
        self.embedding_model = "fake-model"  # 测试中不会真正加载
        self.top_k = 3
        self.dense_weight = 0.7
        self.bm25_weight = 0.3


class DummyCfg:
    def __init__(self, base_dir: Path):
        self.retrieval = DummyRetrievalCfg(base_dir)


# -------------------------------------------------------------------
# BM25Retriever 测试 
# -------------------------------------------------------------------


@pytest.mark.parametrize("query, expected_keyword", [
    ("违约金如何调整？", "违约金"),
    ("解除合同的条件是什么？", "解除合同"),
])
def test_bm25_retriever_build_and_search(tmp_path: Path, query: str, expected_keyword: str):
    """
    使用临时 JSONL 构建 BM25 索引，验证检索结果大致符合关键词匹配直觉。
    """
    base_dir = tmp_path
    cfg = DummyCfg(base_dir)

    # 准备一个非常小的 processed JSONL
    processed_path = Path(cfg.retrieval.processed_file)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    docs = [
        {
            "id": "1",
            "law_name": "民法典·合同编",
            "chapter": "总则",
            "section": "一般规定",
            "article_no": "1",
            "text": "当事人应当按照约定全面履行自己的义务。",
            "source": "test",
            "start_char": 0,
            "end_char": 10,
        },
        {
            "id": "2",
            "law_name": "民法典·合同编",
            "chapter": "违约责任",
            "section": "违约金",
            "article_no": "2",
            "text": "当事人约定的违约金过高的，人民法院可以予以适当调整。",
            "source": "test",
            "start_char": 0,
            "end_char": 20,
        },
        {
            "id": "3",
            "law_name": "民法典·合同编",
            "chapter": "合同的效力",
            "section": "合同的解除",
            "article_no": "3",
            "text": "一方迟延履行主要债务，经催告后在合理期限内仍未履行的，对方可以解除合同。",
            "source": "test",
            "start_char": 0,
            "end_char": 30,
        },
    ]

    with processed_path.open("w", encoding="utf-8") as f:
        for obj in docs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    retriever = BM25Retriever(cfg)

    # build() 会读取 processed_file 并生成 bm25.pkl
    retriever.build()

    hits = retriever.search(query, top_k=2)

    assert len(hits) == 2
    # 命中的文本中应包含我们期望的关键词之一
    top_texts = [h[0].text for h in hits]
    assert any(expected_keyword in t for t in top_texts)


# -------------------------------------------------------------------
# VectorStore 测试：索引缺失时应抛 FileNotFoundError
# -------------------------------------------------------------------


def test_vector_store_raises_when_index_missing(tmp_path: Path):
    """
    当 FAISS 索引或元数据文件不存在时，VectorStore.search 应抛出 FileNotFoundError。
    """
    cfg = DummyCfg(tmp_path)
    vs = VectorStore(cfg)

    with pytest.raises(FileNotFoundError):
        _ = vs.search("违约金如何调整？", top_k=3)

