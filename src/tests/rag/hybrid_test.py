from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings
from rag.index import build_index
from rag.retriever import get_retriever


@pytest.fixture()
def indexed_settings(tmp_path: Path) -> Settings:
    repo = Path(__file__).resolve().parents[3]
    return Settings(
        base_url=None,
        api_key=None,
        model_name="test",
        corpus_dir=repo / "fixtures" / "corpus",
        index_dir=tmp_path / "index",
        index_profile="wiki-graph-v1",
        index_variant="raw-wiki",
        rag_budget=4,
        embedding_mode="fake",
        embedding_model="text-embedding-3-small",
        embedding_base_url=None,
        embedding_api_key=None,
        embedding_device="cpu",
        embedding_model_path=repo / "models" / "bge-small-zh-v1.5",
    )


def test_build_index_and_search(indexed_settings: Settings) -> None:
    build_index(indexed_settings)
    retriever = get_retriever(indexed_settings)
    # FakeEmbeddings 的向量排序噪声大，用文档里英文关键词让 BM25 稳定命中
    hits = retriever.search("300 秒内可视为新鲜", top_k=8)
    assert hits
    assert any("300" in h.content and "新鲜" in h.content for h in hits)


def test_read_note(indexed_settings: Settings) -> None:
    build_index(indexed_settings)
    retriever = get_retriever(indexed_settings)
    chunks = retriever.read_note("dev/http-caching.md", query="max-age", max_chunks=5)
    assert chunks
    assert any("300" in c["content"] for c in chunks)