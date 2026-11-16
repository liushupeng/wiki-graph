from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from wiki_graph.config import Settings

CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass
class SearchHit:
    source: str
    content: str
    score: float
    chunk_id: str


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _rrf_merge(
    dense: list[tuple[str, int]],
    sparse: list[tuple[str, int]],
    k: int = 60,
) -> list[str]:
    scores: dict[str, float] = {}
    for doc_id, rank in dense:
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for doc_id, rank in sparse:
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ordered]


def _embeddings(settings: Settings):
    if settings.embedding_mode == "fake" or not settings.llm_configured():
        return FakeEmbeddings(size=384)
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        openai_api_key=settings.api_key,
        openai_api_base=settings.base_url,
    )


class HybridRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index_dir = settings.index_dir
        self._chunks: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._bm25: BM25Okapi | None = None
        self._vector: Chroma | None = None
        self._load()

    def _load(self) -> None:
        chunks_path = self.index_dir / CHUNKS_FILE
        if not chunks_path.is_file():
            raise FileNotFoundError(
                f"索引不存在: {chunks_path}。请先运行 `python -m wiki_graph.cli index`。"
            )
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self._chunks.append(row)
                self._by_id[row["id"]] = row
        corpus_tokens = [_tokenize(c["content"]) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus_tokens)
        self._vector = Chroma(
            collection_name="wiki_graph",
            persist_directory=str(self.index_dir / "chroma"),
            embedding_function=_embeddings(self.settings),
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        top_k = min(max(top_k, 1), 12)
        assert self._bm25 is not None and self._vector is not None

        dense_docs = self._vector.similarity_search(query, k=min(top_k * 3, 24))
        dense_ranked = [
            (doc.metadata["chunk_id"], i + 1)
            for i, doc in enumerate(dense_docs)
            if doc.metadata.get("chunk_id")
        ]

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        sparse_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[: min(top_k * 3, 24)]
        sparse_ranked = [(self._chunks[i]["id"], rank + 1) for rank, i in enumerate(sparse_indices)]

        fused_ids = _rrf_merge(dense_ranked, sparse_ranked)[:top_k]
        hits: list[SearchHit] = []
        for doc_id in fused_ids:
            chunk = self._by_id.get(doc_id)
            if not chunk:
                continue
            hits.append(
                SearchHit(
                    source=chunk["source"],
                    content=chunk["content"],
                    score=1.0,
                    chunk_id=doc_id,
                )
            )
        return hits

    def read_note(
        self,
        note_path: str,
        *,
        query: str | None = None,
        max_chunks: int = 8,
    ) -> list[dict[str, Any]]:
        max_chunks = min(max(max_chunks, 1), 30)
        normalized = note_path.replace("\\", "/").lstrip("/")
        matched = [c for c in self._chunks if c["source"] == normalized]
        if not matched:
            matched = [c for c in self._chunks if c["source"].endswith(normalized)]
        if query and query.strip():
            q = query.lower()
            matched.sort(
                key=lambda c: sum(1 for t in q.split() if t in c["content"].lower()),
                reverse=True,
            )
        else:
            matched.sort(key=lambda c: c.get("chunk_index", 0))
        return [
            {
                "source": c["source"],
                "content": c["content"],
                "chunkIndex": c.get("chunk_index", 0),
            }
            for c in matched[:max_chunks]
        ]


def get_retriever(settings: Settings | None = None) -> HybridRetriever:
    return HybridRetriever(settings or Settings.from_env())