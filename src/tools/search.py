from __future__ import annotations

import json

from langchain_core.tools import tool

from rag.retriever import HybridRetriever


def make_search_notes_tool(retriever: HybridRetriever):
    @tool
    def search_notes(query: str, top_k: int = 5) -> str:
        """检索本地知识索引；适合概念、经验与文档类问题。"""
        hits = retriever.search(query, top_k=top_k)
        payload = [
            {
                "source": h.source,
                "excerpt": h.content[:600],
                "chunkId": h.chunk_id,
            }
            for h in hits
        ]
        return json.dumps({"matches": payload, "count": len(payload)}, ensure_ascii=False)

    return search_notes