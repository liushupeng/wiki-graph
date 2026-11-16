from __future__ import annotations

import json

from langchain_core.tools import tool

from wiki_graph.rag.retriever import HybridRetriever


def make_read_note_tool(retriever: HybridRetriever):
    @tool
    def read_note(note_path: str, query: str | None = None, max_chunks: int = 8) -> str:
        """按笔记相对路径读取 Markdown 片段，例如 dev/http-caching.md。"""
        chunks = retriever.read_note(note_path, query=query, max_chunks=max_chunks)
        return json.dumps(
            {"notePath": note_path, "chunks": chunks, "count": len(chunks)},
            ensure_ascii=False,
        )

    return read_note