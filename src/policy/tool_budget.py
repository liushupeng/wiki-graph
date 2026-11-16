from __future__ import annotations

RAG_TOOL_NAMES = frozenset({"search_notes", "read_note", "grep_notes"})


def apply_budget_to_active_tools(active_tools: list[str], rag_budget: int) -> list[str]:
    if rag_budget > 0:
        return active_tools
    return [name for name in active_tools if name not in RAG_TOOL_NAMES]