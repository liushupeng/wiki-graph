from __future__ import annotations

from config import Settings
from rag.retriever import HybridRetriever
from tools.grep import make_grep_notes_tool
from tools.mock_ticket import query_incident_ticket
from tools.read_note import make_read_note_tool
from tools.search import make_search_notes_tool

DEFAULT_ACTIVE_TOOLS = [
    "search_notes",
    "grep_notes",
    "read_note",
    "query_incident_ticket",
]


def build_agent_tools(retriever: HybridRetriever, settings: Settings) -> list:
    return [
        make_search_notes_tool(retriever),
        make_grep_notes_tool(settings),
        make_read_note_tool(retriever),
        query_incident_ticket,
    ]


__all__ = [
    "DEFAULT_ACTIVE_TOOLS",
    "build_agent_tools",
    "make_search_notes_tool",
    "make_read_note_tool",
    "make_grep_notes_tool",
    "query_incident_ticket",
]