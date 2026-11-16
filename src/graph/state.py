from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


EvidenceStatus = Literal["insufficient", "partial", "sufficient"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    evidence_status: EvidenceStatus
    rag_budget: int
    rag_calls_used: int
    active_tools: list[str]
    evidence_entries: list[dict[str, Any]]
    turn_metadata: dict[str, Any]