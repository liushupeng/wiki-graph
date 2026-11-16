from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


EvidenceStatus = Literal["insufficient", "partial", "sufficient"]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    evidence_status: EvidenceStatus
    rag_budget: int
    active_tools: list[str]