from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


EvidenceStatus = Literal["insufficient", "partial", "sufficient"]


class AgentState(TypedDict, total=False):
    """LangGraph 图状态合同；增删字段须同步改 builder 各节点及 CLI/eval 的初始 state。"""

    messages: Annotated[list[BaseMessage], add_messages]
    evidence_status: EvidenceStatus
    rag_budget: int
    rag_calls_used: int
    active_tools: list[str]
    evidence_entries: list[dict[str, Any]]
    turn_metadata: dict[str, Any]


def initial_agent_state(
    question: str,
    *,
    rag_budget: int,
    active_tools: list[str],
    turn_metadata: dict[str, Any] | None = None,
) -> AgentState:
    """构造首轮 invoke/stream 的完整图 state（与 ``AgentState`` 合同一致）。"""
    return {
        "messages": [HumanMessage(content=question)],
        "evidence_status": "insufficient",
        "rag_budget": rag_budget,
        "rag_calls_used": 0,
        "evidence_entries": [],
        "active_tools": list(active_tools),
        "turn_metadata": turn_metadata or {},
    }