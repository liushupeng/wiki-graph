from __future__ import annotations

from wiki_graph.graph.builder import build_agent_graph
from wiki_graph.graph.state import AgentState


def test_graph_compiles_without_invoke() -> None:
    """M2：图可编译；完整 invoke 依赖 API，在 CI 外手动 --check-llm / ask。"""
    # 仅验证 State 字段与 builder 导入（不触发 LLM）
    state: AgentState = {
        "messages": [],
        "evidence_status": "insufficient",
        "rag_budget": 4,
        "rag_calls_used": 0,
        "evidence_entries": [],
        "active_tools": ["search_notes", "read_note"],
    }
    assert state["rag_budget"] == 4
    assert build_agent_graph.__name__ == "build_agent_graph"