from __future__ import annotations

from graph.state import initial_agent_state


def test_initial_agent_state_sets_contract_fields() -> None:
    state = initial_agent_state(
        "hello",
        rag_budget=4,
        active_tools=["search_notes"],
        turn_metadata={"mode": "test"},
    )
    assert state["rag_budget"] == 4
    assert state["evidence_status"] == "insufficient"
    assert state["rag_calls_used"] == 0
    assert state["active_tools"] == ["search_notes"]
    assert state["turn_metadata"] == {"mode": "test"}
    assert len(state["messages"]) == 1
    assert state["messages"][0].content == "hello"