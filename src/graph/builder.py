from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from config import Settings
from graph.nodes import (
    make_agent_node,
    make_finalize_node,
    make_llm,
    make_rag_tools_node,
    route_after_agent,
)
from graph.state import AgentState
from rag.retriever import get_retriever
from tools import build_agent_tools


def build_agent_graph(settings: Settings | None = None, *, checkpointer: bool = True):
    settings = settings or Settings.from_env()
    retriever = get_retriever(settings)
    tools = build_agent_tools(retriever, settings)
    llm = make_llm(settings)
    agent = make_agent_node(llm, tools)
    tools_node = make_rag_tools_node(tools)
    finalize = make_finalize_node()

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools_node)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    memory = MemorySaver() if checkpointer else None
    return graph.compile(checkpointer=memory)