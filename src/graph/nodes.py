from __future__ import annotations

from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from config import Settings
from graph.state import AgentState
from policy.evidence import (
    build_model_feedback,
    finalize_content,
    record_tool_result,
    assess_evidence,
)
from policy.tool_budget import RAG_TOOL_NAMES, apply_budget_to_active_tools

SYSTEM_PROMPT = """你是 wiki-graph 知识库助手，回答应基于工具检索到的笔记内容。
需要事实时优先调用 search_notes 或 read_note；回答中注明来源路径（如 dev/http-caching.md）。
若检索不到足够信息，明确说明缺口，不要编造。"""


def make_agent_node(llm: ChatOpenAI, tools: list[BaseTool]):
    tools_by_name = {t.name: t for t in tools}

    def agent(state: AgentState, config: RunnableConfig) -> dict:
        messages = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

        active_names = state.get("active_tools") or [t.name for t in tools]
        active_tools = [tools_by_name[n] for n in active_names if n in tools_by_name]
        if not active_tools:
            active_tools = tools

        bound = llm.bind_tools(active_tools)
        response = bound.invoke(messages, config)
        return {"messages": [response]}

    return agent


def make_rag_tools_node(tools: list[BaseTool]):
    executor = ToolNode(tools)

    def tools_node(state: AgentState, config: RunnableConfig) -> dict:
        result = executor.invoke(state, config)
        new_messages = list(result.get("messages", []))

        entries = list(state.get("evidence_entries") or [])
        rag_budget = int(state.get("rag_budget", 0))
        rag_calls_used = int(state.get("rag_calls_used", 0))
        active_tools = list(state.get("active_tools") or [t.name for t in tools])

        for msg in new_messages:
            if not isinstance(msg, ToolMessage):
                continue
            name = msg.name or ""
            if name in RAG_TOOL_NAMES:
                rag_calls_used += 1
                rag_budget = max(0, rag_budget - 1)
            entries.extend(record_tool_result(name, msg.content if isinstance(msg.content, str) else str(msg.content)))

        evidence_status, _ = assess_evidence(entries)
        active_tools = apply_budget_to_active_tools(active_tools, rag_budget)

        distinct_sources = len({e.get("source") for e in entries if e.get("source")})
        feedback = build_model_feedback(
            evidence_status=evidence_status,
            evidence_count=len(entries),
            distinct_sources=distinct_sources,
            rag_calls_used=rag_calls_used,
            rag_budget=rag_budget,
        )
        new_messages.append(SystemMessage(content=feedback))

        return {
            "messages": new_messages,
            "evidence_entries": entries,
            "evidence_status": evidence_status,
            "rag_budget": rag_budget,
            "rag_calls_used": rag_calls_used,
            "active_tools": active_tools,
        }

    return tools_node


def make_finalize_node():
    def finalize(state: AgentState, config: RunnableConfig) -> dict:
        messages = state["messages"]
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return {}

        content = last.content if isinstance(last.content, str) else str(last.content)
        entries = state.get("evidence_entries") or []
        rag_calls_used = int(state.get("rag_calls_used", 0))
        evidence_status = state.get("evidence_status", "insufficient")
        new_content, _action = finalize_content(
            content,
            rag_calls_used=rag_calls_used,
            evidence_count=len(entries),
            evidence_status=evidence_status,
        )
        if new_content == content:
            return {}

        updates: dict = {"messages": [AIMessage(content=new_content)]}
        if getattr(last, "id", None):
            updates["messages"] = [RemoveMessage(id=last.id), AIMessage(content=new_content)]
        return updates

    return finalize


def make_llm(settings: Settings) -> ChatOpenAI:
    if not settings.llm_configured():
        raise RuntimeError("未配置 BASE_URL / API_KEY，无法创建 Chat 模型。")
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
    )


def route_after_agent(state: AgentState) -> str:
    messages = state["messages"]
    if not messages:
        return "finalize"
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"
    return "finalize"