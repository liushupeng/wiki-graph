from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from policy.tool_budget import RAG_TOOL_NAMES


def tool_names_from_messages(messages: list[BaseMessage]) -> list[str]:
    names: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name:
                    names.append(name)
        elif isinstance(msg, ToolMessage) and msg.name:
            names.append(msg.name)
    return names


def rag_call_count(tool_names: list[str]) -> int:
    return sum(1 for n in tool_names if n in RAG_TOOL_NAMES)


def final_answer_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""