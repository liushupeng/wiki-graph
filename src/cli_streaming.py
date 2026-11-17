from __future__ import annotations

from typing import Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage


def find_latest_assistant(messages: Sequence[BaseMessage]) -> AIMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def assistant_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content) if content is not None else ""


def assistant_has_tool_calls(message: AIMessage) -> bool:
    tool_calls = getattr(message, "tool_calls", None) or []
    return len(tool_calls) > 0


def tool_call_names(message: AIMessage) -> list[str]:
    names: list[str] = []
    for call in getattr(message, "tool_calls", None) or []:
        if isinstance(call, dict):
            name = call.get("name")
        else:
            name = getattr(call, "name", None)
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


class AssistantStreamEmitter:
    """增量输出助手全文，避免重复刷屏（对齐 wiki-agent assistantStream.ts）。"""

    def __init__(self, on_chunk: Callable[[str], None]) -> None:
        self._on_chunk = on_chunk
        self._streamed = ""

    def reset(self) -> None:
        self._streamed = ""

    def emit(self, full_text: str) -> None:
        if not full_text or full_text == self._streamed:
            return
        if full_text.startswith(self._streamed):
            delta = full_text[len(self._streamed) :]
            if delta:
                self._on_chunk(delta)
        else:
            self._on_chunk(f"\n{full_text}" if self._streamed else full_text)
        self._streamed = full_text