from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cli_streaming import (
    AssistantStreamEmitter,
    assistant_has_tool_calls,
    find_latest_assistant,
    tool_call_names,
)


def test_find_latest_assistant_skips_non_assistant_tail():
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="draft", tool_calls=[{"name": "search_notes", "args": {}, "id": "1"}]),
        SystemMessage(content="[RUNTIME_EVIDENCE_POLICY]"),
    ]
    ai = find_latest_assistant(messages)
    assert ai is not None
    assert ai.content == "draft"
    assert assistant_has_tool_calls(ai)


def test_assistant_stream_emitter_emits_incremental_deltas():
    chunks: list[str] = []
    emitter = AssistantStreamEmitter(chunks.append)
    emitter.emit("hello")
    emitter.emit("hello world")
    assert chunks == ["hello", " world"]


def test_assistant_stream_emitter_reset_starts_fresh_prefix():
    chunks: list[str] = []
    emitter = AssistantStreamEmitter(chunks.append)
    emitter.emit("hello")
    emitter.reset()
    emitter.emit("world")
    assert chunks == ["hello", "world"]


def test_tool_call_names():
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "search_notes", "args": {"query": "x"}, "id": "1"}],
    )
    assert tool_call_names(ai) == ["search_notes"]