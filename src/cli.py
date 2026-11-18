from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Callable, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import Settings
from graph.state import AgentState, initial_agent_state


def _load_env() -> None:
    load_dotenv()


def _find_latest_assistant(messages: Sequence[BaseMessage]) -> AIMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _assistant_text(message: AIMessage) -> str:
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


def _assistant_has_tool_calls(message: AIMessage) -> bool:
    tool_calls = getattr(message, "tool_calls", None) or []
    return len(tool_calls) > 0


def _tool_call_names(message: AIMessage) -> list[str]:
    names: list[str] = []
    for call in getattr(message, "tool_calls", None) or []:
        if isinstance(call, dict):
            name = call.get("name")
        else:
            name = getattr(call, "name", None)
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


class _AssistantStreamEmitter:
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


def _index_embedding_api_help(settings: Settings) -> str:
    return (
        "在线 Embedding API 调用失败（模式 "
        f"{settings.embedding_mode!r}，模型 {settings.embedding_model!r}）。\n"
        "离线索引请用 WIKI_GRAPH_EMBEDDING=local；仅 Agent 问答需要 Chat API。"
    )


def cmd_index(_: argparse.Namespace) -> int:
    from rag.embeddings import LocalEmbeddingModelMissing
    from rag.index import build_index

    settings = Settings.from_env()
    try:
        out = build_index(settings)
    except LocalEmbeddingModelMissing as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        err = str(exc).lower()
        if "404" in err or exc.__class__.__name__ in {"NotFoundError", "APIStatusError"}:
            print(_index_embedding_api_help(settings), file=sys.stderr)
            print(f"\n{exc}", file=sys.stderr)
            return 1
        raise
    print(
        f"embedding={settings.embedding_mode} model={settings.resolved_embedding_model()} "
        f"device={settings.embedding_device}"
    )
    print(f"索引已写入: {out}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from rag.retriever import get_retriever

    retriever = get_retriever(Settings.from_env())
    hits = retriever.search(args.query, top_k=args.top_k)
    for h in hits:
        print(f"--- {h.source} ---")
        print(h.content[:500])
        if len(h.content) > 500:
            print("...")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    from rag.retriever import get_retriever

    retriever = get_retriever(Settings.from_env())
    chunks = retriever.read_note(args.path, query=args.query, max_chunks=args.max_chunks)
    print(json.dumps(chunks, ensure_ascii=False, indent=2))
    return 0


def cmd_check_llm(_: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if not settings.llm_configured():
        print("FAIL: 请设置 BASE_URL 与 API_KEY", file=sys.stderr)
        return 1
    from graph.nodes import make_llm

    llm = make_llm(settings)
    reply = llm.invoke("Reply with exactly: ok")
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    print(f"model={settings.model_name}")
    print(f"reply={text.strip()[:200]}")
    print("OK")
    return 0


def _bootstrap_state(question: str, settings: Settings) -> AgentState:
    from tools import DEFAULT_ACTIVE_TOOLS

    return initial_agent_state(
        question,
        rag_budget=settings.rag_budget,
        active_tools=list(DEFAULT_ACTIVE_TOOLS),
        turn_metadata={"mode": "ask"},
    )


def _final_assistant_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not _assistant_has_tool_calls(msg):
            return _assistant_text(msg)
    ai = _find_latest_assistant(messages)
    return _assistant_text(ai) if ai else ""


def _log_tool_plan(ai: AIMessage) -> None:
    names = _tool_call_names(ai)
    if not names:
        print("正在检索知识库…", file=sys.stderr, flush=True)
        return
    label = "、".join(names)
    print(f"正在调用工具：{label}…", file=sys.stderr, flush=True)


def _run_turn(graph, state: AgentState, config: dict, *, no_stream: bool) -> None:
    if not no_stream:
        emitter = _AssistantStreamEmitter(lambda chunk: print(chunk, end="", flush=True))
        last_tool_signature: tuple[str, ...] | None = None
        for event in graph.stream(state, config, stream_mode="values"):
            messages = event.get("messages") or []
            ai = _find_latest_assistant(messages)
            if ai is None:
                continue
            if _assistant_has_tool_calls(ai):
                signature = tuple(_tool_call_names(ai))
                if signature != last_tool_signature:
                    _log_tool_plan(ai)
                    last_tool_signature = signature
                emitter.reset()
                continue
            last_tool_signature = None
            emitter.emit(_assistant_text(ai))
        print()
        return
    final = graph.invoke(state, config)
    print(_final_assistant_text(final.get("messages") or []))


def cmd_ask(args: argparse.Namespace) -> int:
    from graph.builder import build_agent_graph

    settings = Settings.from_env()
    interactive = args.question is None
    if interactive:
        print(
            "正在初始化（加载混合索引与 embedding，首次可能较慢）…",
            file=sys.stderr,
            flush=True,
        )
    elif not args.no_stream:
        print("正在处理问题…", file=sys.stderr, flush=True)
    graph = build_agent_graph(settings)
    thread_id = args.thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    if interactive:
        print("就绪。请输入问题（多轮同一会话，空行退出）：", file=sys.stderr, flush=True)
        bootstrapped = False
        while True:
            try:
                line = input("\n你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                break
            if not bootstrapped:
                state = _bootstrap_state(line, settings)
                bootstrapped = True
            else:
                state: AgentState = {"messages": [HumanMessage(content=line)]}
            _run_turn(graph, state, config, no_stream=args.no_stream)
        return 0

    _run_turn(graph, _bootstrap_state(args.question, settings), config, no_stream=args.no_stream)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from eval.run_e2e import run_e2e_suite

    settings = Settings.from_env()
    try:
        suite = run_e2e_suite(settings, limit=args.limit)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    for row in suite.results:
        status = "PASS" if row.passed else "FAIL"
        print(f"{status} {row.question_id}")
        for f in row.failures:
            print(f"  - {f}")
    print(f"\n{suite.passed}/{suite.total} passed")
    return 0 if suite.passed == suite.total else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wiki-graph", description="wiki-graph CLI")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("index", help="从 fixtures 语料构建混合检索索引")

    sp = sub.add_parser("search", help="无 LLM 检索测试")
    sp.add_argument("query")
    sp.add_argument("--top-k", type=int, default=5)

    rp = sub.add_parser("read", help="无 LLM 读取笔记片段")
    rp.add_argument("path", help="相对路径，如 dev/http-caching.md")
    rp.add_argument("--query", default=None)
    rp.add_argument("--max-chunks", type=int, default=8)

    sub.add_parser("check-llm", help="探测 Chat API 是否可用")

    ap = sub.add_parser("ask", help="提问；不写问题则进入多轮交互")
    ap.add_argument("question", nargs="?", default=None, help="问题正文；省略则空行退出")
    ap.add_argument("--thread", default=None, help="checkpoint thread id（多轮预留）")
    ap.add_argument(
        "--no-stream",
        action="store_true",
        help="等待图执行完毕后一次性输出（脚本/重定向时用）",
    )

    ep = sub.add_parser("eval", help="运行 fixtures/e2e 子集（需 LLM API）")
    ep.add_argument("--limit", type=int, default=None, help="仅跑前 N 题")

    return p


def main(argv: list[str] | None = None) -> None:
    _load_env()
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(0)
    dispatch = {
        "index": cmd_index,
        "search": cmd_search,
        "read": cmd_read,
        "check-llm": cmd_check_llm,
        "ask": cmd_ask,
        "eval": cmd_eval,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        raise SystemExit(1)
    raise SystemExit(handler(args))


if __name__ == "__main__":
    main()