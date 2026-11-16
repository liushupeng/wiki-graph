from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from config import Settings
from eval.run_e2e import run_e2e_suite
from graph.builder import build_agent_graph
from rag.index import build_index
from rag.retriever import get_retriever
from tools import DEFAULT_ACTIVE_TOOLS


def _load_env() -> None:
    load_dotenv()


def cmd_index(_: argparse.Namespace) -> int:
    settings = Settings.from_env()
    out = build_index(settings)
    print(f"索引已写入: {out}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    retriever = get_retriever(Settings.from_env())
    hits = retriever.search(args.query, top_k=args.top_k)
    for h in hits:
        print(f"--- {h.source} ---")
        print(h.content[:500])
        if len(h.content) > 500:
            print("...")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
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


def cmd_ask(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    graph = build_agent_graph(settings)
    thread_id = args.thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial: dict = {
        "messages": [HumanMessage(content=args.question)],
        "evidence_status": "insufficient",
        "rag_budget": settings.rag_budget,
        "rag_calls_used": 0,
        "evidence_entries": [],
        "active_tools": list(DEFAULT_ACTIVE_TOOLS),
        "turn_metadata": {"mode": "ask"},
    }
    if args.stream:
        last_content = ""
        for event in graph.stream(initial, config, stream_mode="values"):
            messages = event.get("messages") or []
            if not messages:
                continue
            last = messages[-1]
            content = getattr(last, "content", "")
            text = content if isinstance(content, str) else str(content)
            if text and text != last_content:
                delta = text[len(last_content) :] if text.startswith(last_content) else text
                print(delta, end="", flush=True)
                last_content = text
        print()
        return 0

    final = graph.invoke(initial, config)
    last = final["messages"][-1]
    content = last.content if isinstance(last.content, str) else str(last.content)
    print(content)
    return 0


def cmd_repl(_: argparse.Namespace) -> int:
    settings = Settings.from_env()
    graph = build_agent_graph(settings)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print("wiki-graph REPL（空行退出）")
    bootstrapped = False
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        if not bootstrapped:
            state = {
                "messages": [HumanMessage(content=line)],
                "evidence_status": "insufficient",
                "rag_budget": settings.rag_budget,
                "rag_calls_used": 0,
                "evidence_entries": [],
                "active_tools": list(DEFAULT_ACTIVE_TOOLS),
                "turn_metadata": {"mode": "repl"},
            }
            bootstrapped = True
        else:
            state = {"messages": [HumanMessage(content=line)]}
        final = graph.invoke(state, config)
        last = final["messages"][-1]
        content = last.content if isinstance(last.content, str) else str(last.content)
        print(content)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
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

    p.add_argument("--check-llm", action="store_true", help="探测 Chat API 是否可用")

    ap = sub.add_parser("ask", help="单次提问（LangGraph Agent）")
    ap.add_argument("question")
    ap.add_argument("--thread", default=None, help="checkpoint thread id（多轮预留）")
    ap.add_argument("--stream", action="store_true", help="流式打印最终回答增量")

    sub.add_parser("repl", help="多轮 REPL（共享 checkpoint thread）")

    ep = sub.add_parser("eval", help="运行 fixtures/e2e 子集（需 LLM API）")
    ep.add_argument("--limit", type=int, default=None, help="仅跑前 N 题")

    return p


def main(argv: list[str] | None = None) -> None:
    _load_env()
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    if argv == ["--check-llm"]:
        raise SystemExit(cmd_check_llm(argparse.Namespace()))
    args = parser.parse_args(argv)
    if args.command is None and not getattr(args, "check_llm", False):
        parser.print_help()
        raise SystemExit(0)
    dispatch = {
        "index": cmd_index,
        "search": cmd_search,
        "read": cmd_read,
        "ask": cmd_ask,
        "repl": cmd_repl,
        "eval": cmd_eval,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        raise SystemExit(1)
    raise SystemExit(handler(args))


if __name__ == "__main__":
    main()