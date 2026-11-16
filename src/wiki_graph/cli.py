from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from wiki_graph.config import Settings
from wiki_graph.graph.builder import build_agent_graph
from wiki_graph.rag.index import build_index
from wiki_graph.rag.retriever import get_retriever


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
    from wiki_graph.graph.nodes import make_llm

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
    initial = {
        "messages": [HumanMessage(content=args.question)],
        "evidence_status": "insufficient",
        "rag_budget": settings.rag_budget,
        "active_tools": ["search_notes", "read_note"],
    }
    final = graph.invoke(initial, config)
    last = final["messages"][-1]
    content = last.content if isinstance(last.content, str) else str(last.content)
    print(content)
    return 0


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
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        raise SystemExit(1)
    raise SystemExit(handler(args))


if __name__ == "__main__":
    main()