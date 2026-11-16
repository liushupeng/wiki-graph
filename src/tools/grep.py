from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.tools import tool

from config import Settings


def _safe_path(corpus_dir: Path, rel: str) -> Path | None:
    rel = rel.replace("\\", "/").lstrip("/")
    candidate = (corpus_dir / rel).resolve()
    try:
        candidate.relative_to(corpus_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def grep_corpus(
    corpus_dir: Path,
    query: str,
    *,
    regex: bool = False,
    max_results: int = 10,
    context_lines: int = 1,
) -> list[dict]:
    max_results = min(max(max_results, 1), 30)
    context_lines = min(max(context_lines, 0), 2)
    corpus_dir = corpus_dir.resolve()
    pattern: re.Pattern[str] | None = None
    if regex:
        pattern = re.compile(query)
    else:
        pattern = re.compile(re.escape(query), re.IGNORECASE)

    matches: list[dict] = []
    for path in sorted(corpus_dir.rglob("*.md")):
        rel = path.relative_to(corpus_dir).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            if not pattern.search(line):
                continue
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            snippet = "\n".join(lines[start:end])
            matches.append(
                {
                    "source": rel,
                    "line": i + 1,
                    "excerpt": snippet,
                }
            )
            if len(matches) >= max_results:
                return matches
    return matches


def make_grep_notes_tool(settings: Settings):
    corpus_dir = settings.corpus_dir.resolve()

    @tool
    def grep_notes(
        query: str,
        regex: bool = False,
        max_results: int = 10,
        context_lines: int = 1,
    ) -> str:
        """在本地 Markdown 语料中做关键词或正则 grep。"""
        hits = grep_corpus(
            corpus_dir,
            query,
            regex=regex,
            max_results=max_results,
            context_lines=context_lines,
        )
        return json.dumps(
            {"query": query, "matches": hits, "count": len(hits)},
            ensure_ascii=False,
        )

    return grep_notes