from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from config import Settings
from eval.contracts import QuestionEval, evaluate_question
from eval.trajectory import final_answer_text, rag_call_count, tool_names_from_messages
from graph.builder import build_agent_graph
from graph.state import initial_agent_state
from tools import DEFAULT_ACTIVE_TOOLS

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = _REPO_ROOT / "fixtures" / "e2e" / "questions.json"


@dataclass
class SuiteResult:
    total: int
    passed: int
    results: list[QuestionEval]


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or DEFAULT_QUESTIONS
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "questions" in raw:
        return raw["questions"]
    if isinstance(raw, list):
        return raw
    raise ValueError(f"无法解析题集: {path}")


def run_e2e_suite(
    settings: Settings | None = None,
    *,
    questions_path: Path | None = None,
    limit: int | None = None,
) -> SuiteResult:
    settings = settings or Settings.from_env()
    if not settings.llm_configured():
        raise RuntimeError("E2E 需要配置 BASE_URL 与 API_KEY")

    questions = load_questions(questions_path)
    if limit is not None:
        questions = questions[:limit]

    graph = build_agent_graph(settings, checkpointer=False)
    results: list[QuestionEval] = []

    for item in questions:
        qid = item.get("id", "?")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial = initial_agent_state(
            item["question"],
            rag_budget=settings.rag_budget,
            active_tools=list(DEFAULT_ACTIVE_TOOLS),
            turn_metadata={"question_id": qid},
        )
        final = graph.invoke(initial, config)
        messages = final.get("messages") or []
        tool_names = tool_names_from_messages(messages)
        rag_used = int(final.get("rag_calls_used", rag_call_count(tool_names)))
        answer = final_answer_text(messages)
        ev = evaluate_question(
            item,
            answer=answer,
            tool_names=tool_names,
            rag_calls_used=rag_used,
            evidence_status=final.get("evidence_status", "insufficient"),
            evidence_entries=final.get("evidence_entries") or [],
        )
        results.append(ev)

    passed = sum(1 for r in results if r.passed)
    return SuiteResult(total=len(results), passed=passed, results=results)