from __future__ import annotations

from eval.contracts import evaluate_question
from eval.run_e2e import load_questions
from policy.evidence import ABSTAIN_MESSAGE, finalize_content


def test_finalize_long_ungrounded() -> None:
    text = "2032 年量子处理器逻辑错误率为 0.001%，该结论来自虚构基准测试，应当被拒绝当作事实断言。"
    out, action = finalize_content(
        text,
        rag_calls_used=1,
        evidence_count=0,
        evidence_status="insufficient",
    )
    assert action == "forced-abstention"
    assert out == ABSTAIN_MESSAGE


def test_contract_abstain_question_phrase() -> None:
    questions = load_questions()
    negative = next(q for q in questions if q["id"] == "e2e-negative-quantum-error-rate")
    good = evaluate_question(
        negative,
        answer="公开语料中未找到相关证据，无法确认该数值。",
        tool_names=["search_notes"],
        rag_calls_used=1,
        evidence_status="insufficient",
        evidence_entries=[],
    )
    assert good.passed
    bad = evaluate_question(
        negative,
        answer="错误率是 0.001%。",
        tool_names=["search_notes"],
        rag_calls_used=1,
        evidence_status="sufficient",
        evidence_entries=[{"source": "dev/a.md"}],
    )
    assert not bad.passed


def test_contract_answer_terms() -> None:
    questions = load_questions()
    item = next(q for q in questions if q["id"] == "e2e-single-http-cache")
    ok = evaluate_question(
        item,
        answer="max-age=300 表示 300 秒内可视为新鲜，来源 dev/http-caching.md",
        tool_names=["read_note"],
        rag_calls_used=1,
        evidence_status="sufficient",
        evidence_entries=[{"source": "dev/http-caching.md"}],
    )
    assert ok.passed