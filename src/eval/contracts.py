from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graph.state import EvidenceStatus


@dataclass
class QuestionEval:
    question_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)


def _term_coverage(answer: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t and t in answer)


def _source_hits(answer: str, entries: list[dict], sources: list[str]) -> list[str]:
    hits: list[str] = []
    text = answer
    for e in entries:
        src = e.get("source") or ""
        if src:
            text += "\n" + src
    for src in sources:
        if src in text or any(src in (e.get("source") or "") for e in entries):
            hits.append(src)
    return hits


def evaluate_question(
    item: dict[str, Any],
    *,
    answer: str,
    tool_names: list[str],
    rag_calls_used: int,
    evidence_status: EvidenceStatus,
    evidence_entries: list[dict],
) -> QuestionEval:
    qid = item.get("id", "?")
    failures: list[str] = []
    contract = item.get("contract") or {}
    tools_spec = contract.get("tools") or {}
    answer_spec = contract.get("answer") or {}
    evidence_spec = contract.get("evidence") or {}
    rubric = item.get("rubric") or {}
    behavior = rubric.get("expectedBehavior", "answer")

    must_use = tools_spec.get("mustUseAny") or []
    if must_use and not any(t in tool_names for t in must_use):
        failures.append(f"tools.mustUseAny 未命中: 需要其一 {must_use}")

    max_rag = tools_spec.get("maxRagCalls")
    if isinstance(max_rag, int) and rag_calls_used > max_rag:
        failures.append(f"超过 maxRagCalls: {rag_calls_used} > {max_rag}")

    if behavior == "insufficient-evidence":
        required = answer_spec.get("requiredPhrasesAny") or [
            "没有找到",
            "未找到",
            "证据不足",
            "无法确认",
        ]
        if not any(p in answer for p in required):
            failures.append(f"拒答题未包含必需短语之一: {required}")
    else:
        expected_terms = item.get("expectedTerms") or []
        min_cov = answer_spec.get("minTermCoverage", 1)
        if expected_terms and _term_coverage(answer, expected_terms) < min_cov:
            failures.append(
                f"answer 关键词不足: 需要 {min_cov} 个，命中 {_term_coverage(answer, expected_terms)}"
            )

        expected_sources = evidence_spec.get("expectedSources") or []
        if not expected_sources and item.get("expectedSource"):
            expected_sources = [item["expectedSource"]]
        if evidence_spec.get("requireSourceHit") and expected_sources:
            hits = _source_hits(answer, evidence_entries, expected_sources)
            need_all = evidence_spec.get("requireAllExpectedSources", False)
            if need_all and len(hits) < len(expected_sources):
                failures.append(f"未引用全部期望来源: {expected_sources}")
            elif not need_all and not hits:
                failures.append(f"未命中期望来源: {expected_sources}")

        min_suff = evidence_spec.get("minimumSufficiency")
        if min_suff == "sufficient" and evidence_status != "sufficient":
            failures.append(f"证据未达 sufficient: {evidence_status}")

    return QuestionEval(question_id=qid, passed=not failures, failures=failures)