from __future__ import annotations

import json
from typing import Literal, TypedDict

EvidenceStatus = Literal["insufficient", "partial", "sufficient"]

ABSTAIN_MESSAGE = (
    "知识库检索没有返回可引用证据，因此我无法确认这个问题的答案。"
    "请补充相关笔记、来源或更具体的检索线索。"
)


class EvidenceEntry(TypedDict, total=False):
    tool_name: str
    source: str
    chunk_id: str
    query: str
    citation: str


def _is_rag_tool(name: str | None) -> bool:
    if not name:
        return False
    for base in ("search_notes", "grep_notes", "read_note"):
        if name == base or name.endswith(f"_{base}"):
            return True
    return False


def record_tool_result(tool_name: str, raw_content: str) -> list[EvidenceEntry]:
    if not _is_rag_tool(tool_name):
        return []
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    entries: list[EvidenceEntry] = []
    query = data.get("query") if isinstance(data.get("query"), str) else None
    note_path = data.get("notePath") if isinstance(data.get("notePath"), str) else None

    for item in data.get("matches") or []:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        if not isinstance(source, str) or not source.strip():
            continue
        entry: EvidenceEntry = {
            "tool_name": tool_name,
            "source": source.strip(),
            "citation": source.strip(),
        }
        chunk_id = item.get("chunkId")
        if isinstance(chunk_id, str):
            entry["chunk_id"] = chunk_id
        if query:
            entry["query"] = query
        entries.append(entry)

    for item in data.get("chunks") or []:
        if not isinstance(item, dict):
            continue
        source = item.get("source") if isinstance(item.get("source"), str) else note_path
        if not isinstance(source, str) or not source.strip():
            continue
        entries.append(
            {
                "tool_name": tool_name,
                "source": source.strip(),
                "citation": source.strip(),
                "query": query,
            }
        )
    return entries


def assess_evidence(entries: list[EvidenceEntry]) -> tuple[EvidenceStatus, list[str]]:
    if not entries:
        return "insufficient", ["无已记录证据"]

    distinct = len({e["source"] for e in entries if e.get("source")})
    reasons: list[str] = []
    if len(entries) < 1:
        reasons.append("需要至少 1 条证据")
    if distinct < 1:
        reasons.append("需要至少 1 个来源")

    if not reasons:
        return "sufficient", []
    return "partial", reasons


def build_model_feedback(
    *,
    evidence_status: EvidenceStatus,
    evidence_count: int,
    distinct_sources: int,
    rag_calls_used: int,
    rag_budget: int,
) -> str:
    exhausted = rag_budget <= 0
    return "\n".join(
        [
            "[RUNTIME_EVIDENCE_POLICY]",
            f"evidence_inventory={evidence_status}",
            f"evidence_count={evidence_count}",
            f"distinct_sources={distinct_sources}",
            f"rag_calls_used={rag_calls_used}",
            f"rag_calls_remaining={rag_budget}",
            "注意：inventory 只描述已检索材料数量，不代表材料已经覆盖用户问题。",
            (
                "RAG 预算已耗尽：不得继续检索；请回答已证实部分并明确缺口，或拒绝作无证据断言。"
                if exhausted
                else "请判断关键问题是否已被证据覆盖；充分则停止检索，不充分时仅做必要的下一次检索。"
            ),
        ]
    )


def finalize_content(
    content: str,
    *,
    rag_calls_used: int,
    evidence_count: int,
    evidence_status: EvidenceStatus,
) -> tuple[str, Literal["allow", "forced-abstention"]]:
    if rag_calls_used > 0 and evidence_count == 0:
        return ABSTAIN_MESSAGE, "forced-abstention"
    if evidence_status == "insufficient" and rag_calls_used > 0 and evidence_count == 0:
        return ABSTAIN_MESSAGE, "forced-abstention"
    return content, "allow"