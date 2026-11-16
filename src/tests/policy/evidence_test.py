from __future__ import annotations

import json

from policy.evidence import (
    ABSTAIN_MESSAGE,
    assess_evidence,
    finalize_content,
    record_tool_result,
)
from policy.tool_budget import apply_budget_to_active_tools


def test_record_search_matches() -> None:
    payload = json.dumps(
        {
            "matches": [{"source": "dev/http-caching.md", "excerpt": "x", "chunkId": "abc"}],
            "count": 1,
        }
    )
    entries = record_tool_result("search_notes", payload)
    assert len(entries) == 1
    assert entries[0]["source"] == "dev/http-caching.md"


def test_assess_sufficient() -> None:
    entries = [{"tool_name": "search_notes", "source": "dev/a.md", "citation": "dev/a.md"}]
    status, reasons = assess_evidence(entries)
    assert status == "sufficient"
    assert not reasons


def test_finalize_forced_abstention() -> None:
    content, action = finalize_content(
        "答案是 42。",
        rag_calls_used=2,
        evidence_count=0,
        evidence_status="insufficient",
    )
    assert action == "forced-abstention"
    assert content == ABSTAIN_MESSAGE


def test_budget_hides_rag_tools() -> None:
    active = apply_budget_to_active_tools(["search_notes", "read_note"], 0)
    assert active == []