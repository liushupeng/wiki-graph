from __future__ import annotations

from pathlib import Path

from tools.grep import grep_corpus


def test_grep_corpus_hits_http_caching() -> None:
    repo = Path(__file__).resolve().parents[3]
    corpus = repo / "fixtures" / "corpus"
    hits = grep_corpus(corpus, "max-age=300", max_results=5)
    assert hits
    assert any("http-caching" in h["source"] for h in hits)