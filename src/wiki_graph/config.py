from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    base_url: str | None
    api_key: str | None
    model_name: str
    corpus_dir: Path
    index_dir: Path
    rag_budget: int
    embedding_mode: str

    @classmethod
    def from_env(cls) -> Settings:
        corpus = os.getenv("WIKI_GRAPH_CORPUS_DIR")
        index = os.getenv("WIKI_GRAPH_INDEX_DIR")
        return cls(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            corpus_dir=Path(corpus) if corpus else _REPO_ROOT / "fixtures" / "corpus",
            index_dir=Path(index) if index else _REPO_ROOT / ".wiki_graph_index",
            rag_budget=_env_int("RAG_BUDGET", 4),
            embedding_mode=os.getenv("WIKI_GRAPH_EMBEDDING", "openai").lower(),
        )

    def llm_configured(self) -> bool:
        return bool(self.api_key and self.base_url)