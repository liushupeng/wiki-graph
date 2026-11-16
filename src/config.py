from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 与 wiki-agent `config/rag.indexes.jsonc` 的 `indexDirectory` + `{profile}/{variant}` 习惯对齐（单一对照 profile）。
DEFAULT_INDEX_ROOT = "var/indexes"
DEFAULT_INDEX_PROFILE = "wiki-graph-v1"
DEFAULT_INDEX_VARIANT = "raw-wiki"
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_EMBEDDING_MODEL_DIR = Path("models") / "bge-small-zh-v1.5"


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
    index_profile: str
    index_variant: str
    rag_budget: int
    embedding_mode: str
    embedding_model: str
    embedding_base_url: str | None
    embedding_api_key: str | None
    embedding_device: str
    embedding_model_path: Path

    @staticmethod
    def _first_env(*names: str) -> str | None:
        for name in names:
            raw = os.getenv(name)
            if raw is not None and raw.strip():
                return raw.strip()
        return None

    @staticmethod
    def _index_profile_and_variant() -> tuple[str, str]:
        profile = os.getenv("WIKI_GRAPH_INDEX_PROFILE", DEFAULT_INDEX_PROFILE)
        variant = (
            os.getenv("WIKI_GRAPH_INDEX_VARIANT")
            or os.getenv("RAG_INDEX_VARIANT")
            or DEFAULT_INDEX_VARIANT
        )
        return profile, variant

    @classmethod
    def _resolve_embedding_mode(cls) -> str:
        raw = os.getenv("WIKI_GRAPH_EMBEDDING", "local").lower()
        if raw == "auto":
            if cls._first_env("EMBEDDING_BASE_URL", "RAG_EMBEDDING_BASE_URL"):
                return "openai"
            return "local"
        return raw

    @classmethod
    def _default_embedding_model(cls, mode: str) -> str:
        if mode == "openai":
            return "text-embedding-3-small"
        if mode == "local":
            return DEFAULT_LOCAL_EMBEDDING_MODEL
        return DEFAULT_LOCAL_EMBEDDING_MODEL

    @classmethod
    def _resolve_index_dir(cls, profile: str, variant: str) -> Path:
        explicit = os.getenv("WIKI_GRAPH_INDEX_DIR")
        if explicit:
            return Path(explicit)
        root = os.getenv("WIKI_GRAPH_INDEX_ROOT", DEFAULT_INDEX_ROOT)
        return _REPO_ROOT / root / profile / variant

    @classmethod
    def from_env(cls) -> Settings:
        corpus = os.getenv("WIKI_GRAPH_CORPUS_DIR")
        profile, variant = cls._index_profile_and_variant()
        base_url = os.getenv("BASE_URL")
        api_key = os.getenv("API_KEY")
        embedding_base = cls._first_env("EMBEDDING_BASE_URL", "RAG_EMBEDDING_BASE_URL") or base_url
        embedding_key = cls._first_env("EMBEDDING_API_KEY", "RAG_EMBEDDING_API_KEY") or api_key
        embedding_mode = cls._resolve_embedding_mode()
        model_path = os.getenv("WIKI_GRAPH_EMBEDDING_MODEL_PATH")
        embedding_dir = (
            Path(model_path) if model_path else (_REPO_ROOT / DEFAULT_EMBEDDING_MODEL_DIR)
        )
        return cls(
            base_url=base_url,
            api_key=api_key,
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            corpus_dir=Path(corpus) if corpus else _REPO_ROOT / "fixtures" / "corpus",
            index_dir=cls._resolve_index_dir(profile, variant),
            index_profile=profile,
            index_variant=variant,
            rag_budget=_env_int("RAG_BUDGET", 4),
            embedding_mode=embedding_mode,
            embedding_model=os.getenv(
                "EMBEDDING_MODEL_NAME", cls._default_embedding_model(embedding_mode)
            ),
            embedding_base_url=embedding_base,
            embedding_api_key=embedding_key,
            embedding_device=os.getenv("WIKI_GRAPH_EMBEDDING_DEVICE", "cpu"),
            embedding_model_path=embedding_dir,
        )

    def llm_configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    def embeddings_api_configured(self) -> bool:
        return bool(self.embedding_api_key and self.embedding_base_url)

    def resolved_embedding_model(self) -> str:
        if self.embedding_mode == "local":
            return str(self.embedding_model_path.resolve())
        return self.embedding_model