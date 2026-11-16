from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings
from rag.embeddings import make_embeddings
from rag.retriever import CHUNKS_FILE, MANIFEST_FILE

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def _relative_source(corpus_dir: Path, path: Path) -> str:
    rel = path.relative_to(corpus_dir).as_posix()
    return rel


def build_index(settings: Settings | None = None, *, reset: bool = True) -> Path:
    settings = settings or Settings.from_env()
    corpus_dir = settings.corpus_dir.resolve()
    index_dir = settings.index_dir.resolve()

    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"语料目录不存在: {corpus_dir}")

    if reset and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    loader = DirectoryLoader(
        str(corpus_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )
    raw_docs = loader.load()
    for doc in raw_docs:
        src_path = Path(doc.metadata.get("source", ""))
        doc.metadata["source"] = _relative_source(corpus_dir, src_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n", " "],
    )
    splits = splitter.split_documents(raw_docs)

    chunk_rows: list[dict] = []
    lc_docs: list[Document] = []
    per_source_index: dict[str, int] = {}
    for doc in splits:
        chunk_id = str(uuid.uuid4())
        source = doc.metadata.get("source", "")
        chunk_index = per_source_index.get(source, 0)
        per_source_index[source] = chunk_index + 1
        row = {
            "id": chunk_id,
            "source": source,
            "content": doc.page_content,
            "chunk_index": chunk_index,
        }
        chunk_rows.append(row)
        lc_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": source, "chunk_id": chunk_id},
            )
        )

    chunks_path = index_dir / CHUNKS_FILE
    with chunks_path.open("w", encoding="utf-8") as f:
        for row in chunk_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    chroma_dir = index_dir / "chroma"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    Chroma.from_documents(
        lc_docs,
        embedding=make_embeddings(settings),
        collection_name="wiki_graph",
        persist_directory=str(chroma_dir),
    )

    manifest = {
        "version": 1,
        "profile": settings.index_profile,
        "variant": settings.index_variant,
        "corpus_dir": str(corpus_dir),
        "chunk_count": len(chunk_rows),
        "chunkSize": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "embedding_mode": settings.embedding_mode,
        "embedding_model": settings.embedding_model,
        "stores": {"sparse": CHUNKS_FILE, "dense": "chroma/"},
    }
    with (index_dir / MANIFEST_FILE).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return index_dir