from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from langchain_community.embeddings import FakeEmbeddings
from langchain_openai import OpenAIEmbeddings

from config import DEFAULT_LOCAL_EMBEDDING_MODEL, Settings


class LocalEmbeddingModelMissing(RuntimeError):
    """本地 embedding 目录缺失或文件不完整。"""


def local_embedding_setup_help(
    *,
    model_id: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
    target_dir: Path | None = None,
) -> str:
    target = target_dir or Path("models") / "bge-small-zh-v1.5"
    return f"""离线索引需要本地 embedding 权重（不自动联网下载）。

请将模型「{model_id}」的完整仓库放到目录：
  {target.resolve()}

任选来源下载后解压/拷贝到该目录（需含 config.json 与 *.safetensors 或 pytorch_model.bin）：
  https://huggingface.co/{model_id}
  https://hf-mirror.com/{model_id}

自定义路径：.env 中设置 WIKI_GRAPH_EMBEDDING_MODEL_PATH=你的目录
临时跳过向量、仅用 BM25：WIKI_GRAPH_EMBEDDING=fake"""


def _validate_local_model_dir(path: Path, model_id: str) -> None:
    if not path.is_dir():
        raise LocalEmbeddingModelMissing(local_embedding_setup_help(model_id=model_id, target_dir=path))
    if not (path / "config.json").is_file():
        raise LocalEmbeddingModelMissing(
            f"目录 {path} 缺少 config.json。\n\n{local_embedding_setup_help(model_id=model_id, target_dir=path)}"
        )
    has_weights = any(path.glob("*.safetensors")) or (path / "pytorch_model.bin").is_file()
    if not has_weights:
        raise LocalEmbeddingModelMissing(
            f"目录 {path} 缺少权重文件（*.safetensors 或 pytorch_model.bin）。\n\n"
            f"{local_embedding_setup_help(model_id=model_id, target_dir=path)}"
        )


@contextmanager
def _quiet_ml_loading():
    """CLI 下加载 sentence-transformers 时关闭 tqdm 进度条。"""
    prev = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        try:
            from transformers.utils import logging as tf_logging

            tf_logging.set_verbosity_error()
        except ImportError:
            pass
        yield
    finally:
        if prev is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = prev


def _local_embeddings(settings: Settings):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError("请安装 langchain-huggingface：pip install -e '.[dev]'") from exc
    path = settings.embedding_model_path
    _validate_local_model_dir(path, settings.embedding_model)
    with _quiet_ml_loading():
        return HuggingFaceEmbeddings(
            model_name=str(path.resolve()),
            model_kwargs={"device": settings.embedding_device, "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )


def make_embeddings(settings: Settings) -> FakeEmbeddings | OpenAIEmbeddings:
    mode = settings.embedding_mode
    if mode == "fake":
        return FakeEmbeddings(size=384)
    if mode == "openai":
        if not settings.embeddings_api_configured():
            raise RuntimeError(
                "WIKI_GRAPH_EMBEDDING=openai 时需要 EMBEDDING_BASE_URL 与 EMBEDDING_API_KEY"
            )
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.embedding_api_key,
            openai_api_base=settings.embedding_base_url,
        )
    return _local_embeddings(settings)