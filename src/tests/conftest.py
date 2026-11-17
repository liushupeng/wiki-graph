from __future__ import annotations

import os


def pytest_configure(config) -> None:  # noqa: ARG001
    """单元测试默认 fake embedding；环境已设置时不覆盖。"""
    if not (os.getenv("WIKI_GRAPH_EMBEDDING") or "").strip():
        os.environ["WIKI_GRAPH_EMBEDDING"] = "fake"