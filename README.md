# wiki-graph

[wiki-agent](https://github.com/liushupeng/wiki-agent) 的 **Python / LangGraph 对照实现**：同一套知识库 Agent 产品故事（多步 Agent、混合 RAG、证据与预算、可评测），技术栈为 **LangChain + LangGraph**，**不是** wiki-agent 的 fork 替代。

设计背景与边界见根目录 [HANDOFF.md](./HANDOFF.md)。

## 技术栈

- Python 3.11+
- 依赖管理：**uv**（`uv sync`）
- LangGraph（Agent 图 + checkpoint）
- LangChain（Loader / Splitter / Embeddings / Chroma）
- 混合检索（向量 + BM25，简化版 RRF）

## 快速开始

```bash
uv sync
uv run python -m wiki_graph.cli index
uv run python -m wiki_graph.cli --check-llm
uv run python -m wiki_graph.cli ask "HTTP 缓存里 max-age=300 表示什么？"
```

## 环境变量

| 变量 | 含义 |
|------|------|
| `BASE_URL` | OpenAI 兼容 API base |
| `API_KEY` | API key |
| `MODEL_NAME` | 模型 id |
| `WIKI_GRAPH_CORPUS_DIR` | 语料目录，默认 `fixtures/corpus` |
| `WIKI_GRAPH_INDEX_DIR` | 索引持久化目录，默认 `.wiki_graph_index` |
| `RAG_BUDGET` | 最大 RAG 调用次数（M3+ 使用，默认 `4`） |
| `WIKI_GRAPH_EMBEDDING` | `openai`（默认）或 `fake`（仅测试 / 离线索引） |

## 里程碑（当前）

- [x] **M0**：`pyproject`、README、HANDOFF、`fixtures` 语料
- [x] **M1**：索引 CLI + `search_notes` / `read_note`（无 LLM）
- [x] **M2**：LangGraph `agent` ↔ `tools` 最小循环 + `--check-llm`
- [ ] M3：证据、`rag_budget`、藏工具、拒答闸门
- [ ] M4：e2e 评测子集
- [ ] M5：Mock 业务工具 + 对照表

## 语料溯源

`fixtures/corpus/` 来自 wiki-agent 公开快照 [`examples/public-knowledge`](https://github.com/liushupeng/wiki-agent/tree/main/examples/public-knowledge)。  
`fixtures/e2e/` 为评测题占位（M4 从 wiki-agent `evaluation/e2e/questions.json` 复制子集）。

## License

MIT — 与 wiki-agent 一致。