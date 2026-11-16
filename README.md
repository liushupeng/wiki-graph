# wiki-graph

[wiki-agent](https://github.com/liushupeng/wiki-agent) 的 **Python / LangGraph 对照实现**：用 **LangChain + LangGraph** 栈复刻「检索 → 证据感知 Agent → 有来源回答 / 明确拒答」的产品叙事，**不是** wiki-agent 的 fork 或替代品。

```text
fixtures corpus
  -> hybrid index (Chroma + BM25)
  -> LangGraph agent + RAG tools
  -> evidence status + RAG budget
  -> cited answer / explicit abstention
```

## 与 wiki-agent

| | [wiki-agent](https://github.com/liushupeng/wiki-agent) | wiki-graph（本仓） |
|---|------------------------|-------------------|
| 定位 | TypeScript 主实现：完整混合检索、治理、可观测、离线评测 | Python 对照：LangGraph 编排 + 简化 RAG + 证据预算 |
| 索引 | BM25 + BGE-M3 + LanceDB，离线索引管线 | Chroma + BM25，RRF 融合 |
| 运行时 | Native / Pi 双 Harness | LangGraph `agent` ↔ `tools` + `finalize` |
| 配置习惯 | `BASE_URL` / `API_KEY` / `MODEL_NAME` | 同名 LLM 变量 + `WIKI_GRAPH_*` / `RAG_BUDGET` |

生产级细节、评测数字与架构说明以母项目为准；本仓侧重 **可克隆、可索引、可 ask、可与母项目同题对照**。

## 架构

```mermaid
flowchart LR
    user["CLI"]
    graph["LangGraph<br/>agent ↔ tools → finalize"]
    llm["OpenAI-compatible LLM"]
    tools["search_notes / read_note"]
    idx[".wiki_graph_index<br/>Chroma + chunks.jsonl"]
    policy["Evidence + RAG budget<br/>hide tools / abstain"]

    user --> graph
    graph <--> llm
    graph --> tools --> idx
    tools --> policy --> graph
```

## 核心能力

- **混合 RAG（简化版）**：Markdown 语料分块；向量检索与 BM25 RRF 融合；`index` / `search` / `read` 可脱离 LLM 调试。
- **证据与预算**：RAG 工具结果记入库存，`evidence_status` 为 `insufficient | partial | sufficient`；`RAG_BUDGET` 耗尽后不再向模型暴露 RAG 工具；无证据时拒答模板收口。
- **工具命名**：`search_notes`、`read_note` 与 wiki-agent 对齐，便于共享 fixture 与评测叙事。
- **Checkpoint**：内存 checkpointer，便于后续多轮扩展。

## 快速开始

**要求**

- Python 3.11+
- OpenAI 兼容 Chat API（`ask` / `--check-llm`）
- Embedding：默认走 API；本地测试可设 `WIKI_GRAPH_EMBEDDING=fake`

**安装**

```bash
git clone https://github.com/liushupeng/wiki-graph.git
cd wiki-graph
uv sync
# 或: pip install -e ".[dev]"
```

**配置**

复制 [.env.example](.env.example) 为 `.env`（Linux/macOS：`cp .env.example .env`）。

至少在 `.env` 中配置：

```env
BASE_URL=https://api.deepseek.com
API_KEY=your_api_key
MODEL_NAME=your_model
```

**索引与问答**

```bash
python -m wiki_graph.cli index
python -m wiki_graph.cli --check-llm
python -m wiki_graph.cli ask "HTTP 缓存里 max-age=300 表示什么？"
```

### 常用 CLI

| 命令 | 说明 |
|------|------|
| `index` | 从语料目录构建 `.wiki_graph_index` |
| `search <query>` | 仅检索，不调用 LLM |
| `read <path>` | 按相对路径读笔记片段 |
| `--check-llm` | 探测 Chat API |
| `ask <question>` | LangGraph Agent 单次问答 |

`pip install -e` 后也可使用 **`wiki-graph`**（与 `python -m wiki_graph.cli` 等价）。

## 开发与验证

```bash
# 无 API 的索引与测试
export WIKI_GRAPH_EMBEDDING=fake   # Windows: $env:WIKI_GRAPH_EMBEDDING="fake"
python -m wiki_graph.cli index
pytest -q
```

## 环境变量

| 变量 | 含义 |
|------|------|
| `BASE_URL` | OpenAI 兼容 API base |
| `API_KEY` | API key |
| `MODEL_NAME` | 模型 id |
| `WIKI_GRAPH_CORPUS_DIR` | 语料目录，默认 `fixtures/corpus` |
| `WIKI_GRAPH_INDEX_DIR` | 索引目录，默认 `.wiki_graph_index` |
| `RAG_BUDGET` | 单次对话内 RAG 工具调用上限，默认 `4` |
| `WIKI_GRAPH_EMBEDDING` | `openai`（默认）或 `fake`（测试 / 无 Embedding API） |

完整说明见 [.env.example](.env.example)。

## 语料

`fixtures/corpus/` 来自 wiki-agent 公开目录 [`examples/public-knowledge`](https://github.com/liushupeng/wiki-agent/tree/main/examples/public-knowledge)。索引产物写入 `.wiki_graph_index/`（本地生成，不入库）。

## 项目边界

- 本地 CLI，无 Web UI、无多租户部署面
- 不复制 wiki-agent 的 BGE / LanceDB 离线索引与完整 TaskContract / Trace 管线
- 私有语料、API 密钥与索引目录不进入 Git
- LLM 与 Embedding 行为受供应商与模型版本影响

## License

[MIT](LICENSE)