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

## 源码布局

与 [wiki-agent `src/`](https://github.com/liushupeng/wiki-agent/tree/main/src) 类似：`src/` 按功能分目录（`graph`、`tools`、`rag`、`policy`、`eval`），离线测试在 `src/tests/` 下与模块一一对应；CLI 实现在 `src/cli.py`。

## 架构

```mermaid
flowchart LR
    user["CLI"]
    agentLoop["LangGraph: agent, tools, finalize"]
    llm["OpenAI-compatible LLM"]
    ragTools["search / grep / read + mock ticket"]
    idx["Local index: Chroma + BM25"]
    policy["Evidence + RAG budget"]

    user --> agentLoop
    agentLoop <--> llm
    agentLoop --> ragTools --> idx
    ragTools --> policy --> agentLoop
```

## 核心能力

- **混合 RAG（简化版）**：Markdown 语料分块；向量检索与 BM25 RRF 融合；`index` / `search` / `read` 可脱离 LLM 调试。
- **证据与预算**：RAG 工具结果记入库存，`evidence_status` 为 `insufficient | partial | sufficient`；`RAG_BUDGET` 耗尽后不再向模型暴露 RAG 工具；无证据时拒答模板收口。
- **工具**：`search_notes`、`grep_notes`、`read_note` 与 wiki-agent 对齐；`query_incident_ticket` 为 Mock 工单（非 RAG，展示 Tool Calling 广度）。
- **评测**：`fixtures/e2e/questions.json` 为母项目 12 题子集；`wiki-graph eval` 跑简化 TaskContract（需 LLM）。
- **Checkpoint**：内存 checkpointer + `repl` 多轮会话。

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
wiki-graph index
wiki-graph --check-llm
wiki-graph ask "HTTP 缓存里 max-age=300 表示什么？"
```

### 常用 CLI

| 命令 | 说明 |
|------|------|
| `index` | 从语料目录构建 `.wiki_graph_index` |
| `search <query>` | 仅检索，不调用 LLM |
| `read <path>` | 按相对路径读笔记片段 |
| `--check-llm` | 探测 Chat API |
| `ask <question>` | LangGraph Agent 单次问答 |
| `ask <question> --stream` | 流式打印回答增量 |
| `repl` | 多轮 REPL（共享 checkpoint） |
| `eval` | E2E 子集（简化 contract，需 API） |

安装 `pip install -e ".[dev]"` 后使用 `wiki-graph`，或 `python -m cli`（需在仓库根目录且 `PYTHONPATH` 含 `src`，一般仅开发调试）。

## 开发与验证

```bash
# 无 API 的索引与离线测试
export WIKI_GRAPH_EMBEDDING=fake   # Windows: $env:WIKI_GRAPH_EMBEDDING="fake"
wiki-graph index
pytest -q

# 有 API 时跑 E2E 子集（LLM 波动可能导致偶发失败）
wiki-graph eval
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

`fixtures/corpus/` 与 `fixtures/e2e/questions.json` 来自 wiki-agent 公开快照（语料与 e2e 题 `id` 保留，便于交叉验证）。索引产物写入 `.wiki_graph_index/`（本地生成，不入库）。

## 项目边界

- 本地 CLI，无 Web UI、无多租户部署面
- 不复制 wiki-agent 的 BGE / LanceDB 离线索引与完整 TaskContract / Trace 管线
- 私有语料、API 密钥与索引目录不进入 Git
- LLM 与 Embedding 行为受供应商与模型版本影响

## License

[MIT](LICENSE)