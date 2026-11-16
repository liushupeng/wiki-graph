from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from wiki_graph.config import Settings
from wiki_graph.graph.state import AgentState

SYSTEM_PROMPT = """你是 wiki-graph 知识库助手，回答应基于工具检索到的笔记内容。
需要事实时优先调用 search_notes 或 read_note；回答中注明来源路径（如 dev/http-caching.md）。
若检索不到足够信息，明确说明缺口，不要编造。"""


def make_agent_node(llm: ChatOpenAI, tools: list):
    bound = llm.bind_tools(tools)

    def agent(state: AgentState, config: RunnableConfig) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = bound.invoke(messages, config)
        return {"messages": [response]}

    return agent


def make_llm(settings: Settings) -> ChatOpenAI:
    if not settings.llm_configured():
        raise RuntimeError("未配置 BASE_URL / API_KEY，无法创建 Chat 模型。")
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
    )