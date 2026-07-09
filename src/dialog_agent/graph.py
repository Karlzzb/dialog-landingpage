"""编译后的 LangGraph 图 + `invoke(session_id, user_input)` 主接缝。

切片 1 拓扑（最薄行走骨架）：

    START → rewrite → react_core ─(无工具)→ chat_flow_answer → finalize → END
                            └────(工具调用)→ [切片 2 接入检索工具循环]

图对 `Models` 依赖注入，`build_graph(models=...)` 可用桩确定性驱动；持久化用内存版
MemorySaver（Redis 版见切片 7）。
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings
from .models import Models, build_models
from .nodes import (
    make_chat_flow_answer_node,
    make_finalize_node,
    make_react_core_node,
    make_rewrite_node,
    route_after_core,
)
from .observability import build_langfuse_callbacks
from .state import TurnState


def build_graph(
    models: Models | None = None,
    *,
    tools: list | None = None,
    checkpointer: Any | None = None,
):
    """组装并编译对话图。

    models：两档模型（省略则从 .env 构造真实 ChatOpenAI）。
    tools：内核可用的检索工具（切片 1 为空，seam 预留）。
    checkpointer：跨轮持久化（省略则用内存版 MemorySaver）。
    """
    models = models or build_models()
    tools = tools or []
    checkpointer = checkpointer or MemorySaver()

    builder = StateGraph(TurnState)
    builder.add_node("rewrite", make_rewrite_node())
    builder.add_node("react_core", make_react_core_node(models, tools))
    builder.add_node("chat_flow_answer", make_chat_flow_answer_node(models))
    builder.add_node("finalize", make_finalize_node())

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "react_core")
    # 内核自主判定：无工具调用 → 对话流作答。工具分支（"tools"）在切片 2 引入检索工具时接入。
    builder.add_conditional_edges(
        "react_core",
        route_after_core,
        {"chat_flow_answer": "chat_flow_answer"},
    )
    builder.add_edge("chat_flow_answer", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


# 生产环境的进程内单例图（懒构造，真实模型 + 内存 checkpointer）。
_default_graph = None


def _get_default_graph():
    global _default_graph
    if _default_graph is None:
        _default_graph = build_graph()
    return _default_graph


def invoke(
    session_id: str,
    user_input: str,
    *,
    graph: Any | None = None,
    settings: Settings | None = None,
) -> tuple[str, dict[str, Any]]:
    """主接缝：驱动一轮对话，返回 (最终回复, 完整 State)。

    session_id 即 LangGraph thread_id（会话隔离仅靠其唯一性，ADR 0005）。
    graph/settings 可注入用于测试；省略则用生产单例图并按 .env 挂 Langfuse tracing。
    """
    settings = settings or get_settings()
    graph = graph or _get_default_graph()
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id},
        "callbacks": build_langfuse_callbacks(settings),
    }
    result = graph.invoke({"user_input": user_input}, config=config)
    return result.get("final_reply", ""), result
