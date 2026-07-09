"""编译后的 LangGraph 图 + `invoke(session_id, user_input)` 主接缝。

切片 3 拓扑（知识流接入数据库层）：

    START → rewrite → react_core ─(无工具)─────────────────────────────────→ chat_flow_answer → finalize → END
                          └─(plan_coverage)→ build_coverage → retrieve_knowledge → retrieve_database → final_answer → finalize → END

图对 `Models`、知识库检索器 `KnowledgeRetriever`、数据库工具 `DatabaseTool` 依赖注入，可用桩
确定性驱动全图；持久化用内存版 MemorySaver（Redis 版见切片 7）。联网层沿同一覆盖度表在后续切片接入。
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings
from .database_tool import DatabaseTool, build_default_database_tool
from .knowledge_tool import FakeKnowledgeRetriever, KnowledgeRetriever
from .models import Models, build_models
from .nodes import (
    make_build_coverage_node,
    make_chat_flow_answer_node,
    make_final_answer_node,
    make_finalize_node,
    make_react_core_node,
    make_retrieve_database_node,
    make_retrieve_knowledge_node,
    make_rewrite_node,
    route_after_core,
)
from .observability import build_langfuse_callbacks
from .state import TurnState


def build_graph(
    models: Models | None = None,
    *,
    knowledge_retriever: KnowledgeRetriever | None = None,
    database_tool: DatabaseTool | None = None,
    checkpointer: Any | None = None,
):
    """组装并编译对话图。

    models：两档模型（省略则从 .env 构造真实 ChatOpenAI）。
    knowledge_retriever：知识库检索适配层（省略则用假实现打桩，检索端点到位后替换实现）。
    database_tool：参数化查询能力集（省略则用内置能力 + 假后端打桩，真实只读库到位后替换后端）。
    checkpointer：跨轮持久化（省略则用内存版 MemorySaver）。
    """
    models = models or build_models()
    knowledge_retriever = knowledge_retriever or FakeKnowledgeRetriever()
    database_tool = database_tool or build_default_database_tool()
    checkpointer = checkpointer or MemorySaver()

    builder = StateGraph(TurnState)
    builder.add_node("rewrite", make_rewrite_node())
    builder.add_node("react_core", make_react_core_node(models))
    builder.add_node("build_coverage", make_build_coverage_node())
    builder.add_node("retrieve_knowledge", make_retrieve_knowledge_node(knowledge_retriever))
    builder.add_node("retrieve_database", make_retrieve_database_node(database_tool, models))
    builder.add_node("final_answer", make_final_answer_node(models))
    builder.add_node("chat_flow_answer", make_chat_flow_answer_node(models))
    builder.add_node("finalize", make_finalize_node())

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "react_core")
    # 内核自主判定：调 plan_coverage → 知识流；无工具 → 对话流。不设入口硬分类器。
    builder.add_conditional_edges(
        "react_core",
        route_after_core,
        {
            "build_coverage": "build_coverage",
            "chat_flow_answer": "chat_flow_answer",
        },
    )
    # 知识流渐进检索：产覆盖度表 → 知识库 → 数据库 → 强制作答。命中即停由各层内部按覆盖度表判定。
    builder.add_edge("build_coverage", "retrieve_knowledge")
    builder.add_edge("retrieve_knowledge", "retrieve_database")
    builder.add_edge("retrieve_database", "final_answer")
    builder.add_edge("final_answer", "finalize")
    # 对话流出口。
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
