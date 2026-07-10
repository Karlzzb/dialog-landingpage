"""编译后的 LangGraph 图 + `invoke(session_id, user_input)` 主接缝。

切片 5 拓扑（渐进检索命中即停 + 随观察精修 + 对比分析）：

    START → rewrite → react_core ─(无工具)────────────────────────────────────────────→ chat_flow_answer → finalize → END
                          └─(plan_coverage)→ build_coverage → retrieve_knowledge → refine_after_knowledge
                              → retrieve_database → refine_after_database → retrieve_external → final_answer → finalize → END

精修步（`refine_after_knowledge` / `refine_after_database`）随观察补充新单元 / 修正数据源匹配
（ADR 0002 动态特性），补出的单元由后续层自然拾取；命中即停（`remaining_units==0`）时精修步
早退、不调 LLM。图对 `Models`、知识库检索器 `KnowledgeRetriever`、数据库工具 `DatabaseTool`、
联网工具 `WebSearchTool`、`SafetyCaps` 依赖注入，可用桩确定性驱动全图。

切片 7：默认 checkpointer 由内存版 `MemorySaver` 切为 Redis 版 `RedisSaver`——`.env` 配了 Redis
host 即用 Redis 持久化整个 State（`thread_id = session_id`、会话记忆带可配置 TTL、checkpointer db
与业务隔离）；本地无 Redis / 连接失败时优雅降级回 `MemorySaver`，不阻塞主体与测试。

切片 8：loop 终止的另两条退出路径 + 全局兜底接入此管线——
- 安全阀：检索层/精修步后经 `make_safety_route` 条件边判定，`tool_call_count`/`iteration_count`
  触顶即短路到 `final_answer` 兜底强制作答（未取到的单元落盲区）。阈值 `SafetyCaps` 经 .env 配置。
- 盲区：`final_answer` 入口把仍 REMAINING 的单元标记为 BLIND_SPOT，结论坦诚告知边界 + 给指引。
- 优雅兜底：`invoke` 外层 try/except 捕获任意异常 → 固定人性化话术（`safety.FALLBACK_REPLY`），
  技术堆栈绝不外泄（异常细节仅落运维日志）。
"""

from __future__ import annotations

import logging
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
    make_refine_coverage_node,
    make_retrieve_database_node,
    make_retrieve_external_node,
    make_retrieve_knowledge_node,
    make_rewrite_node,
    make_safety_route,
    route_after_core,
)
from .observability import build_langfuse_callbacks
from .persistence import build_redis_checkpointer
from .safety import FALLBACK_REPLY, SafetyCaps
from .state import TurnState
from .web_search_tool import WebSearchTool, build_default_web_search_tool

logger = logging.getLogger(__name__)


def _build_default_checkpointer(settings: Settings) -> Any:
    """默认 checkpointer：配了 Redis host 即用 Redis 版；否则 / 连接失败降级内存版。

    生产环境应有可达的 Redis Stack（RedisSaver 需 RedisJSON + RediSearch）。Redis 不可达时降级
    回 `MemorySaver` 仅作本地开发/测试兜底，不阻塞主体；该路径不进生产（生产会因 Redis 不可用
    在 invoke 时失败，不应静默降级丢持久化——但此处降级保的是「图能编出来」，invoke 的可达性
    由部署环境保证）。
    """
    if not settings.has_redis_config:
        return MemorySaver()
    try:
        return build_redis_checkpointer(settings)
    except Exception as exc:  # noqa: BLE001 —— 降级兜底，不阻断图编译。
        logger.warning(
            "Redis checkpointer 初始化失败，降级为内存版（会话不跨进程持久化）：%s", exc
        )
        return MemorySaver()


def build_graph(
    models: Models | None = None,
    *,
    knowledge_retriever: KnowledgeRetriever | None = None,
    database_tool: DatabaseTool | None = None,
    web_search_tool: WebSearchTool | None = None,
    checkpointer: Any | None = None,
    settings: Settings | None = None,
    safety_caps: SafetyCaps | None = None,
):
    """组装并编译对话图。

    models：两档模型（省略则从 .env 构造真实 ChatOpenAI）。
    knowledge_retriever：知识库检索适配层（省略则用假实现打桩，检索端点到位后替换实现）。
    database_tool：参数化查询能力集（省略则用内置能力 + 假后端打桩，真实只读库到位后替换后端）。
    web_search_tool：联网工具（省略则用假后端 + 默认空词表过滤器打桩，真实检索后端/合规词表
        到位后注入）。
    checkpointer：跨轮持久化（省略则按 settings 用 RedisSaver，配了 Redis host 即 Redis 持久化，
        否则/连接失败降级 MemorySaver；切片 7）。
    settings：配置（省略则用进程单例，用于决定默认 checkpointer 与 Langfuse tracing）。
    safety_caps：loop 安全阀阈值（省略则按 settings 构造，`MAX_TOOL_CALLS`/`MAX_ITERATIONS` 可配；
        切片 8）。
    """
    settings = settings or get_settings()
    models = models or build_models()
    knowledge_retriever = knowledge_retriever or FakeKnowledgeRetriever()
    database_tool = database_tool or build_default_database_tool()
    web_search_tool = web_search_tool or build_default_web_search_tool()
    checkpointer = checkpointer or _build_default_checkpointer(settings)
    safety_caps = safety_caps or SafetyCaps.from_settings(settings)

    builder = StateGraph(TurnState)
    builder.add_node("rewrite", make_rewrite_node(models))
    builder.add_node("react_core", make_react_core_node(models))
    builder.add_node("build_coverage", make_build_coverage_node())
    builder.add_node(
        "retrieve_knowledge", make_retrieve_knowledge_node(knowledge_retriever, safety_caps)
    )
    builder.add_node(
        "refine_after_knowledge", make_refine_coverage_node(models, safety_caps)
    )
    builder.add_node(
        "retrieve_database",
        make_retrieve_database_node(database_tool, models, safety_caps),
    )
    builder.add_node(
        "refine_after_database", make_refine_coverage_node(models, safety_caps)
    )
    builder.add_node(
        "retrieve_external", make_retrieve_external_node(web_search_tool, safety_caps)
    )
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
    # 知识流渐进检索：产覆盖度表 → 知识库 → 精修 → 数据库 → 精修 → 联网 → 强制作答。
    # 命中即停由各层与精修步内部按覆盖度表判定（无待覆盖单元即早退，不向下穿透）。
    # 安全阀（切片 8）：每层/精修后经 make_safety_route 判定，触顶即短路到 final_answer 兜底
    # 强制作答；retrieve_external 为最末层，其后即 final_answer，故无需单独条件边。
    builder.add_edge("build_coverage", "retrieve_knowledge")
    builder.add_conditional_edges(
        "retrieve_knowledge",
        make_safety_route(safety_caps, "refine_after_knowledge"),
        {"refine_after_knowledge": "refine_after_knowledge", "final_answer": "final_answer"},
    )
    builder.add_conditional_edges(
        "refine_after_knowledge",
        make_safety_route(safety_caps, "retrieve_database"),
        {"retrieve_database": "retrieve_database", "final_answer": "final_answer"},
    )
    builder.add_conditional_edges(
        "retrieve_database",
        make_safety_route(safety_caps, "refine_after_database"),
        {"refine_after_database": "refine_after_database", "final_answer": "final_answer"},
    )
    builder.add_conditional_edges(
        "refine_after_database",
        make_safety_route(safety_caps, "retrieve_external"),
        {"retrieve_external": "retrieve_external", "final_answer": "final_answer"},
    )
    builder.add_edge("retrieve_external", "final_answer")
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

    优雅兜底（切片 8）：任意异常被捕获 → 返回固定人性化话术（`FALLBACK_REPLY`），技术堆栈绝不
    外泄（异常细节经 `logger.exception` 落运维日志，不进回复）。用户侧只见人性化兜底，不感知
    是哪个节点 / 哪个工具 / 什么报错。节点内的局部降级（精修解析异常、数据库非法入参等）仍在
    各自 try/except 内吞掉、不阻断整轮，不会冒泡到此处。
    """
    settings = settings or get_settings()
    graph = graph or _get_default_graph()
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id},
        "callbacks": build_langfuse_callbacks(settings),
    }
    try:
        result = graph.invoke({"user_input": user_input}, config=config)
    except Exception:
        # 任意异常 → 人性化兜底，技术堆栈绝不外泄。异常细节落运维日志供排查。
        logger.exception("对话执行异常，已优雅降级为兜底话术（session_id=%s）", session_id)
        return FALLBACK_REPLY, {"final_reply": FALLBACK_REPLY}
    return result.get("final_reply", ""), result
