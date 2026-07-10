"""切片 8 —— loop 终止的另两条退出路径 + 全局兜底的图入口行为测试。

主接缝：编译后的图入口 `invoke(session_id, user_input)`，桩化三工具 + 桩化 LLM，断言：
- 盲区路径：三层穷尽仍缺失的单元落入 BLIND_SPOT（一等状态），结论坦诚告知边界 + 给下一步
  指引 + 不编造；
- 安全阀（工具调用）：阈值可配，触顶即短路到 final_answer 强制作答、未取到的单元落盲区、
  后续层零调用；
- 安全阀（迭代步数）：阈值可配，触顶即短路、跳过后续检索层；
- 阈值可配：经 Settings / .env 注入；
- 优雅兜底：任意异常被捕获 → 人性化固定话术，技术堆栈绝不外泄。
只测外部行为，不测内部函数如何被调用。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from dialog_agent.config import Settings
from dialog_agent.coverage import CoverageStatus, CoverageTable
from dialog_agent.database_tool import FakeDatabaseBackend, build_default_database_tool
from dialog_agent.graph import build_graph, invoke
from dialog_agent.knowledge_tool import FakeKnowledgeRetriever
from dialog_agent.safety import FALLBACK_REPLY, SafetyCaps
from dialog_agent.web_search_tool import (
    ContentFilter,
    FakeWebSearchBackend,
    WebSearchTool,
)
from conftest import make_stub_models, plan_coverage_call, refine_noop_call, rewrite_call


def _empty_tools():
    """三层全空的桩工具：任何查询都查不到（模拟三层穷尽仍缺失 → 盲区）。"""
    return (
        FakeKnowledgeRetriever({}),
        build_default_database_tool(FakeDatabaseBackend({})),
        WebSearchTool(backend=FakeWebSearchBackend({}), content_filter=ContentFilter()),
    )


# ── 盲区路径 ──


def test_blind_spot_path_all_sources_miss(test_settings):
    """三层穷尽仍缺失 → 单元落 BLIND_SPOT（一等状态）；结论 prompt 含盲区+指引指令、回复坦诚不编造。"""
    blind_reply = (
        "三项信息当前均未能查到：校企合作补贴标准、机电学院对口率、某科技营收。"
        "建议核实实体名称或换关键词后重试；校内数据可联系相关业务部门核实。"
    )
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [
                    {"id": "u1", "need": "校企合作补贴标准", "sources": ["INTERNAL_KNOWLEDGE"]},
                    {"id": "u2", "need": "机电学院对口率", "sources": ["INTERNAL_DATABASE"]},
                    {"id": "u3", "need": "某科技营收", "sources": ["EXTERNAL_SEARCH"]},
                ]
            ),
            # 知识库未命中、仍有缺失 → 精修步触发，无需精修。
            refine_noop_call(),
            # 数据库层：无能力匹配（无工具调用），u2 未覆盖。
            AIMessage(content=""),
            # 数据库后仍有缺失 → 精修步再次触发，无需精修。
            refine_noop_call(),
            AIMessage(content=blind_reply),
        ],
        fast_responses=[rewrite_call("补贴标准和对口率和某科技营收")],
    )
    kb, db, web = _empty_tools()
    graph = build_graph(
        models,
        knowledge_retriever=kb,
        database_tool=db,
        web_search_tool=web,
        checkpointer=MemorySaver(),
    )

    reply, state = invoke(
        "bs-1", "补贴标准和对口率和某科技营收", graph=graph, settings=test_settings
    )

    table = CoverageTable.from_dict(state["coverage"])
    # 三单元穷尽候选源后均落盲区（一等状态，非异常）。
    assert {u.id for u in table.blind_spot_units} == {"u1", "u2", "u3"}
    assert all(u.status == CoverageStatus.BLIND_SPOT for u in table.units)
    assert table.remaining_units == []
    # 无任何证据累积。
    assert not state.get("evidence")
    # 喂给结论生成步的 prompt 含盲区标注 + 指引指令（由素材结构 + prompt 导出，非事后质检）。
    final_human_text = models.strong.invocations[-1][-1].content
    assert "盲区" in final_human_text
    assert "下一步指引" in final_human_text
    # 回复坦诚告知边界 + 给指引 + 不编造（无臆造数字）。
    assert reply == blind_reply
    assert "未能查到" in reply
    assert "建议" in reply


# ── 安全阀：工具调用触顶 ──


def test_safety_valve_tool_calls_force_answer(test_settings):
    """max_tool_calls 可配；触顶即在 per-unit 循环内 break、条件边短路到 final_answer 强制作答。"""
    # 三单元均走知识库，语料对每个单元都命中（同一条 chunk）。
    corpus = {
        "对口率": [
            {
                "content": "机电学院数控技术实习对口率 87.5%。",
                "file_name": "实习统计.pdf",
                "knowledge_name": "校内统计库",
                "chunk_index": 2,
            }
        ]
    }
    kb = FakeKnowledgeRetriever(corpus)
    db_backend = FakeDatabaseBackend({})
    web_backend = FakeWebSearchBackend({})
    web_tool = WebSearchTool(backend=web_backend, content_filter=ContentFilter())
    forced_reply = "已查到部分对口率数据；其余因检索上限暂未取到，建议缩小范围后重试。"
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [
                    {"id": "u1", "need": "数控技术对口率", "sources": ["INTERNAL_KNOWLEDGE"]},
                    {"id": "u2", "need": "机电技术对口率", "sources": ["INTERNAL_KNOWLEDGE"]},
                    {"id": "u3", "need": "智能制造对口率", "sources": ["INTERNAL_KNOWLEDGE"]},
                ]
            ),
            # 安全阀在 retrieve_knowledge 内触顶 → 条件边短路到 final_answer，跳过精修/DB/精修/联网。
            AIMessage(content=forced_reply),
        ],
        fast_responses=[rewrite_call("三个专业对口率")],
    )
    low_caps_settings = Settings(_env_file=None, max_tool_calls=2)
    graph = build_graph(
        models,
        knowledge_retriever=kb,
        database_tool=build_default_database_tool(db_backend),
        web_search_tool=web_tool,
        checkpointer=MemorySaver(),
        settings=low_caps_settings,
    )

    reply, state = invoke(
        "sv-1", "三个专业对口率", graph=graph, settings=low_caps_settings
    )

    # 工具调用触顶于 2：第 3 个单元未被检索（in-loop break）。
    assert state["tool_call_count"] == 2
    assert len(kb.calls) == 2
    # 条件边短路：数据库 / 联网后端零调用。
    assert db_backend.calls == []
    assert web_backend.calls == []
    # u1/u2 已覆盖，u3 未取到 → 落盲区（强制作答时由 final_answer 标记）。
    table = CoverageTable.from_dict(state["coverage"])
    by_id = {u.id: u for u in table.units}
    assert by_id["u1"].status == CoverageStatus.COVERED
    assert by_id["u2"].status == CoverageStatus.COVERED
    assert by_id["u3"].status == CoverageStatus.BLIND_SPOT
    # 强制作答产出回复。
    assert reply == forced_reply


# ── 安全阀：迭代步数触顶 ──


def test_safety_valve_iterations_force_answer(test_settings):
    """max_iterations 可配；触顶即条件边短路、跳过后续检索层（DB 层从未被触达）。"""
    db_backend = FakeDatabaseBackend({})
    web_backend = FakeWebSearchBackend({})
    forced_reply = "检索步数已达上限，暂未能查到该项校内数据，建议换关键词或联系业务部门。"
    models = make_stub_models(
        strong_responses=[
            # 单元走数据库，但 max_iterations=2 会在 refine 后触顶、跳过数据库层。
            plan_coverage_call(
                [{"id": "u1", "need": "机电学院对口率", "sources": ["INTERNAL_DATABASE"]}]
            ),
            # 知识库层无可覆盖单元、u1 仍缺失 → 精修步触发（计一步迭代后触顶，无需精修即早退）。
            # refine 在 valve 触顶时早退、不消耗模型响应，故此处不需再给 refine 响应。
            AIMessage(content=forced_reply),
        ],
        fast_responses=[rewrite_call("机电学院对口率")],
    )
    low_iter_settings = Settings(_env_file=None, max_iterations=2)
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever({}),
        database_tool=build_default_database_tool(db_backend),
        web_search_tool=WebSearchTool(backend=web_backend, content_filter=ContentFilter()),
        checkpointer=MemorySaver(),
        settings=low_iter_settings,
    )

    reply, state = invoke(
        "sv-2", "机电学院对口率", graph=graph, settings=low_iter_settings
    )

    # 迭代触顶于 2：retrieve_knowledge(1) → refine_after_knowledge(2) 触顶 → final_answer。
    assert state["iteration_count"] == 2
    # 数据库层被短路、从未触达（默认阈值下本应经 query_capability 查询）。
    assert db_backend.calls == []
    assert web_backend.calls == []
    # u1 未取到 → 落盲区。
    table = CoverageTable.from_dict(state["coverage"])
    assert table.units[0].status == CoverageStatus.BLIND_SPOT
    assert reply == forced_reply


# ── 阈值可配 ──


def test_safety_caps_configurable_via_settings():
    """SafetyCaps 从 Settings 读取 MAX_TOOL_CALLS / MAX_ITERATIONS（.env 可配）。"""
    settings = Settings(_env_file=None, max_tool_calls=3, max_iterations=5)
    caps = SafetyCaps.from_settings(settings)
    assert caps.max_tool_calls == 3
    assert caps.max_iterations == 5
    # 默认值（PRD/CONTEXT.md 初值 6 / 8）。
    defaults = SafetyCaps.from_settings(Settings(_env_file=None))
    assert defaults.max_tool_calls == 6
    assert defaults.max_iterations == 8


def test_safety_caps_from_env(monkeypatch):
    """经环境变量注入阈值（.env 即环境变量，验证可配置性）。"""
    monkeypatch.setenv("MAX_TOOL_CALLS", "4")
    monkeypatch.setenv("MAX_ITERATIONS", "9")
    settings = Settings(_env_file=None)
    assert settings.max_tool_calls == 4
    assert settings.max_iterations == 9
    caps = SafetyCaps.from_settings(settings)
    assert caps.max_tool_calls == 4
    assert caps.max_iterations == 9


# ── 优雅兜底：任意异常 → 人性化话术，技术堆栈绝不外泄 ──


class _RaisingRetriever:
    """检索器抛异常（模拟任意运行时异常）：异常信息含可辨识的「技术细节」串。"""

    def __init__(self, message: str) -> None:
        self._message = message

    def retrieve(self, query, *, knowledge_id=None, top_k=3):  # noqa: D401
        raise RuntimeError(self._message)


def test_global_exception_fallback(test_settings):
    """任意异常被捕获 → 固定人性化话术；回复绝不出现技术堆栈/报错细节。"""
    secret = "kaboom-secret-stacktrace-detail-XYZ"
    retriever = _RaisingRetriever(secret)
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [{"id": "u1", "need": "校企合作补贴", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
        ],
        fast_responses=[rewrite_call("校企合作补贴")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=retriever,
        database_tool=build_default_database_tool(FakeDatabaseBackend({})),
        web_search_tool=WebSearchTool(
            backend=FakeWebSearchBackend({}), content_filter=ContentFilter()
        ),
        checkpointer=MemorySaver(),
    )

    # 不抛异常：invoke 内部 try/except 兜底，返回人性化话术。
    reply, state = invoke(
        "fb-1", "校企合作补贴", graph=graph, settings=test_settings
    )

    assert reply == FALLBACK_REPLY
    assert state["final_reply"] == FALLBACK_REPLY
    # 技术堆栈 / 报错细节 / 异常信息串绝不外泄进回复。
    assert secret not in reply
    assert "Traceback" not in reply
    assert "RuntimeError" not in reply
    assert "Error" not in reply
