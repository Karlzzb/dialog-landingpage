"""切片 3 —— 数据库层的图入口行为测试。

主接缝：编译后的图入口 `invoke(session_id, user_input)`，注入桩化数据库工具（假后端）+ 桩化
LLM，断言：数据库单元被覆盖、数字溯源正确（结论数字取自 raw）、命中即停（知识库已覆盖则不
查数据库）、注入型入参降级入盲区。只测外部行为，不测内部函数如何被调用。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from dialog_agent.coverage import CoverageStatus, CoverageTable
from dialog_agent.database_tool import FakeDatabaseBackend, build_default_database_tool
from dialog_agent.evidence import SourceType
from dialog_agent.graph import build_graph, invoke
from dialog_agent.knowledge_tool import FakeKnowledgeRetriever
from conftest import make_stub_models, plan_coverage_call, query_capability_call, refine_noop_call, rewrite_call

# 数据库假数据：某学院某专业对口率一行（数字供 raw 溯源）。
DB_ROWS = {
    "internship_placement_rate": [
        {
            "college": "机电学院",
            "major": "数控技术",
            "rate": 87.5,
            "sample_size": 120,
            "term": "2025-秋",
        }
    ]
}


def _db_tool(backend_rows=DB_ROWS):
    return build_default_database_tool(FakeDatabaseBackend(backend_rows))


def test_database_unit_covered_with_number_traceability(test_settings):
    """数据库单元经参数化能力覆盖 → 覆盖度表标记已覆盖、Evidence 为 INTERNAL_DATABASE、数字取自 raw。"""
    final_reply = "机电学院数控技术专业的实习对口率为 87.5%（据校内数据库，2025 秋季学期，样本 120 人）。"
    models = make_stub_models(
        strong_responses=[
            # 内核首步：拆出一个数据库单元。
            plan_coverage_call(
                [{"id": "u1", "need": "机电学院数控技术实习对口率", "sources": ["INTERNAL_DATABASE"]}]
            ),
            # 知识库层无可覆盖单元（u1 不含知识库源），仍缺失 → 精修步触发，无需精修。
            refine_noop_call(),
            # 数据库层：选能力 + 填参。
            query_capability_call(
                "internship_placement_rate", {"college": "机电学院", "major": "数控技术"}
            ),
            # 结论生成。
            AIMessage(content=final_reply),
        ],
        fast_responses=[rewrite_call("机电学院数控技术专业实习对口率是多少？")],
    )
    # 知识库对该查询无命中（留给数据库层覆盖）。
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever({}),
        database_tool=_db_tool(),
        checkpointer=MemorySaver(),
    )

    reply, state = invoke(
        "db-1", "机电学院数控技术专业实习对口率是多少？", graph=graph, settings=test_settings
    )

    # 覆盖度表：单元被数据库层覆盖、无剩余。
    table = CoverageTable.from_dict(state["coverage"])
    assert table.units[0].status == CoverageStatus.COVERED
    assert table.remaining_units == []

    # 分源 Evidence：INTERNAL_DATABASE，raw 保留原始行供数字溯源。
    assert state["evidence"]
    ev = state["evidence"][0]
    assert ev["source_type"] == SourceType.INTERNAL_DATABASE.value
    assert ev["raw"]["rate"] == 87.5
    assert ev["raw"]["sample_size"] == 120

    # 结论为强制作答产物。
    assert reply == final_reply


def test_final_answer_prompt_carries_db_raw_fields(test_settings):
    """结论生成步拿到数据库证据的原始字段（数字溯源由素材结构导出，非事后质检）。"""
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [{"id": "u1", "need": "机电学院数控技术实习对口率", "sources": ["INTERNAL_DATABASE"]}]
            ),
            refine_noop_call(),
            query_capability_call(
                "internship_placement_rate", {"college": "机电学院", "major": "数控技术"}
            ),
            AIMessage(content="结论。"),
        ],
        fast_responses=[rewrite_call("机电学院数控技术对口率")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever({}),
        database_tool=_db_tool(),
        checkpointer=MemorySaver(),
    )

    invoke("db-2", "机电学院数控技术对口率", graph=graph, settings=test_settings)

    # 强模型末次调用（结论生成）的输入含原始数字字段，供忠于原始数据作答。
    final_human_text = models.strong.invocations[-1][-1].content
    assert "87.5" in final_human_text
    assert "internship_stats" in final_human_text


def test_hit_and_stop_knowledge_covers_skips_database(test_settings):
    """命中即停：知识库已覆盖全部单元 → 数据库层不触发任何后端查询（不向下穿透）。"""
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
    backend = FakeDatabaseBackend(DB_ROWS)
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                # 单元候选源同时含知识库与数据库，知识库优先。
                [
                    {
                        "id": "u1",
                        "need": "机电学院数控技术对口率",
                        "sources": ["INTERNAL_KNOWLEDGE", "INTERNAL_DATABASE"],
                    }
                ]
            ),
            AIMessage(content="据校内统计库，对口率 87.5%。"),
        ],
        fast_responses=[rewrite_call("机电学院数控技术对口率")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever(corpus),
        database_tool=build_default_database_tool(backend),
        checkpointer=MemorySaver(),
    )

    _, state = invoke("db-3", "机电学院数控技术对口率", graph=graph, settings=test_settings)

    # 知识库已覆盖 → 数据库后端零调用（命中即停）。
    assert backend.calls == []
    # 强模型只被调用两次（规划 + 结论），数据库选能力步未发生。
    assert len(models.strong.invocations) == 2
    ev_sources = {ev["source_type"] for ev in state["evidence"]}
    assert ev_sources == {SourceType.INTERNAL_KNOWLEDGE.value}


def test_injection_param_degrades_to_blind_spot(test_settings):
    """局部降级：内核填入注入型参数 → 工具拒绝 → 该单元落盲区，结论仍生成（不外泄技术细节）。"""
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [{"id": "u1", "need": "某学院对口率", "sources": ["INTERNAL_DATABASE"]}]
            ),
            # 知识库层无可覆盖单元，仍缺失 → 精修步触发，无需精修。
            refine_noop_call(),
            # 注入型 college 参数：被数据库工具强类型/字符校验拒绝。
            query_capability_call(
                "internship_placement_rate",
                {"college": "机电学院'; DROP TABLE internship_stats; --", "major": "数控技术"},
            ),
            # 注入被拒后该单元仍缺失 → 精修步再次触发，仍无需精修。
            refine_noop_call(),
            AIMessage(content="抱歉，暂未查询到该专业的实习对口率数据，建议核对学院/专业名称后再试。"),
        ],
        fast_responses=[rewrite_call("查一下对口率")],
    )
    backend = FakeDatabaseBackend(DB_ROWS)
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever({}),
        database_tool=build_default_database_tool(backend),
        checkpointer=MemorySaver(),
    )

    reply, state = invoke("db-4", "查一下对口率", graph=graph, settings=test_settings)

    # 注入被拒 → 后端从未执行（run_query 未被触达）。
    assert backend.calls == []
    # 该单元穷尽候选源仍未覆盖 → 落盲区（一等状态），无数据库 Evidence。
    table = CoverageTable.from_dict(state["coverage"])
    assert table.units[0].status == CoverageStatus.BLIND_SPOT
    assert not state.get("evidence")
    # 结论仍产出人性化兜底，不外泄技术堆栈。
    assert reply
    assert "DROP" not in reply and "SQL" not in reply


def test_no_matching_capability_degrades(test_settings):
    """内核判定无能力匹配（未发起工具调用）→ 该数据库单元落盲区，不阻断整轮。"""
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [{"id": "u1", "need": "某项无对应能力的统计", "sources": ["INTERNAL_DATABASE"]}]
            ),
            # 知识库层无可覆盖单元，仍缺失 → 精修步触发，无需精修。
            refine_noop_call(),
            AIMessage(content=""),  # 数据库层：无工具调用（无匹配能力）
            # 数据库层无匹配能力 → 该单元仍缺失 → 精修步再次触发，无需精修。
            refine_noop_call(),
            AIMessage(content="暂无法查询该项数据，建议提供更具体的口径。"),
        ],
        fast_responses=[rewrite_call("查个冷门统计")],
    )
    backend = FakeDatabaseBackend(DB_ROWS)
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever({}),
        database_tool=build_default_database_tool(backend),
        checkpointer=MemorySaver(),
    )

    _, state = invoke("db-5", "查个冷门统计", graph=graph, settings=test_settings)

    assert backend.calls == []
    table = CoverageTable.from_dict(state["coverage"])
    assert table.units[0].status == CoverageStatus.BLIND_SPOT
    assert not state.get("evidence")
