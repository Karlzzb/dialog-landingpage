"""切片 4 —— 联网层的图入口行为测试。

主接缝：编译后的图入口 `invoke(session_id, user_input)`，注入桩化联网工具（假后端）+ 桩化
LLM，断言：外部单元被覆盖、Evidence 为 EXTERNAL_SEARCH、数字取自 raw、敏感词在边界被清洗且
**不透传进 final_answer 的 LLM 输入**、外部数据带「据互联网公开信息」弱化标注且与内部数据
分源隔离、命中即停（更高层已覆盖则不查联网）。只测外部行为，不测内部函数如何被调用。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from dialog_agent.coverage import CoverageStatus, CoverageTable
from dialog_agent.evidence import SourceType
from dialog_agent.graph import build_graph, invoke
from dialog_agent.knowledge_tool import FakeKnowledgeRetriever
from dialog_agent.web_search_tool import (
    ContentFilter,
    FakeWebSearchBackend,
    WebSearchTool,
)
from conftest import make_stub_models, plan_coverage_call, query_capability_call, refine_noop_call, rewrite_call

# 联网假数据：某企业工商信息一条（数字供 raw 溯源），命中关键子串「某科技」。
WEB_CORPUS = {
    "某科技": [
        {
            "content": "某科技有限公司 2024 年营业收入 1.2 亿元，参保 80 人。",
            "url": "https://example.com/firm/123",
            "revenue_yi": 1.2,
            "headcount": 80,
        }
    ]
}


def _web_tool(corpus=WEB_CORPUS, *, mask_terms=(), block_terms=()) -> WebSearchTool:
    return WebSearchTool(
        backend=FakeWebSearchBackend(corpus),
        content_filter=ContentFilter(mask_terms=mask_terms, block_terms=block_terms),
    )


def _kb_tool() -> FakeKnowledgeRetriever:
    return FakeKnowledgeRetriever({})


def _no_db():
    from dialog_agent.database_tool import build_default_database_tool, FakeDatabaseBackend

    return build_default_database_tool(FakeDatabaseBackend({}))


def test_external_unit_covered_with_external_search_evidence(test_settings):
    """外部单元经联网覆盖 → Evidence 为 EXTERNAL_SEARCH、数字取自 raw、覆盖度表归零。"""
    final_reply = "据互联网公开信息，某科技有限公司 2024 年营业收入约 1.2 亿元，参保 80 人。"
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [
                    {
                        "id": "u1",
                        "need": "某科技有限公司 2024 营业收入",
                        "sources": ["EXTERNAL_SEARCH"],
                    }
                ]
            ),
            # 知识库/数据库层均无可覆盖单元（u1 仅联网），仍缺失 → 两次精修步触发，无需精修。
            refine_noop_call(),
            refine_noop_call(),
            AIMessage(content=final_reply),
        ],
        fast_responses=[rewrite_call("某科技有限公司去年营收多少？")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=_kb_tool(),
        database_tool=_no_db(),
        web_search_tool=_web_tool(),
        checkpointer=MemorySaver(),
    )

    reply, state = invoke(
        "ext-1", "某科技有限公司去年营收多少？", graph=graph, settings=test_settings
    )

    table = CoverageTable.from_dict(state["coverage"])
    assert table.units[0].status == CoverageStatus.COVERED
    assert table.remaining_units == []

    assert state["evidence"]
    ev = state["evidence"][0]
    assert ev["source_type"] == SourceType.EXTERNAL_SEARCH.value
    # 数字取自 raw 字段（外部数字溯源）。
    assert ev["raw"]["revenue_yi"] == 1.2
    assert ev["raw"]["headcount"] == 80
    assert reply == final_reply


def test_external_raw_fields_fed_to_final_answer(test_settings):
    """结论生成步拿到外部证据的清洗后原始字段（数字溯源由素材结构导出，非事后质检）。"""
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [{"id": "u1", "need": "某科技营收", "sources": ["EXTERNAL_SEARCH"]}]
            ),
            refine_noop_call(),
            refine_noop_call(),
            AIMessage(content="结论。"),
        ],
        fast_responses=[rewrite_call("某科技营收")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=_kb_tool(),
        database_tool=_no_db(),
        web_search_tool=_web_tool(),
        checkpointer=MemorySaver(),
    )

    invoke("ext-2", "某科技营收", graph=graph, settings=test_settings)

    # 强模型末次调用（结论生成）的输入含外部原始数字字段 + 弱化标注提示。
    final_human_text = models.strong.invocations[-1][-1].content
    assert "1.2" in final_human_text
    assert "据互联网公开信息" in final_human_text


def test_sensitive_external_content_never_reaches_llm(test_settings):
    """边界过滤生效：含 mask 词的外部原文 → State 中 Evidence 已脱敏，且不透传进 final_answer 的 LLM 输入。"""
    sensitive_corpus = {
        "某科技": [
            {
                "content": "某敏感违禁表述：某科技有限公司 2024 营收 1.2 亿元，参保 80 人。",
                "url": "https://example.com/firm/123",
                "revenue_yi": 1.2,
                "headcount": 80,
            }
        ]
    }
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [{"id": "u1", "need": "某科技营收", "sources": ["EXTERNAL_SEARCH"]}]
            ),
            refine_noop_call(),
            refine_noop_call(),
            AIMessage(content="据互联网公开信息，营收约 1.2 亿元。"),
        ],
        fast_responses=[rewrite_call("某科技营收")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=_kb_tool(),
        database_tool=_no_db(),
        web_search_tool=_web_tool(sensitive_corpus, mask_terms=("某敏感违禁表述",)),
        checkpointer=MemorySaver(),
    )

    _, state = invoke("ext-3", "某科技营收", graph=graph, settings=test_settings)

    # State 中的 Evidence 已脱敏：敏感词绝不进 State。
    ev = state["evidence"][0]
    assert "某敏感违禁表述" not in ev["content"]
    assert "某敏感违禁表述" not in str(ev["raw"])
    assert "[已过滤]" in ev["content"]
    # 关键：喂给结论生成 LLM 的输入里也不含敏感原文（绝不裸奔进 LLM）。
    final_human_text = models.strong.invocations[-1][-1].content
    assert "某敏感违禁表述" not in final_human_text


def test_blocked_external_item_dropped_no_evidence(test_settings):
    """block 词命中整条 → 该条目不产 Evidence，落盲区，结论仍生成。"""
    blocked_corpus = {
        "某科技": [
            {"content": "严重违规内容不应外泄：某公司 1.2 亿元", "url": "https://b"}
        ]
    }
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [{"id": "u1", "need": "某科技营收", "sources": ["EXTERNAL_SEARCH"]}]
            ),
            refine_noop_call(),
            refine_noop_call(),
            AIMessage(content="抱歉，暂未检索到该企业的公开信息，建议核实名称后再试。"),
        ],
        fast_responses=[rewrite_call("某科技营收")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=_kb_tool(),
        database_tool=_no_db(),
        web_search_tool=_web_tool(blocked_corpus, block_terms=("严重违规内容",)),
        checkpointer=MemorySaver(),
    )

    reply, state = invoke("ext-4", "某科技营收", graph=graph, settings=test_settings)

    # 整条拦截 → 无 Evidence、单元落盲区。
    assert not state.get("evidence")
    table = CoverageTable.from_dict(state["coverage"])
    assert table.units[0].status == CoverageStatus.REMAINING
    # 结论仍产出人性化兜底，不外泄技术细节、不含被拦截的敏感原文。
    assert reply
    assert "严重违规内容" not in reply


def test_external_data_isolated_from_internal_with_weak_label(test_settings):
    """内外部隔离 + 弱化标注：同一轮既有内部 DB 又有外部联网证据 → 分源呈现、外部带「据互联网公开信息」。"""
    from dialog_agent.database_tool import build_default_database_tool, FakeDatabaseBackend

    db_rows = {
        "internship_placement_rate": [
            {"college": "机电学院", "major": "数控技术", "rate": 87.5, "sample_size": 120, "term": "2025-秋"}
        ]
    }
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [
                    # 内部单元：校内数据库（先于联网覆盖）。
                    {
                        "id": "u1",
                        "need": "机电学院数控技术实习对口率",
                        "sources": ["INTERNAL_DATABASE"],
                    },
                    # 外部单元：校内没有的企业工商信息。
                    {
                        "id": "u2",
                        "need": "某科技有限公司 2024 营收",
                        "sources": ["EXTERNAL_SEARCH"],
                    },
                ]
            ),
            # 知识库层无可覆盖单元，仍有缺失 → 精修步触发，无需精修。
            refine_noop_call(),
            # 数据库层选能力 + 填参（u1）。
            query_capability_call(
                "internship_placement_rate", {"college": "机电学院", "major": "数控技术"}
            ),
            # u1 已覆盖、u2 仍缺失 → 精修步再次触发，无需精修。
            refine_noop_call(),
            # 结论生成。
            AIMessage(
                content=(
                    "机电学院数控技术实习对口率为 87.5%（据校内数据库，2025 秋季，样本 120 人）。"
                    "另据互联网公开信息，某科技有限公司 2024 年营业收入约 1.2 亿元。"
                )
            ),
        ],
        fast_responses=[rewrite_call("某科技营收和机电学院对口率")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=_kb_tool(),
        database_tool=build_default_database_tool(FakeDatabaseBackend(db_rows)),
        web_search_tool=_web_tool(),
        checkpointer=MemorySaver(),
    )

    reply, state = invoke(
        "ext-5", "某科技营收和机电学院对口率", graph=graph, settings=test_settings
    )

    # 两类分源证据并存。
    sources = {ev["source_type"] for ev in state["evidence"]}
    assert sources == {
        SourceType.INTERNAL_DATABASE.value,
        SourceType.EXTERNAL_SEARCH.value,
    }
    # 结论里内外部分源呈现、外部带弱化标注。
    assert "据互联网公开信息" in reply
    assert "据校内数据库" in reply
    # 喂给结论生成 LLM 的素材也显式标注了外部隔离要求。
    final_human_text = models.strong.invocations[-1][-1].content
    assert "EXTERNAL_SEARCH" in final_human_text
    assert "据互联网公开信息" in final_human_text


def test_hit_and_stop_internal_covers_skips_external(test_settings):
    """命中即停：更高层（知识库）已覆盖全部单元 → 联网层零调用，不向下穿透。"""
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
    web_backend = FakeWebSearchBackend(WEB_CORPUS)
    web_tool = WebSearchTool(backend=web_backend, content_filter=ContentFilter())
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                # 单元候选源同时含知识库与联网，知识库优先覆盖。
                [
                    {
                        "id": "u1",
                        "need": "机电学院数控技术对口率",
                        "sources": ["INTERNAL_KNOWLEDGE", "EXTERNAL_SEARCH"],
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
        database_tool=_no_db(),
        web_search_tool=web_tool,
        checkpointer=MemorySaver(),
    )

    _, state = invoke("ext-6", "机电学院数控技术对口率", graph=graph, settings=test_settings)

    # 知识库已覆盖 → 联网后端零调用（命中即停，不向下穿透）。
    assert web_backend.calls == []
    ev_sources = {ev["source_type"] for ev in state["evidence"]}
    assert ev_sources == {SourceType.INTERNAL_KNOWLEDGE.value}
