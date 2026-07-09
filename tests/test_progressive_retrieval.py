"""切片 5 —— 三层渐进检索命中即停 + 随观察精修 + 对比分析的图入口行为测试。

主接缝：编译后的图入口 `invoke(session_id, user_input)`，同时桩化三工具（知识库/数据库/联网）
+ 桩化 LLM，断言：
- 层序：知识库 → 数据库 → 联网 的调用顺序，且更高层命中即停时不向下穿透；
- 命中即停：某层满足全部剩余单元后，后续层零调用；
- 覆盖度表演化：单元随层逐步 REMAINING → COVERED、来源标注按层落位；
- 随观察精修：精修步补充的新单元被后续层拾取覆盖；修正的数据源匹配改走对应层；
- 隐性对比：对比性问题喂给结论生成步的 prompt 含对比指令、回复为对比分析而非并列罗列。
只测外部行为，不测内部函数如何被调用。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from dialog_agent.coverage import CoverageStatus, CoverageTable
from dialog_agent.database_tool import FakeDatabaseBackend, build_default_database_tool
from dialog_agent.evidence import SourceType
from dialog_agent.graph import build_graph, invoke
from dialog_agent.knowledge_tool import FakeKnowledgeRetriever
from dialog_agent.web_search_tool import (
    ContentFilter,
    FakeWebSearchBackend,
    WebSearchTool,
)
from conftest import (
    make_stub_models,
    plan_coverage_call,
    query_capability_call,
    refine_coverage_call,
    refine_noop_call,
    rewrite_call,
)

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

# 知识库假语料：查询含「对口率」即命中一条带出处的 chunk。
KB_CORPUS = {
    "对口率": [
        {
            "content": "机电学院数控技术实习对口率 87.5%。",
            "file_name": "实习统计.pdf",
            "knowledge_name": "校内统计库",
            "chunk_index": 2,
        }
    ]
}

# 联网假数据：某企业工商信息一条（数字供 raw 溯源）。
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


# ── 带调用顺序记录的桩工具：把每次调用追加进共享 order 列表，供层序断言 ──


class _OrderedKB(FakeKnowledgeRetriever):
    def __init__(self, corpus, order: list[str]) -> None:
        super().__init__(corpus)
        self._order = order

    def retrieve(self, query, *, knowledge_id=None, top_k=3):  # noqa: D401
        self._order.append("knowledge")
        return super().retrieve(query, knowledge_id=knowledge_id, top_k=top_k)


class _OrderedDB(FakeDatabaseBackend):
    def __init__(self, rows, order: list[str]) -> None:
        super().__init__(rows)
        self._order = order

    def run_query(self, sql, params):
        self._order.append("database")
        return super().run_query(sql, params)


class _OrderedWeb(FakeWebSearchBackend):
    def __init__(self, corpus, order: list[str]) -> None:
        super().__init__(corpus)
        self._order = order

    def search(self, query, *, top_k=5):
        self._order.append("external")
        return super().search(query, top_k=top_k)


def test_layer_order_and_coverage_evolution(test_settings):
    """层序 + 命中即停 + 覆盖度表演化：两单元分别落知识库与联网，DB 因无可覆盖单元早退。"""
    order: list[str] = []
    kb = _OrderedKB(KB_CORPUS, order)
    db = _OrderedDB(DB_ROWS, order)
    web = _OrderedWeb(WEB_CORPUS, order)
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [
                    {"id": "u1", "need": "机电学院数控技术对口率", "sources": ["INTERNAL_KNOWLEDGE"]},
                    {"id": "u2", "need": "某科技有限公司 2024 营收", "sources": ["EXTERNAL_SEARCH"]},
                ]
            ),
            # 知识库覆盖 u1 后仍剩 u2 → 精修步触发，无需精修。
            refine_noop_call(),
            # 数据库层无可覆盖单元（命中即停早退）→ 精修步仍因 u2 缺失触发，无需精修。
            refine_noop_call(),
            AIMessage(content="据校内统计库对口率 87.5%；另据互联网公开信息某科技营收约 1.2 亿元。"),
        ],
        fast_responses=[rewrite_call("对口率和某科技营收")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=kb,
        database_tool=build_default_database_tool(db),
        web_search_tool=WebSearchTool(backend=web, content_filter=ContentFilter()),
        checkpointer=MemorySaver(),
    )

    _, state = invoke(
        "prog-1", "对口率和某科技营收", graph=graph, settings=test_settings
    )

    # 层序：知识库先于联网；数据库因无可覆盖单元（命中即停）未被调用。
    assert order == ["knowledge", "external"]
    # 覆盖度表演化终态：两单元均覆盖、无剩余。
    table = CoverageTable.from_dict(state["coverage"])
    assert {u.id for u in table.units} == {"u1", "u2"}
    assert all(u.status == CoverageStatus.COVERED for u in table.units)
    assert table.remaining_units == []
    # 来源标注按层落位：u1 落知识库出处、u2 落联网 URL。
    by_id = {u.id: u for u in table.units}
    assert any("实习统计.pdf" in c for c in by_id["u1"].citations)
    assert any("example.com" in c for c in by_id["u2"].citations)
    # 分源 Evidence 累积：知识库与联网各一。
    assert {ev["source_type"] for ev in state["evidence"]} == {
        SourceType.INTERNAL_KNOWLEDGE.value,
        SourceType.EXTERNAL_SEARCH.value,
    }


def test_refine_adds_unit_covered_downstream(test_settings):
    """随观察补充新单元：精修步追加的 INTERNAL_DATABASE 单元被数据库层拾取并覆盖。"""
    db = FakeDatabaseBackend(DB_ROWS)
    models = make_stub_models(
        strong_responses=[
            # 内核首步只拆出 u1（数据库单元）。
            plan_coverage_call(
                [{"id": "u1", "need": "机电学院数控技术对口率", "sources": ["INTERNAL_DATABASE"]}]
            ),
            # 知识库层无可覆盖单元、u1 仍缺失 → 精修步触发，追加新单元 u2（同为数据库源）。
            refine_coverage_call(
                add_units=[
                    {"id": "u2", "need": "机电学院机电技术对口率", "sources": ["INTERNAL_DATABASE"]}
                ]
            ),
            # 数据库层覆盖 u1。
            query_capability_call(
                "internship_placement_rate", {"college": "机电学院", "major": "数控技术"}
            ),
            # 数据库层覆盖精修追加的 u2（同一能力、不同参数）。
            query_capability_call(
                "internship_placement_rate", {"college": "机电学院", "major": "机电技术"}
            ),
            AIMessage(content="数控技术与机电技术对口率均已查到。"),
        ],
        fast_responses=[rewrite_call("机电学院对口率")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever({}),
        database_tool=build_default_database_tool(db),
        web_search_tool=WebSearchTool(
            backend=FakeWebSearchBackend({}), content_filter=ContentFilter()
        ),
        checkpointer=MemorySaver(),
    )

    _, state = invoke(
        "prog-2", "机电学院对口率", graph=graph, settings=test_settings
    )

    # 精修追加的 u2 出现在覆盖度表且被数据库层覆盖（被后续层拾取）。
    table = CoverageTable.from_dict(state["coverage"])
    by_id = {u.id: u for u in table.units}
    assert {"u1", "u2"} <= set(by_id)
    assert by_id["u2"].status == CoverageStatus.COVERED
    # 数据库后端被调用两次（u1 + 精修追加的 u2）。
    assert len(db.calls) == 2


def test_refine_reassigns_source_match(test_settings):
    """修正数据源匹配：单元初匹配知识库、未命中，精修步改匹配为数据库后被数据库层覆盖。"""
    db = FakeDatabaseBackend(DB_ROWS)
    models = make_stub_models(
        strong_responses=[
            # 内核首步把 u1 匹配到知识库。
            plan_coverage_call(
                [{"id": "u1", "need": "机电学院数控技术对口率", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
            # 知识库未命中、u1 仍缺失 → 精修步触发，把 u1 改匹配为数据库。
            refine_coverage_call(
                reassign=[{"id": "u1", "sources": ["INTERNAL_DATABASE"]}]
            ),
            # 数据库层覆盖（改匹配后由数据库层拾取）。
            query_capability_call(
                "internship_placement_rate", {"college": "机电学院", "major": "数控技术"}
            ),
            AIMessage(content="据校内数据库，对口率 87.5%。"),
        ],
        fast_responses=[rewrite_call("机电学院数控技术对口率")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever({}),  # 知识库空语料，模拟未命中
        database_tool=build_default_database_tool(db),
        web_search_tool=WebSearchTool(
            backend=FakeWebSearchBackend({}), content_filter=ContentFilter()
        ),
        checkpointer=MemorySaver(),
    )

    _, state = invoke(
        "prog-3", "机电学院数控技术对口率", graph=graph, settings=test_settings
    )

    table = CoverageTable.from_dict(state["coverage"])
    unit = table.units[0]
    # 数据源匹配已被修正为数据库。
    assert unit.source_matches == [SourceType.INTERNAL_DATABASE]
    # 且经数据库层覆盖。
    assert unit.status == CoverageStatus.COVERED
    assert len(db.calls) == 1


def test_implicit_comparison_produces_analysis(test_settings):
    """隐性对比：「这两个专业就业率」→ 喂给结论生成步的 prompt 含对比指令、回复为对比分析。"""
    db = FakeDatabaseBackend(DB_ROWS)
    comparison_reply = (
        "数控技术就业率 87.5%，高于机电技术的 82.0%，建议优先扩数控方向。"
    )
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [
                    {"id": "u1", "need": "机电学院数控技术就业率", "sources": ["INTERNAL_DATABASE"]},
                    {"id": "u2", "need": "机电学院机电技术就业率", "sources": ["INTERNAL_DATABASE"]},
                ]
            ),
            # 知识库层无可覆盖单元、两单元仍缺失 → 精修步触发，无需精修。
            refine_noop_call(),
            # 数据库层覆盖 u1。
            query_capability_call(
                "internship_placement_rate", {"college": "机电学院", "major": "数控技术"}
            ),
            # 数据库层覆盖 u2。
            query_capability_call(
                "internship_placement_rate", {"college": "机电学院", "major": "机电技术"}
            ),
            AIMessage(content=comparison_reply),
        ],
        fast_responses=[rewrite_call("这两个专业就业率")],
    )
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever({}),
        database_tool=build_default_database_tool(db),
        web_search_tool=WebSearchTool(
            backend=FakeWebSearchBackend({}), content_filter=ContentFilter()
        ),
        checkpointer=MemorySaver(),
    )

    reply, state = invoke(
        "prog-4", "这两个专业就业率", graph=graph, settings=test_settings
    )

    # 两单元均覆盖。
    table = CoverageTable.from_dict(state["coverage"])
    assert all(u.status == CoverageStatus.COVERED for u in table.units)
    # 喂给结论生成步的 prompt 含对比分析指令（非事后质检，由素材结构 + prompt 导出）。
    final_human_text = models.strong.invocations[-1][-1].content
    assert "对比分析" in final_human_text
    # 回复为对比分析（含比较与建议），而非仅并列罗列数据。
    assert reply == comparison_reply
    assert "高于" in reply
