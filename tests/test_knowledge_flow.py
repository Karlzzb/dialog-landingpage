"""切片 2 —— 单信息单元、单数据源（知识库）命中即作答的图入口行为测试。

主接缝：编译后的图入口 `invoke(session_id, user_input)`，注入桩化知识库检索器 + 桩化 LLM，
断言覆盖度表演化（信息单元 → 已覆盖、remaining_units==0）与结论的来源标注。只测外部行为，
不测内部函数如何被调用。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from dialog_agent.coverage import CoverageStatus, CoverageTable
from dialog_agent.evidence import SourceType
from dialog_agent.graph import build_graph, invoke
from dialog_agent.knowledge_tool import FakeKnowledgeRetriever
from conftest import make_stub_models, plan_coverage_call, rewrite_call

# 知识库假语料：查询含「校企合作补贴」即命中一条带出处的 chunk。
CORPUS = {
    "校企合作补贴": [
        {
            "content": "武汉市对校企合作企业给予每生每年 2000 元补贴。",
            "file_name": "校企合作补贴办法.pdf",
            "knowledge_name": "武汉市产教融合政策库",
            "chunk_index": 7,
        }
    ]
}


def _knowledge_models(final_reply: str, user_input: str) -> object:
    """强模型：首步发起 plan_coverage 工具调用，末步产出结论。快模型仅做入口改写。"""
    return make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [{"id": "u1", "need": "武汉市校企合作补贴标准", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
            AIMessage(content=final_reply),
        ],
        fast_responses=[rewrite_call(user_input)],
    )


def test_knowledge_single_unit_hit_and_answer(test_settings):
    """单信息单元经知识库命中 → 覆盖度表标记已覆盖、remaining==0 → 强制作答带来源标注。"""
    final_reply = (
        "据武汉市产教融合政策库《校企合作补贴办法.pdf》记载，武汉市对开展校企合作的企业，"
        "按每生每年 2000 元的标准给予补贴。"
    )
    models = _knowledge_models(final_reply, "武汉市校企合作补贴是多少？")
    retriever = FakeKnowledgeRetriever(CORPUS)
    graph = build_graph(models, knowledge_retriever=retriever, checkpointer=MemorySaver())

    reply, state = invoke(
        "kb-1", "武汉市校企合作补贴是多少？", graph=graph, settings=test_settings
    )

    # 覆盖度表存于 State 且结构化：单元已覆盖、无剩余。
    table = CoverageTable.from_dict(state["coverage"])
    assert len(table.units) == 1
    unit = table.units[0]
    assert unit.status == CoverageStatus.COVERED
    assert table.remaining_units == []

    # 命中单元记录了可回溯来源标注（知识库/文件/chunk）。
    assert any("校企合作补贴办法.pdf" in c for c in unit.citations)

    # 分源 Evidence 累积进 State，source_type 为 INTERNAL_KNOWLEDGE。
    assert state["evidence"]
    ev = state["evidence"][0]
    assert ev["source_type"] == SourceType.INTERNAL_KNOWLEDGE.value
    assert ev["citation"] in unit.citations
    # 数字可溯源：raw 保留原始 chunk。
    assert ev["raw"]["chunk_index"] == 7

    # 结论为强制作答产物，引用内部数据并带来源标注。
    assert reply == final_reply
    assert "校企合作补贴办法.pdf" in reply

    # 知识流不做 ≤50 字截断（截断仅对话流）。
    assert len(reply) > 50

    # 知识库被查询恰一次（命中即停，未穿透下层）。
    assert len(retriever.calls) == 1


def test_knowledge_flow_evidence_prompt_reaches_final_model(test_settings):
    """结论生成步确实拿到分源 Evidence（素材结构导出来源标注，非事后质检）。"""
    models = _knowledge_models("结论。", "校企合作补贴标准")
    retriever = FakeKnowledgeRetriever(CORPUS)
    graph = build_graph(models, knowledge_retriever=retriever, checkpointer=MemorySaver())

    invoke("kb-2", "校企合作补贴标准", graph=graph, settings=test_settings)

    # 强模型第二次调用（结论生成）的输入里含检索到的证据内容与出处。
    final_call_messages = models.strong.invocations[1]
    human_text = final_call_messages[-1].content
    assert "2000 元补贴" in human_text
    assert "校企合作补贴办法.pdf" in human_text


def test_knowledge_flow_records_tool_message(test_settings):
    """知识流保留合法消息轨迹：plan_coverage 工具调用有对应 ToolMessage 回应。"""
    models = _knowledge_models("结论。", "校企合作补贴标准")
    graph = build_graph(
        models, knowledge_retriever=FakeKnowledgeRetriever(CORPUS), checkpointer=MemorySaver()
    )

    _, state = invoke("kb-3", "校企合作补贴标准", graph=graph, settings=test_settings)

    assert any(isinstance(m, ToolMessage) for m in state["messages"])


def test_chitchat_still_bypasses_knowledge(test_settings):
    """纯寒暄仍走对话流：内核不调 plan_coverage，知识库零调用，无覆盖度表。"""
    retriever = FakeKnowledgeRetriever(CORPUS)
    models = make_stub_models(
        strong_responses=[AIMessage(content="")],  # 无 tool_calls
        fast_responses=[
            rewrite_call("你好啊"),
            AIMessage(content="您好！产教融合相关问题随时为您服务。"),
        ],
    )
    graph = build_graph(models, knowledge_retriever=retriever, checkpointer=MemorySaver())

    reply, state = invoke("kb-4", "你好啊", graph=graph, settings=test_settings)

    assert retriever.calls == []
    assert not state.get("coverage")
    assert not state.get("evidence")
    assert len(reply) <= 50  # 对话流截断兜底
