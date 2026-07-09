"""切片 6 —— 结构化会话记忆 + 自包含查询改写（多轮）的图入口行为测试。

主接缝：编译后的图入口 `invoke(session_id, user_input)`，借 checkpointer 跨两轮驱动同一
会话，注入桩化知识库检索器 + 桩化 LLM，断言：
- 省略句被改写为自包含查询：轮2「江夏区有额外的吗」继承轮1的「武汉市/校企合作」→
  「武汉市江夏区校企合作补贴」，且内核收到的正是该自包含查询；
- 会话记忆跨轮持久并每轮更新：实体摘要从轮1继承到轮2；
- 单轮消息轨迹本轮用完即弃：轮2的内核不被喂轮1的全量历史；
- 混合意图先承接闲聊再进入查询：「你好，帮我查下政策」走知识流（改写不做流程分叉），
  回复先承接后答；
- 改写脏输出降级：快模型未发起工具调用时兜底为本轮输入，不阻断整轮。
只测外部行为，不测内部函数如何被调用。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from dialog_agent.graph import build_graph, invoke
from dialog_agent.knowledge_tool import FakeKnowledgeRetriever
from dialog_agent.session_memory import SessionMemory
from conftest import make_stub_models, plan_coverage_call, rewrite_call, rewrite_noop_call

# 知识库假语料：查询含「校企合作补贴」即命中一条带出处的 chunk（轮1/轮2均覆盖）。
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


def test_omitted_sentence_completed_via_memory_across_turns(test_settings):
    """省略句补全：轮1建立「武汉市/校企合作」→ 轮2「江夏区有额外的吗」改写为自包含查询。"""
    turn1_final = (
        "据武汉市产教融合政策库《校企合作补贴办法.pdf》，武汉市对开展校企合作的企业，"
        "按每生每年 2000 元的标准给予补贴。"
    )
    turn2_final = (
        "据武汉市产教融合政策库，江夏区校企合作补贴按市级标准每生每年 2000 元执行。"
    )
    models = make_stub_models(
        strong_responses=[
            # 轮1：内核规划 + 结论。
            plan_coverage_call(
                [{"id": "u1", "need": "武汉市校企合作补贴标准", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
            AIMessage(content=turn1_final),
            # 轮2：内核规划 + 结论。
            plan_coverage_call(
                [{"id": "u1", "need": "武汉市江夏区校企合作补贴", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
            AIMessage(content=turn2_final),
        ],
        fast_responses=[
            # 轮1改写：原样自包含，并落地实体摘要（地域/主题）。
            rewrite_call(
                "武汉市校企合作补贴是多少",
                entities={"地域": "武汉市", "主题": "校企合作"},
            ),
            # 轮2改写：读会话记忆，把省略句补全为自包含查询。
            rewrite_call(
                "武汉市江夏区校企合作补贴",
                entities={"地域": "武汉市江夏区", "主题": "校企合作"},
            ),
        ],
    )
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever(CORPUS),
        checkpointer=MemorySaver(),
    )

    # 跨两轮驱动同一会话（thread_id = session id）。
    invoke("mem-1", "武汉市校企合作补贴是多少", graph=graph, settings=test_settings)
    reply2, state2 = invoke(
        "mem-1", "江夏区有额外的吗", graph=graph, settings=test_settings
    )

    # 轮2被改写为自包含查询（继承轮1的地域/主题）。
    assert state2["rewritten_query"] == "武汉市江夏区校企合作补贴"
    # 结论为轮2强制作答产物。
    assert reply2 == turn2_final

    # 会话记忆跨轮持久并更新：地域从「武汉市」精修为「武汉市江夏区」，主题继承。
    memory = SessionMemory.from_dict(state2["session_memory"])
    assert memory.entities["主题"] == "校企合作"
    assert "江夏区" in memory.entities["地域"]

    # 内核轮2收到的正是自包含查询（而非省略句原文「江夏区有额外的吗」）。
    turn2_core_input = models.strong.invocations[2]  # 轮2的 plan_coverage 调用
    assert turn2_core_input[-1].content == "武汉市江夏区校企合作补贴"
    assert "江夏区有额外的吗" not in turn2_core_input[-1].content

    # 单轮消息轨迹本轮用完即弃：轮2的内核不被喂轮1的全量历史（轮1结论不进轮2内核输入）。
    turn1_final_fragment = "按每生每年 2000 元的标准给予补贴"
    assert all(
        turn1_final_fragment not in getattr(m, "content", "")
        for m in turn2_core_input
    )


def test_mixed_intent_acknowledges_then_enters_query(test_settings):
    """混合意图「你好，帮我查下政策」→ 走知识流（改写不做流程分叉），回复先承接后答。"""
    final_reply = "您好！产教融合政策方面，请问您想了解校企合作、专业设置还是补贴标准？"
    models = make_stub_models(
        strong_responses=[
            # 内核识别业务意图 → plan_coverage（混合输入不被误判为纯寒暄）。
            plan_coverage_call(
                [{"id": "u1", "need": "产教融合校企合作补贴政策", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
            AIMessage(content=final_reply),
        ],
        fast_responses=[
            # 改写：保留承接前缀、补全业务诉求，不做流程分叉。
            rewrite_call("你好，帮我查下产教融合相关政策"),
        ],
    )
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever(CORPUS),
        checkpointer=MemorySaver(),
    )

    reply, state = invoke(
        "mix-1", "你好，帮我查下政策", graph=graph, settings=test_settings
    )

    # 改写保留承接+补全业务：自包含查询同时含寒暄与业务诉求。
    assert "你好" in state["rewritten_query"]
    assert "产教融合" in state["rewritten_query"]
    # 流向知识流（plan_coverage 被调用），证明改写未把业务误丢、未做流程分叉。
    assert state.get("coverage")
    assert state.get("flow") == "knowledge"
    # 回复先承接闲聊、再进入查询（含承接与业务引导）。
    assert "您好" in reply
    assert "产教融合" in reply


def test_rewrite_dirty_output_falls_back_to_passthrough(test_settings):
    """改写脏输出降级：快模型未发起工具调用 → 兜底为本轮输入，不阻断整轮。"""
    final_reply = "据武汉市产教融合政策库，校企合作有专项补贴。"
    models = make_stub_models(
        strong_responses=[
            plan_coverage_call(
                [{"id": "u1", "need": "武汉市校企合作补贴", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
            AIMessage(content=final_reply),
        ],
        fast_responses=[
            # 改写步脏输出：无 tool_calls（parse 返回 None）→ 兜底透传本轮输入。
            rewrite_noop_call("武汉市校企合作补贴是多少"),
        ],
    )
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever(CORPUS),
        checkpointer=MemorySaver(),
    )

    reply, state = invoke(
        "dirty-1", "武汉市校企合作补贴是多少", graph=graph, settings=test_settings
    )

    # 脏输出兜底：rewritten_query 为本轮输入，记忆保持空（不阻断）。
    assert state["rewritten_query"] == "武汉市校企合作补贴是多少"
    assert SessionMemory.from_dict(state["session_memory"]).entities == {}
    # 整轮仍正常完成、走知识流。
    assert reply == final_reply
    assert state.get("flow") == "knowledge"


def test_rewrite_only_resolves_reference_no_flow_fork(test_settings):
    """改写仅指代消解、不做流程分叉：纯寒暄仍走对话流、业务查询仍走知识流，由内核判定。"""
    models = make_stub_models(
        strong_responses=[
            # 轮1：纯寒暄 → 内核无工具 → 对话流。
            AIMessage(content=""),
            # 轮2：业务查询 → 内核 plan_coverage → 知识流。
            plan_coverage_call(
                [{"id": "u1", "need": "武汉市校企合作补贴标准", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
            AIMessage(content="据武汉市产教融合政策库，校企合作按每生每年 2000 元补贴。"),
        ],
        fast_responses=[
            # 轮1改写：寒暄原样，不改意图、不分流。
            rewrite_call("你好"),
            AIMessage(content="您好！产教融合有问必答。"),
            # 轮2改写：业务查询原样自包含。
            rewrite_call("武汉市校企合作补贴是多少"),
        ],
    )
    graph = build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever(CORPUS),
        checkpointer=MemorySaver(),
    )

    reply1, state1 = invoke("fork-1", "你好", graph=graph, settings=test_settings)
    reply2, state2 = invoke(
        "fork-1", "武汉市校企合作补贴是多少", graph=graph, settings=test_settings
    )

    # 轮1纯寒暄 → 对话流（内核判定无需工具，非改写分流）。
    assert state1.get("flow") == "chat"
    assert not state1.get("coverage")
    # 轮2业务查询 → 知识流（同一会话，改写不改意图）。
    assert state2.get("flow") == "knowledge"
    assert state2.get("coverage")
    assert "校企合作" in reply2
