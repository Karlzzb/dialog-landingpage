"""切片 1 行走骨架 —— 闲聊流单轮端到端行为测试（先行样例）。

只测外部行为：给定桩化 LLM，断言 `invoke(session_id, user_input)` 的最终回复与 State 演化，
不断言内部函数如何被调用。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from dialog_agent.chat_flow import CHAT_REPLY_MAX_CHARS
from dialog_agent.graph import build_graph, invoke
from conftest import make_stub_models, rewrite_call


def _no_tool_decision() -> AIMessage:
    """内核判定「无需工具」：不带 tool_calls 的 AIMessage。"""
    return AIMessage(content="")


def test_chitchat_single_turn_end_to_end(test_settings):
    """纯寒暄一轮：返回回复+State、不触发工具、回复 ≤50 字且尾部带业务引导。"""
    chat_reply = "您好！有产教融合相关问题随时帮您查询。"
    models = make_stub_models(
        strong_responses=[_no_tool_decision()],
        fast_responses=[
            rewrite_call("你好啊"),  # 入口改写：无记忆，原样透传。
            AIMessage(content=chat_reply),
        ],
    )
    graph = build_graph(models, checkpointer=MemorySaver())

    reply, state = invoke("sess-1", "你好啊", graph=graph, settings=test_settings)

    # 主接缝返回最终回复与完整 State。
    assert reply == state["final_reply"]
    assert reply  # 非空

    # 闲聊 ≤50 字。
    assert len(reply) <= CHAT_REPLY_MAX_CHARS

    # 尾部带业务引导（承接后把话题拉回产教融合业务）。
    assert "产教融合" in reply

    # 不触发任何检索：内核决策无 tool_calls，且轨迹中无工具消息。
    assert not any(isinstance(m, ToolMessage) for m in state["messages"])
    core_decision = next(m for m in state["messages"] if isinstance(m, AIMessage))
    assert not getattr(core_decision, "tool_calls", [])

    # 改写步（快模型）+ 对话流作答（快模型）各一次；强模型仅做一次决策。
    assert len(models.strong.invocations) == 1
    assert len(models.fast.invocations) == 2


def test_chitchat_reply_truncated_to_50_chars(test_settings):
    """快模型返回超长闲聊时，纯代码后处理截断兜底生效（≤50 字）。"""
    long_reply = "您好呀" + "很高兴见到您" * 20  # 远超 50 字
    models = make_stub_models(
        strong_responses=[_no_tool_decision()],
        fast_responses=[
            rewrite_call("在吗"),
            AIMessage(content=long_reply),
        ],
    )
    graph = build_graph(models, checkpointer=MemorySaver())

    reply, _ = invoke("sess-2", "在吗", graph=graph, settings=test_settings)

    assert len(reply) == CHAT_REPLY_MAX_CHARS


def test_sessions_are_isolated_by_thread_id(test_settings):
    """不同 session_id 各走独立 thread，互不串台（会话隔离仅靠 session id 唯一性）。"""
    models = make_stub_models(
        strong_responses=[_no_tool_decision(), _no_tool_decision()],
        fast_responses=[
            rewrite_call("你好"),
            AIMessage(content="您好，产教融合有问必答。"),
            rewrite_call("在吗"),
            AIMessage(content="在的，产教融合随时为您服务。"),
        ],
    )
    graph = build_graph(models, checkpointer=MemorySaver())

    reply_a, state_a = invoke("user-a", "你好", graph=graph, settings=test_settings)
    reply_b, state_b = invoke("user-b", "在吗", graph=graph, settings=test_settings)

    assert reply_a != reply_b
    # 各会话 State 只含本会话本轮消息，不串台。
    assert any("你好" in getattr(m, "content", "") for m in state_a["messages"])
    assert any("在吗" in getattr(m, "content", "") for m in state_b["messages"])
    assert not any("在吗" in getattr(m, "content", "") for m in state_a["messages"])
