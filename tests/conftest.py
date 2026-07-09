"""测试夹具：可编排的桩化 LLM + 关闭 tracing 的测试配置。

遵循 PRD「测试决策」：主接缝是编译后的图入口，注入桩化 LLM（预置每步返回），断言最终回复
与 State 演化，不打真实模型、不触发 Langfuse。作为后续所有测试的先行样例。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from dialog_agent.config import Settings
from dialog_agent.models import Models


class StubChatModel:
    """可编排的桩化聊天模型。

    预置一列 AIMessage，按 invoke 次序逐条返回；记录调用与 bind_tools，供断言。
    实现节点依赖的最小面：invoke / bind_tools。
    """

    def __init__(self, responses: list[AIMessage], name: str = "stub") -> None:
        self._responses = list(responses)
        self.name = name
        self.invocations: list[list] = []
        self.bound_tools: list | None = None

    def bind_tools(self, tools: list) -> "StubChatModel":
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages, **_: object) -> AIMessage:
        self.invocations.append(list(messages))
        if not self._responses:
            raise AssertionError(f"桩模型 {self.name} 的预置返回已耗尽")
        return self._responses.pop(0)


def make_stub_models(*, strong_responses: list[AIMessage], fast_responses: list[AIMessage]) -> Models:
    """构造注入图的桩化两档模型。"""
    return Models(
        strong=StubChatModel(strong_responses, name="strong"),
        fast=StubChatModel(fast_responses, name="fast"),
    )


def plan_coverage_call(units: list[dict], *, call_id: str = "call-1") -> AIMessage:
    """构造一条发起 plan_coverage 工具调用的内核决策消息（桩）。

    units 形如 [{"id": "u1", "need": "...", "sources": ["INTERNAL_KNOWLEDGE"]}]。
    """
    return AIMessage(
        content="",
        tool_calls=[{"name": "plan_coverage", "args": {"units": units}, "id": call_id}],
    )


def query_capability_call(
    capability: str, params: dict, *, call_id: str = "call-db-1"
) -> AIMessage:
    """构造一条发起 query_capability 工具调用的内核决策消息（桩）。

    模拟数据库检索层内核选能力 + 填参：{capability, params}，不含 SQL。
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_capability",
                "args": {"capability": capability, "params": params},
                "id": call_id,
            }
        ],
    )


@pytest.fixture
def test_settings() -> Settings:
    """脱离 .env 的测试配置：无模型凭据、无 Langfuse（tracing 关闭）。"""
    return Settings(_env_file=None)
