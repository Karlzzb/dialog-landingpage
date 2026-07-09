"""图节点 —— 确定性骨架的入口/出口 + ReAct 内核（切片 1 最薄形态）。

节点均为闭包工厂，绑定注入的 `Models`，使 `build_graph(models=...)` 可用桩驱动。
本切片路径：rewrite（透传）→ react_core（内核判定是否用工具）→ chat_flow_answer（对话流
作答）→ finalize（≤50 字截断）。工具循环回边在切片 2 引入检索工具时接入。
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .chat_flow import CHAT_FLOW_SYSTEM_PROMPT, truncate_reply
from .models import Models
from .state import TurnState

# ReAct 内核的 system prompt：自主判定「要不要调检索工具」，不设入口硬分类器。
# 切片 1 尚无检索工具可用，内核对纯寒暄自然判定无需工具 → 走对话流出口。
REACT_CORE_SYSTEM_PROMPT = (
    "你是「产教融合专家助理」的推理内核。"
    "先判断用户诉求是否需要检索客观事实（政策/制度/标准/校内统计数据等）。"
    "需要时调用相应检索工具获取依据；若只是寒暄或与业务无关的闲聊，则无需任何工具，"
    "直接进入对话流作答。"
)


def make_rewrite_node() -> Callable[[TurnState], dict[str, Any]]:
    """入口改写步。

    切片 1 为透传（不接会话记忆）：把原始输入原样作为自包含查询，并作为内核的首条消息。
    切片（会话记忆）会在此基于滚动实体摘要做指代消解，接口位置不变。
    """

    def rewrite(state: TurnState) -> dict[str, Any]:
        user_input = state["user_input"]
        return {
            "rewritten_query": user_input,
            "messages": [HumanMessage(content=user_input)],
        }

    return rewrite


def make_react_core_node(models: Models, tools: list) -> Callable[[TurnState], dict[str, Any]]:
    """ReAct 内核决策步（强模型）。

    绑定当前可用工具后调用模型：模型自主决定发起工具调用还是不用工具。产出的 AIMessage
    进入 messages，由条件边依据其 `tool_calls` 是否为空决定后续路由。
    """

    def react_core(state: TurnState) -> dict[str, Any]:
        llm = models.strong.bind_tools(tools) if tools else models.strong
        messages = [SystemMessage(content=REACT_CORE_SYSTEM_PROMPT), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    return react_core


def make_chat_flow_answer_node(models: Models) -> Callable[[TurnState], dict[str, Any]]:
    """对话流作答步（快模型）。

    内核判定无需工具后的自然出口：按对话流约束（专家口吻、尾部业务引导、≤50 字）生成闲聊
    回复。原始文本存入 final_reply，由 finalize 做截断兜底。
    """

    def chat_flow_answer(state: TurnState) -> dict[str, Any]:
        messages = [
            SystemMessage(content=CHAT_FLOW_SYSTEM_PROMPT),
            HumanMessage(content=state["rewritten_query"]),
        ]
        response: AIMessage = models.fast.invoke(messages)
        return {
            "messages": [response],
            "final_reply": response.content or "",
        }

    return chat_flow_answer


def make_finalize_node() -> Callable[[TurnState], dict[str, Any]]:
    """出口后处理：对话流回复 ≤50 字纯代码截断兜底。"""

    def finalize(state: TurnState) -> dict[str, Any]:
        return {"final_reply": truncate_reply(state.get("final_reply", ""))}

    return finalize


def route_after_core(state: TurnState) -> str:
    """内核后的条件路由：模型发起工具调用 → 走工具（未来切片）；否则 → 对话流作答。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "chat_flow_answer"
