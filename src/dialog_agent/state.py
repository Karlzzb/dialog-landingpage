"""单轮 State schema。

本轮用完即弃的最小状态。切片 1 只需承载：原始输入、改写后的自包含查询、ReAct 内核的
消息轨迹、最终回复。后续切片在此扩展覆盖度表、分源 Evidence 等字段。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class TurnState(TypedDict, total=False):
    """一轮对话的图状态。"""

    # 后端每轮传入的原始用户输入。
    user_input: str
    # 改写步产出的自包含查询（切片 1 为透传，不接会话记忆）。
    rewritten_query: str
    # ReAct 内核的消息轨迹（工具决策等）；add_messages 负责累加。
    messages: Annotated[list[AnyMessage], add_messages]
    # 最终呈现给用户的回复（对话流经 ≤50 字截断兜底）。
    final_reply: str
