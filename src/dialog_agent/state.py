"""单轮 State schema。

本轮用完即弃的最小状态。切片 2 在切片 1（输入/改写/消息轨迹/最终回复）之上，承载知识流
所需的结构化对象：覆盖度表、逐层累积的分源 Evidence、以及出口截断所依据的流别。
后续切片再扩展会话记忆、数据库/联网 Evidence 等字段。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class TurnState(TypedDict, total=False):
    """一轮对话的图状态。"""

    # 后端每轮传入的原始用户输入。
    user_input: str
    # 改写步产出的自包含查询（切片 2 仍为透传，不接会话记忆）。
    rewritten_query: str
    # ReAct 内核的消息轨迹（规划/工具决策等）；add_messages 负责累加。
    messages: Annotated[list[AnyMessage], add_messages]
    # 覆盖度表（ADR 0002）：内核首步产出、每层查询后更新的结构化 dict（CoverageTable.as_dict）。
    coverage: dict[str, Any]
    # 逐层累积的分源 Evidence（ADR 0007，Evidence.as_dict）；operator.add 使各层追加不覆盖。
    evidence: Annotated[list[dict[str, Any]], operator.add]
    # 本轮流别："chat"（对话流）/"knowledge"（知识流）；finalize 据此决定是否 ≤50 字截断。
    flow: str
    # 最终呈现给用户的回复。
    final_reply: str
