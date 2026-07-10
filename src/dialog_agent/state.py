"""单轮 State schema。

本轮用完即弃的最小状态。切片 2 在切片 1（输入/改写/消息轨迹/最终回复）之上，承载知识流
所需的结构化对象：覆盖度表、逐层累积的分源 Evidence、以及出口截断所依据的流别。
切片 6 增设跨轮持久的 `session_memory`（结构化会话记忆，ADR 0003）——单轮字段每轮入口重置，
仅会话记忆跨轮续接，故 LLM 不被喂全量对话历史。
切片 8 增设安全阀计数（`tool_call_count` / `iteration_count`）：触顶即兜底强制作答。
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
    # 改写步产出的自包含查询（切片 6 起基于结构化会话记忆做指代消解）。
    rewritten_query: str
    # 结构化会话记忆（ADR 0003）：用户画像 + 滚动关键实体/约束摘要；跨轮持久，每轮入口更新。
    # SessionMemory.as_dict() 往返；改写步读它做指代消解、写它合并本轮实体。
    session_memory: dict[str, Any]
    # ReAct 内核的消息轨迹（规划/工具决策等）；add_messages 负责累加。
    messages: Annotated[list[AnyMessage], add_messages]
    # 覆盖度表（ADR 0002）：内核首步产出、每层查询后更新的结构化 dict（CoverageTable.as_dict）。
    coverage: dict[str, Any]
    # 逐层累积的分源 Evidence（ADR 0007，Evidence.as_dict）；operator.add 使各层追加不覆盖。
    evidence: Annotated[list[dict[str, Any]], operator.add]
    # 安全阀计数（切片 8）：本轮检索工具实际调用次数 / 检索层+精修步的节点访问数。
    # 触顶即经条件边短路到 final_answer 兜底强制作答，不死循环烧 token。阈值见 SafetyCaps。
    tool_call_count: int
    iteration_count: int
    # 本轮流别："chat"（对话流）/"knowledge"（知识流）；finalize 据此决定是否 ≤50 字截断。
    flow: str
    # 最终呈现给用户的回复。
    final_reply: str
