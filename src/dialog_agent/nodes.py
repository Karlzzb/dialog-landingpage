"""图节点 —— 确定性骨架的入口/出口 + ReAct 内核（切片 2：接入首个知识流路径）。

节点均为闭包工厂，绑定注入的 `Models` / 知识库检索器，使 `build_graph(...)` 可用桩确定性
驱动。切片 2 路径：

    rewrite（透传）
      → react_core（内核首步：需事实→调 plan_coverage 产覆盖度表；纯寒暄→无工具）
          ├─(无工具)→ chat_flow_answer → finalize（≤50 字截断）
          └─(plan_coverage)→ build_coverage（解析覆盖度表）
                → retrieve_knowledge（查知识库、更新覆盖状态、累积 Evidence）
                    → final_answer（remaining_units==0 强制作答，内部数据带来源标注）
                        → finalize（知识流不截断）

数据库/联网层、盲区/降级路径在后续切片沿同一覆盖度表接入。
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .chat_flow import CHAT_FLOW_SYSTEM_PROMPT, truncate_reply
from .coverage import (
    CoverageTable,
    citations_of,
    parse_plan_coverage,
    plan_coverage,
)
from .evidence import Evidence, SourceType
from .knowledge_tool import KnowledgeRetriever
from .models import Models
from .state import TurnState

# 流别标识：finalize 据此决定是否对回复做 ≤50 字截断。
FLOW_CHAT = "chat"
FLOW_KNOWLEDGE = "knowledge"

# ReAct 内核首步的 system prompt：不设入口硬分类器。内核自主判断是否需要检索客观事实；
# 需要则调 plan_coverage 产出结构化覆盖度表（拆信息单元 + 匹配数据源），纯寒暄则不调工具。
REACT_CORE_SYSTEM_PROMPT = (
    "你是「产教融合专家助理」的推理内核。"
    "先判断用户诉求：若需要检索客观事实（政策/制度/标准/校内统计数据等），"
    "必须调用 plan_coverage 工具，把问题拆成信息单元并为每个单元匹配候选数据源"
    "（知识库 INTERNAL_KNOWLEDGE / 数据库 INTERNAL_DATABASE / 联网 EXTERNAL_SEARCH，按可信优先级）。"
    "若只是寒暄或与业务无关的闲聊，则不要调用任何工具，直接进入对话流作答。"
)

# 结论生成（强制作答）的 system prompt：关闭工具，逼模型基于 State 已有 Evidence 直接答。
# 内外部隔离、来源标注、数字取自 raw —— 由素材结构 + 本 prompt 自然导出（非事后质检）。
FINAL_ANSWER_SYSTEM_PROMPT = (
    "你是「产教融合专家助理」。以下是已检索到的分源证据。请仅依据这些证据作答，不要臆造。"
    "内部知识库数据可直接引用，并在结论中标注其来源（文件/知识库/条款）。"
    "每个关键结论都要能追溯到给定证据；若证据不足以覆盖某信息单元，请坦诚说明边界，不要编造。"
)


def make_rewrite_node() -> Callable[[TurnState], dict[str, Any]]:
    """入口改写步。

    切片 2 仍为透传（不接会话记忆）：原始输入原样作为自包含查询与内核首条消息。
    会话记忆切片会在此基于滚动实体摘要做指代消解，接口位置不变。
    """

    def rewrite(state: TurnState) -> dict[str, Any]:
        user_input = state["user_input"]
        return {
            "rewritten_query": user_input,
            "messages": [HumanMessage(content=user_input)],
        }

    return rewrite


def make_react_core_node(models: Models) -> Callable[[TurnState], dict[str, Any]]:
    """ReAct 内核首步（强模型）。

    绑定 plan_coverage 规划工具后调用模型：需事实则发起 plan_coverage 工具调用，纯寒暄则
    不调工具。产出的 AIMessage 进入 messages，由条件边依据其 tool_calls 是否为空路由。
    """

    def react_core(state: TurnState) -> dict[str, Any]:
        llm = models.strong.bind_tools([plan_coverage])
        messages = [SystemMessage(content=REACT_CORE_SYSTEM_PROMPT), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    return react_core


def make_build_coverage_node() -> Callable[[TurnState], dict[str, Any]]:
    """把内核的 plan_coverage 工具调用解析成结构化覆盖度表，存入 State。

    同时回应一条 ToolMessage 以保持消息轨迹合法（工具调用需有对应结果），并标记本轮为知识流。
    """

    def build_coverage(state: TurnState) -> dict[str, Any]:
        last = state["messages"][-1]
        call = _first_plan_call(last)
        table = parse_plan_coverage(call["args"])
        ack = ToolMessage(
            content=f"覆盖度表已生成，共 {len(table.units)} 个信息单元。",
            tool_call_id=call.get("id", ""),
        )
        return {
            "coverage": table.as_dict(),
            "flow": FLOW_KNOWLEDGE,
            "messages": [ack],
        }

    return build_coverage


def make_retrieve_knowledge_node(
    retriever: KnowledgeRetriever,
) -> Callable[[TurnState], dict[str, Any]]:
    """知识库检索层（渐进检索的第一层）。

    对覆盖度表中仍缺失、且候选源含 INTERNAL_KNOWLEDGE 的每个信息单元查知识库；命中则把该
    单元标记为已覆盖并记录来源标注，Evidence 追加进 State（分源累积）。未命中的单元保持缺失
    （留待后续层，切片 2 只有此一层）。
    """

    def retrieve_knowledge(state: TurnState) -> dict[str, Any]:
        table = CoverageTable.from_dict(state.get("coverage"))
        new_evidence: list[Evidence] = []
        for unit in table.remaining_for_source(SourceType.INTERNAL_KNOWLEDGE):
            hits = retriever.retrieve(unit.need)
            if hits:
                new_evidence.extend(hits)
                table.mark_covered(unit.id, citations_of(hits))
        return {
            "coverage": table.as_dict(),
            "evidence": [ev.as_dict() for ev in new_evidence],
        }

    return retrieve_knowledge


def make_final_answer_node(models: Models) -> Callable[[TurnState], dict[str, Any]]:
    """结论生成（强制作答，强模型）。

    loop 终止时的终止动作：关闭工具调用，逼模型基于 State 已累积的分源 Evidence 直接作答。
    内部数据直接引用并带来源标注、数字取自 raw —— 由素材结构 + prompt 导出。
    """

    def final_answer(state: TurnState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        table = CoverageTable.from_dict(state.get("coverage"))
        prompt = _render_evidence_prompt(state["rewritten_query"], evidence, table)
        response: AIMessage = models.strong.invoke(
            [
                SystemMessage(content=FINAL_ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        return {
            "messages": [response],
            "final_reply": response.content or "",
        }

    return final_answer


def make_chat_flow_answer_node(models: Models) -> Callable[[TurnState], dict[str, Any]]:
    """对话流作答步（快模型）。

    内核判定无需工具后的自然出口：按对话流约束（专家口吻、尾部业务引导、≤50 字）生成闲聊
    回复。标记本轮为对话流，原始文本存入 final_reply，由 finalize 做截断兜底。
    """

    def chat_flow_answer(state: TurnState) -> dict[str, Any]:
        messages = [
            SystemMessage(content=CHAT_FLOW_SYSTEM_PROMPT),
            HumanMessage(content=state["rewritten_query"]),
        ]
        response: AIMessage = models.fast.invoke(messages)
        return {
            "messages": [response],
            "flow": FLOW_CHAT,
            "final_reply": response.content or "",
        }

    return chat_flow_answer


def make_finalize_node() -> Callable[[TurnState], dict[str, Any]]:
    """出口后处理：仅对话流回复做 ≤50 字纯代码截断兜底；知识流结论不截断。"""

    def finalize(state: TurnState) -> dict[str, Any]:
        reply = state.get("final_reply", "")
        if state.get("flow") == FLOW_CHAT:
            reply = truncate_reply(reply)
        return {"final_reply": reply}

    return finalize


def route_after_core(state: TurnState) -> str:
    """内核后的条件路由：发起 plan_coverage 工具调用 → 建覆盖度表走知识流；否则 → 对话流。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "build_coverage"
    return "chat_flow_answer"


def _first_plan_call(message: Any) -> dict[str, Any]:
    """取出消息里的首个 plan_coverage 工具调用（规范化为 dict）。"""
    for call in getattr(message, "tool_calls", None) or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name == plan_coverage.__name__:
            args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
            cid = call.get("id", "") if isinstance(call, dict) else getattr(call, "id", "")
            return {"name": name, "args": args or {}, "id": cid or ""}
    raise ValueError("内核消息中未找到 plan_coverage 工具调用")


def _render_evidence_prompt(
    query: str, evidence: list[dict[str, Any]], table: CoverageTable
) -> str:
    """把自包含查询 + 分源 Evidence + 覆盖状态渲染成结论生成的输入。

    Evidence 按来源标注逐条列出（content + citation），使模型作答时能直接引用并标注来源；
    仍缺失的信息单元显式列为盲区提示，导出坦诚告知边界的行为。
    """
    lines = [f"用户问题：{query}", "", "已检索证据："]
    if evidence:
        for ev in evidence:
            lines.append(
                f"- [{ev['source_type']}] {ev['content']}（来源：{ev['citation']}）"
            )
    else:
        lines.append("（暂无证据）")

    remaining = table.remaining_units
    if remaining:
        lines.append("")
        lines.append("仍未覆盖的信息单元（如无其他来源请坦诚说明边界）：")
        lines.extend(f"- {u.need}" for u in remaining)
    return "\n".join(lines)
