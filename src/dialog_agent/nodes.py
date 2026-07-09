"""图节点 —— 确定性骨架的入口/出口 + ReAct 内核（切片 5：渐进检索命中即停 + 对比分析）。

节点均为闭包工厂，绑定注入的 `Models` / 知识库检索器 / 数据库工具 / 联网工具，使
`build_graph(...)` 可用桩确定性驱动。切片 5 路径：

    rewrite（透传）
      → react_core（内核首步：需事实→调 plan_coverage 产覆盖度表；纯寒暄→无工具）
          ├─(无工具)→ chat_flow_answer → finalize（≤50 字截断）
          └─(plan_coverage)→ build_coverage（解析覆盖度表）
                → retrieve_knowledge（查知识库、更新覆盖状态、累积 Evidence）
                    → refine_after_knowledge（随观察补充新单元 / 修正数据源匹配；
                        remaining_units==0 即早退不调 LLM，保命中即停）
                        → retrieve_database（渐进检索第二层：仍缺失且候选含 DB 的单元经参数化
                            能力集查询，命中即停不穿透；ADR 0004）
                            → refine_after_database（同上精修；早退保命中即停）
                                → retrieve_external（渐进检索第三层：仍缺失且候选含联网的单元查
                                    联网；边界内置 ContentFilter 清洗/拦截，原始外部文本不进 State）
                                    → final_answer（remaining_units==0 强制作答，内部直接引用、
                                        外部带「据互联网公开信息」弱化提示且与内部数据分源隔离、
                                        数字取自 raw；隐性对比意图产出对比分析而非仅罗列）
                                        → finalize（知识流不截断）

精修步是有界的 plan-and-execute + ReAct 混合形态（ADR 0002）：结构层序与命中即停由图边保证，
「随观察补充新单元 / 修正数据源匹配」由精修步承接，补出的单元交由后续层自然拾取。盲区/降级
与自由 ReAct 循环 + 安全阀属后续切片沿同一覆盖度表接入。
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .chat_flow import CHAT_FLOW_SYSTEM_PROMPT, truncate_reply
from .coverage import (
    CoverageTable,
    citations_of,
    parse_plan_coverage,
    parse_refine_coverage,
    plan_coverage,
    refine_coverage,
)
from .database_tool import DatabaseTool, DatabaseToolError, query_capability
from .evidence import Evidence, SourceType
from .knowledge_tool import KnowledgeRetriever
from .models import Models
from .state import TurnState
from .web_search_tool import WebSearchTool

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
# 内外部隔离、来源标注、数字取自 raw、隐性对比产出对比分析 —— 由素材结构 + 本 prompt 自然导出
# （非事后质检，不设独立对比分类器，与「不设入口硬分类器」一致）。
FINAL_ANSWER_SYSTEM_PROMPT = (
    "你是「产教融合专家助理」。以下是已检索到的分源证据。请仅依据这些证据作答，不要臆造。"
    "内部知识库/数据库数据可直接引用，并在结论中标注其来源（文件/知识库/条款/表字段）。"
    "外部联网数据（EXTERNAL_SEARCH）必须与内部数据严格分源隔离呈现，"
    "并自动带「据互联网公开信息」的弱化提示，不得与内部权威数据混同为确证事实。"
    "涉及数字时必须取自证据的原始字段（raw），忠于原始数据、不得改写或估算。"
    "每个关键结论都要能追溯到给定证据；若证据不足以覆盖某信息单元，请坦诚说明边界，不要编造。"
    "当问题隐含对比意图（如「这两个专业就业率」「A 与 B 哪个更…」）时，不得仅罗列各实体数据，"
    "须给出对比分析与可操作的结论建议，使结论可直接用于决策。"
)

# 覆盖度表精修步的 system prompt：随观察补充新单元 / 修正数据源匹配（ADR 0002 动态特性）。
# 仅在仍存在缺失单元时触发；模型可调 refine_coverage 追加此前未识别的信息需求、或把某单元的
# 候选数据源改成更恰当者（如原判知识库、实为校内统计）。无需精修则不调用工具。
REFINE_COVERAGE_SYSTEM_PROMPT = (
    "你是「产教融合专家助理」的检索规划精修器。当前覆盖度表已检索若干层，仍有信息单元缺失。"
    "请依据已检索到的证据与仍缺失的单元判断：是否发现了此前未识别的信息需求（需追加为新单元），"
    "或某缺失单元的候选数据源匹配有误需修正（如原判走知识库、实为校内统计则改为数据库）。"
    "确有补充或修正时调用 refine_coverage 工具；无需精修则不要调用任何工具。"
    "追加的新单元须能被后续层（数据库/联网）覆盖方有意义。"
)

# 数据库检索层选能力的 system prompt：内核只选已注册能力名 + 填参数，不写 SQL（ADR 0004）。
# 参数强类型校验、只读、注入拦截在数据库工具层结构性保证；此处只负责把信息需求映射到能力调用。
DB_CAPABILITY_SYSTEM_PROMPT = (
    "你是「产教融合专家助理」的数据库查询编排器。"
    "针对给定的信息需求，从可用的业务查询能力中选择恰当的一个，调用 query_capability 工具，"
    "capability 取能力名，params 按该能力要求填写参数。"
    "SQL 由系统写死且只读，你只需选能力与填参数，绝不要提供 SQL。"
    "若没有任何能力匹配该需求，则不要调用工具。"
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


def make_refine_coverage_node(
    models: Models,
) -> Callable[[TurnState], dict[str, Any]]:
    """覆盖度表精修步（随观察补充新单元 / 修正数据源匹配，ADR 0002）。

    插在两层检索之间。命中即停时（`remaining_units` 为空）直接早退、不调 LLM，保住命中即停
    用例的模型调用计数；仍存在缺失单元时调强模型（绑 `refine_coverage` 工具），有工具调用则
    `add_unit` / `reassign_sources` 写回 coverage 并补一条 ToolMessage ack，无工具调用则幂等。
    任何解析异常吞掉、原 coverage 不变（局部降级，绝不阻断整轮）。
    """

    def refine_coverage_node(state: TurnState) -> dict[str, Any]:
        table = CoverageTable.from_dict(state.get("coverage"))
        if table.is_complete:
            # 命中即停：无缺失单元，精修无对象，不消耗模型调用。
            return {}

        evidence = state.get("evidence", [])
        prompt = _render_refine_prompt(table, evidence)
        llm = models.strong.bind_tools([refine_coverage])
        response = llm.invoke(
            [
                SystemMessage(content=REFINE_COVERAGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        call = _first_refine_call(response)
        if call is None:
            # 模型判定无需精修（未发起工具调用）——幂等，不动 coverage。
            return {"messages": [response]}

        try:
            new_units, reassigns = parse_refine_coverage(call["args"])
            for unit in new_units:
                table.add_unit(unit)
            for unit_id, sources in reassigns:
                table.reassign_sources(unit_id, sources)
        except (KeyError, ValueError, TypeError):
            # 脏输出降级：coverage 原样回写，不阻断整轮。
            return {"messages": [response]}

        ack = ToolMessage(
            content=(
                f"覆盖度表已精修：新增 {len(new_units)} 个单元、重匹配 "
                f"{len(reassigns)} 个单元。"
            ),
            tool_call_id=call.get("id", ""),
        )
        return {"coverage": table.as_dict(), "messages": [response, ack]}

    return refine_coverage_node


def make_retrieve_database_node(
    database_tool: DatabaseTool,
    models: Models,
) -> Callable[[TurnState], dict[str, Any]]:
    """数据库检索层（渐进检索的第二层，知识库之后）。

    命中即停：仅对覆盖度表中**仍缺失**且候选源含 INTERNAL_DATABASE 的单元查数据库；若知识库
    层已覆盖全部单元，则此层无剩余 DB 单元、直接早退（不空调模型、不向下穿透）。

    每个待覆盖单元用强模型经 `query_capability` 选能力 + 填参（不写 SQL，ADR 0004），交
    `DatabaseTool` 强类型校验后只读执行；命中则标记已覆盖并累积 `INTERNAL_DATABASE` Evidence
    （raw 保留原始行供数字溯源）。局部降级：非法入参/无匹配能力/后端异常时放弃该单元该源，
    loop 继续（切片 3 只有知识库+数据库两层，未覆盖单元留待结论坦诚告知边界）。
    """

    def retrieve_database(state: TurnState) -> dict[str, Any]:
        table = CoverageTable.from_dict(state.get("coverage"))
        pending = table.remaining_for_source(SourceType.INTERNAL_DATABASE)
        if not pending:
            # 命中即停：无待覆盖的数据库单元，不触发模型调用。
            return {}

        llm = models.strong.bind_tools([query_capability])
        capabilities_desc = database_tool.describe_capabilities()
        new_evidence: list[Evidence] = []
        for unit in pending:
            hits = _query_one_unit(llm, database_tool, capabilities_desc, unit.need)
            if hits:
                new_evidence.extend(hits)
                table.mark_covered(unit.id, citations_of(hits))

        result: dict[str, Any] = {"coverage": table.as_dict()}
        if new_evidence:
            result["evidence"] = [ev.as_dict() for ev in new_evidence]
        return result

    return retrieve_database


def make_retrieve_external_node(
    web_tool: WebSearchTool,
) -> Callable[[TurnState], dict[str, Any]]:
    """联网检索层（渐进检索的第三层、最末层，数据库之后）。

    命中即停：仅对覆盖度表中**仍缺失**且候选源含 EXTERNAL_SEARCH 的单元查联网；若更高层
    （知识库/数据库）已覆盖全部单元，则此层无剩余外部单元、直接早退（不空调、不向下穿透）。

    外部内容在工具边界已被 `WebSearchTool` 内置的 `ContentFilter` 清洗/拦截，进 State 的
    Evidence.content 与 raw 均为脱敏副本，**原始外部文本绝不喂 LLM**（ADR 0007 数据层职责）。
    命中则标记已覆盖并累积 `EXTERNAL_SEARCH` Evidence（raw 保留清洗后字段供数字溯源）。
    未命中的单元保持缺失，由结论生成坦诚告知边界（盲区）。
    """

    def retrieve_external(state: TurnState) -> dict[str, Any]:
        table = CoverageTable.from_dict(state.get("coverage"))
        pending = table.remaining_for_source(SourceType.EXTERNAL_SEARCH)
        if not pending:
            # 命中即停：无待覆盖的外部单元，不触发联网检索。
            return {}

        new_evidence: list[Evidence] = []
        for unit in pending:
            hits = web_tool.search(unit.need)
            if hits:
                new_evidence.extend(hits)
                table.mark_covered(unit.id, citations_of(hits))

        result: dict[str, Any] = {"coverage": table.as_dict()}
        if new_evidence:
            result["evidence"] = [ev.as_dict() for ev in new_evidence]
        return result

    return retrieve_external


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


def _query_one_unit(
    llm: Any, database_tool: DatabaseTool, capabilities_desc: str, need: str
) -> list[Evidence]:
    """为单个信息需求选一次能力并执行，返回该单元命中的 Evidence（失败降级为空列表）。

    内核经 `query_capability` 只输出 {capability, params}；工具层做强类型校验与只读执行。
    任何拒绝路径（无工具调用 / 非法入参 / 无匹配能力）都吞成空列表——该单元该源未覆盖，
    交由后续层或结论坦诚告知边界，绝不外泄技术细节。
    """
    prompt = f"信息需求：{need}\n\n可用查询能力：\n{capabilities_desc}"
    response = llm.invoke(
        [
            SystemMessage(content=DB_CAPABILITY_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    call = _first_capability_call(response)
    if call is None:
        return []  # 内核判定无能力匹配（未发起工具调用）。
    try:
        return database_tool.run(call["capability"], call["params"])
    except DatabaseToolError:
        # 局部降级：非法入参/未知能力/注入被拒——放弃该单元该源，不阻断整轮。
        return []


def _first_capability_call(message: Any) -> dict[str, Any] | None:
    """取出消息里的首个 query_capability 工具调用（规范化）；无则返回 None。"""
    for call in getattr(message, "tool_calls", None) or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name == query_capability.__name__:
            args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
            args = args or {}
            return {
                "capability": str(args.get("capability", "")),
                "params": args.get("params", {}) or {},
            }
    return None


def _first_plan_call(message: Any) -> dict[str, Any]:
    """取出消息里的首个 plan_coverage 工具调用（规范化为 dict）。"""
    for call in getattr(message, "tool_calls", None) or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name == plan_coverage.__name__:
            args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
            cid = call.get("id", "") if isinstance(call, dict) else getattr(call, "id", "")
            return {"name": name, "args": args or {}, "id": cid or ""}
    raise ValueError("内核消息中未找到 plan_coverage 工具调用")


def _first_refine_call(message: Any) -> dict[str, Any] | None:
    """取出消息里的首个 refine_coverage 工具调用（规范化为 dict）；无则返回 None。"""
    for call in getattr(message, "tool_calls", None) or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name == refine_coverage.__name__:
            args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
            cid = call.get("id", "") if isinstance(call, dict) else getattr(call, "id", "")
            return {"name": name, "args": args or {}, "id": cid or ""}
    return None


def _render_evidence_prompt(
    query: str, evidence: list[dict[str, Any]], table: CoverageTable
) -> str:
    """把自包含查询 + 分源 Evidence + 覆盖状态渲染成结论生成的输入。

    Evidence 按来源标注逐条列出（content + citation），使模型作答时能直接引用并标注来源；
    外部证据显式标记为「须注明据互联网公开信息、与内部数据分源隔离」，并附清洗后的原始字段
    供数字溯源；数据库证据同样附原始字段强化数字溯源；仍缺失的单元列为盲区，导出坦诚告知边界。
    隐性对比意图由素材结构（多实体同类数据）+ 提示导出对比分析，不设独立分类器。
    """
    lines = [f"用户问题：{query}", "", "已检索证据："]
    if evidence:
        for ev in evidence:
            line = f"- [{ev['source_type']}] {ev['content']}（来源：{ev['citation']}）"
            if ev["source_type"] == SourceType.EXTERNAL_SEARCH.value:
                # 外部数据：弱化提示 + 内外部隔离 + 数字取自 raw（已脱敏）。
                line += "｜外部数据：结论中须注明「据互联网公开信息」，与内部数据分源隔离呈现"
                if ev.get("raw"):
                    line += f"｜原始字段：{ev['raw']}"
                lines.append(line)
            elif ev["source_type"] == SourceType.INTERNAL_DATABASE.value and ev.get("raw"):
                # 数据库证据附原始字段，强化数字溯源：结论里的数字须取自 raw，不得改写或估算。
                lines.append(f"{line}｜原始字段：{ev['raw']}")
            else:
                lines.append(line)
    else:
        lines.append("（暂无证据）")

    remaining = table.remaining_units
    if remaining:
        lines.append("")
        lines.append("仍未覆盖的信息单元（如无其他来源请坦诚说明边界）：")
        lines.extend(f"- {u.need}" for u in remaining)

    # 隐性对比提示：当证据涉及多个可比实体（如两个专业各自的同类数据）时，引导产出对比分析
    # 而非并列罗列。判定为「多实体同类」即提示；对比意图的最终判定由模型依 query + 素材完成。
    if _looks_comparative(query, evidence):
        lines.append("")
        lines.append("该问题隐含对比意图：结论须给出对比分析与决策建议，不得仅并列罗列各实体数据。")
    return "\n".join(lines)


def _looks_comparative(query: str, evidence: list[dict[str, Any]]) -> bool:
    """粗筛对比意图：query 含对比词，或同源证据含多个可比实体。

    只作「是否提示对比」的廉价启发式，不替代模型判断；命中即给提示，漏判不损正确性。
    """
    comparative_hints = ("对比", "哪个", "哪个更", "这两个", "这两个专业", "比较", "谁更", "相差")
    if any(hint in query for hint in comparative_hints):
        return True
    # 同一数据源 ≥2 条证据，且其 content 含可区分实体（仅作粗筛：同源多条即提示）。
    by_source: dict[str, int] = {}
    for ev in evidence:
        by_source[ev.get("source_type", "")] = by_source.get(ev.get("source_type", ""), 0) + 1
    return any(count >= 2 for count in by_source.values())


def _render_refine_prompt(table: CoverageTable, evidence: list[dict[str, Any]]) -> str:
    """把仍缺失的单元 + 已检索证据摘要渲染成精修步的输入。

    仅列出仍缺失单元（精修对象）与已有证据摘要（供模型判断是否需补单元 / 改匹配），不重复
    已覆盖单元。模型据此决定调用 refine_coverage 或不调。
    """
    lines = ["当前仍缺失的信息单元（精修对象）："]
    for unit in table.remaining_units:
        lines.append(f"- {unit.id}：{unit.need}｜候选源：{unit.source_matches}")
    lines.append("")
    lines.append("已检索证据摘要：")
    if evidence:
        for ev in evidence:
            lines.append(f"- [{ev['source_type']}] {ev['content']}（来源：{ev['citation']}）")
    else:
        lines.append("（暂无证据）")
    return "\n".join(lines)
