"""结构化会话记忆 + 自包含查询改写（ADR 0003）。

会话记忆替代全量对话历史，追求 context 最小：只维护用户画像 + 滚动的「关键实体/约束
摘要」（如 地域=武汉市、主题=校企合作），跨轮持久、每轮经改写步更新。改写步读会话记忆、
把用户输入补全成自包含查询（消解指代/补全省略），**只做指代消解、不做流程分叉**，因此与
ADR 0001「不设入口硬分类器」不冲突。

本模块提供：
- `SessionMemory` —— State 内的结构化会话记忆对象及其序列化。
- `rewrite_query` 结构化 schema —— 绑定快模型，改写步以工具调用同时产出「自包含查询」
  与「本轮合并后的实体摘要」（一次快模型调用既完成改写又完成记忆更新）。
- `parse_rewrite_call` —— 把快模型的工具调用参数解析成 (query, entities)，脏输出降级。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class SessionMemory:
    """跨轮持久的结构化会话记忆。

    user_profile：用户画像（姓名/角色/组织），稀疏维护，可推断时填、不可推断时空。
    entities：滚动的「关键实体/约束摘要」，每轮经改写步合并更新（如
        {"地域": "武汉市", "主题": "校企合作"}）。值为本轮结束时应有的完整摘要，非增量。
    """

    user_profile: dict[str, Any] = field(default_factory=dict)
    entities: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_profile": dict(self.user_profile),
            "entities": dict(self.entities),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SessionMemory":
        if not data:
            return cls()
        return cls(
            user_profile=dict(data.get("user_profile") or {}),
            entities=dict(data.get("entities") or {}),
        )

    def merged_with_entities(self, entities: dict[str, Any]) -> "SessionMemory":
        """返回用新实体摘要合并后的记忆（新值覆盖同名键，保留其余历史键）。

        改写步产出的 entities 视为本轮合并后的完整摘要；但模型可能只输出本轮变化的键，
        故按「新值覆盖、旧值保留」合并，保证历史实体不因模型漏输出而丢失（稳健降级）。
        """
        merged = dict(self.entities)
        merged.update({k: v for k, v in entities.items() if v not in (None, "")})
        return SessionMemory(user_profile=dict(self.user_profile), entities=merged)


# ── 改写步的结构化工具 schema（绑定快模型）──


class rewrite_query(BaseModel):  # noqa: N801 —— 类名即工具名，须与领域动作一致
    """把用户输入补全成自包含查询，并产出本轮合并后的关键实体/约束摘要。

    改写只做指代消解/省略补全，不做流程分叉：不判断该走对话流还是知识流，也不改变意图，
    仅让查询不依赖上下文即可被理解。输入已自包含时原样返回。混合寒暄+业务时保留自然承接
    前缀、补全业务诉求。entities 为本轮结束时应有的完整摘要（合并历史与新输入）。
    """

    self_contained_query: str = Field(
        description=(
            "补全指代/省略后的自包含查询，不依赖上下文即可理解。已自包含则原样返回。"
            "混合寒暄+业务时保留自然承接前缀再补全业务诉求。"
        )
    )
    entities: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "本轮结束时应有的关键实体/约束摘要（合并历史与新输入），如 "
            '{"地域": "武汉市", "主题": "校企合作"}。无新实体可更新时返回历史摘要或空。'
        ),
    )


def parse_rewrite_call(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """把快模型 `rewrite_query` 工具调用的参数解析成 (自包含查询, 实体摘要)。

    脏输出降级：缺 self_contained_query 视为空串、entities 非 dict 视为空，不阻断整轮。
    """
    query = str(args.get("self_contained_query", "") or "")
    raw_entities = args.get("entities") or {}
    if not isinstance(raw_entities, dict):
        raw_entities = {}
    entities = {str(k): v for k, v in raw_entities.items() if v not in (None, "")}
    return query, entities
