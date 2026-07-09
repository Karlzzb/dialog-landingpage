"""Evidence 契约 —— 三工具统一的结构化返回（ADR 0007）。

结论生成要做到内外部隔离、来源标注、数字忠于原始数据，前提是内核拿到的检索结果结构化、
带来源、可回溯。三个检索工具（知识库/数据库/联网）都返回同一 `Evidence`：

    { source_type, content, citation, raw }

- source_type：证据来源分类，决定结论生成时的隔离与弱化提示。
- content：可直接进结论的文本片段/字段值。
- citation：文件名/条款号/URL/表字段等可回溯标注。
- raw：原始 JSON，供数字溯源（数据库工具尤其重要）。

切片 2 仅落地 `INTERNAL_KNOWLEDGE` 一路；数据库/联网在后续切片沿用同一契约。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    """证据来源分类（CONTEXT.md「数据源」）。

    继承 str 使其可直接进 JSON / 日志 / prompt，值即领域术语里的稳定标识。
    """

    INTERNAL_KNOWLEDGE = "INTERNAL_KNOWLEDGE"  # 知识库：规章制度/标准文件/红头文件
    INTERNAL_DATABASE = "INTERNAL_DATABASE"  # 数据库：校内统计（对口率/考勤率/薪资…）
    EXTERNAL_SEARCH = "EXTERNAL_SEARCH"  # 联网：校内没有的外部信息（工商/财报/新闻…）


@dataclass(frozen=True)
class Evidence:
    """一条结构化证据。三工具统一返回，结论生成据此天然做隔离/标注/溯源。

    不可变（frozen）：证据一旦从工具边界产出即为事实快照，内核只读不改。
    """

    source_type: SourceType
    content: str
    citation: str
    # 原始 JSON（数字溯源）；知识库为原始 chunk，数据库为查询行，联网为原文条目。
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """序列化为纯 dict（存入 State / 断言 / trace）。source_type 取其字符串值。"""
        return {
            "source_type": self.source_type.value,
            "content": self.content,
            "citation": self.citation,
            "raw": dict(self.raw),
        }
