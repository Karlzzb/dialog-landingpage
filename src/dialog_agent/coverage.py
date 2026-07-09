"""覆盖度表 —— 内核首步强制产出的结构化规划对象（ADR 0002）。

覆盖度表把用户的自包含查询拆成《信息单元》清单，为每个单元匹配候选数据源，并跟踪其覆盖
状态。内核按 知识库 → 数据库 → 联网 逐层查询，每层查完更新覆盖状态；`remaining_units`
归零即命中即停、强制作答。

它是 T11-T13（信息单元拆解 / 单元→数据源匹配 / 覆盖度评估）的确定性载体：存在于 State
即可被断言，而非事后从文本里检。

本模块提供：
- `plan_coverage` 结构化 schema —— 绑定强模型，内核首步以工具调用产出覆盖度表。
- `CoverageTable` / `InformationUnit` —— State 内的结构化对象及其更新语义。
- `parse_plan_coverage` —— 把内核的工具调用参数解析成 `CoverageTable`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .evidence import Evidence, SourceType

# 数据源的逐层查询优先级（知识库 → 数据库 → 联网）。命中即停据此顺序。
SOURCE_PRIORITY: tuple[SourceType, ...] = (
    SourceType.INTERNAL_KNOWLEDGE,
    SourceType.INTERNAL_DATABASE,
    SourceType.EXTERNAL_SEARCH,
)


class CoverageStatus(str, Enum):
    """单元覆盖状态。切片 2 用 REMAINING / COVERED 两态；盲区/降级态在后续切片引入。"""

    REMAINING = "REMAINING"  # 仍缺失，需继续查询
    COVERED = "COVERED"  # 已被某数据源命中覆盖


# ── 内核首步产出覆盖度表的结构化工具 schema（绑定强模型）──


class PlanCoverageUnit(BaseModel):
    """规划阶段的单个信息单元（LLM 输出结构）。"""

    id: str = Field(description="信息单元的稳定标识，如 u1")
    need: str = Field(description="该单元对应的、自包含的具体信息需求")
    sources: list[str] = Field(
        description=(
            "该单元的候选数据源，取值 INTERNAL_KNOWLEDGE / INTERNAL_DATABASE / "
            "EXTERNAL_SEARCH，按可信优先级排列"
        )
    )


class plan_coverage(BaseModel):  # noqa: N801 —— 类名即工具名，须与领域动作一致
    """产出《覆盖度表》：把用户查询拆成信息单元，为每个单元匹配候选数据源。

    需要检索客观事实时，内核首步必须调用本工具产出结构化规划；纯寒暄则不调用。
    """

    units: list[PlanCoverageUnit] = Field(description="信息单元清单（至少一个）")


@dataclass
class InformationUnit:
    """覆盖度表里的一个信息单元。"""

    id: str
    need: str
    # 候选数据源（按优先级）；决定该单元在哪一层被尝试覆盖。
    source_matches: list[SourceType]
    status: CoverageStatus = CoverageStatus.REMAINING
    # 命中来源的可回溯标注（文件名/知识库/chunk_index 等），供结论标注。
    citations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "need": self.need,
            "source_matches": [s.value for s in self.source_matches],
            "status": self.status.value,
            "citations": list(self.citations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InformationUnit":
        return cls(
            id=data["id"],
            need=data["need"],
            source_matches=[SourceType(s) for s in data.get("source_matches", [])],
            status=CoverageStatus(data.get("status", CoverageStatus.REMAINING.value)),
            citations=list(data.get("citations", [])),
        )


@dataclass
class CoverageTable:
    """覆盖度表：信息单元清单 + 覆盖状态跟踪。

    以 dict 形式存入 State（`from_state` / `as_dict` 往返），节点读出、更新、写回。
    """

    units: list[InformationUnit] = field(default_factory=list)

    # ── 覆盖度查询 ──

    @property
    def remaining_units(self) -> list[InformationUnit]:
        """仍缺失（未覆盖）的信息单元。归零即命中即停。"""
        return [u for u in self.units if u.status == CoverageStatus.REMAINING]

    @property
    def is_complete(self) -> bool:
        """全部信息单元均已覆盖。"""
        return not self.remaining_units

    def remaining_for_source(self, source: SourceType) -> list[InformationUnit]:
        """当前层可尝试的单元：仍缺失且候选源包含该数据源。"""
        return [u for u in self.remaining_units if source in u.source_matches]

    # ── 覆盖度更新 ──

    def mark_covered(self, unit_id: str, citations: list[str]) -> None:
        """把某单元标记为已覆盖，并记录其来源标注。"""
        for unit in self.units:
            if unit.id == unit_id:
                unit.status = CoverageStatus.COVERED
                unit.citations = list(citations)
                return
        raise KeyError(f"覆盖度表中不存在信息单元：{unit_id}")

    # ── 序列化 ──

    def as_dict(self) -> dict[str, Any]:
        return {"units": [u.as_dict() for u in self.units]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CoverageTable":
        if not data:
            return cls(units=[])
        return cls(units=[InformationUnit.from_dict(u) for u in data.get("units", [])])


def parse_plan_coverage(args: dict[str, Any]) -> CoverageTable:
    """把内核 `plan_coverage` 工具调用的参数解析成覆盖度表。

    未知数据源标识被忽略（不炸），保证内核偶发的脏输出不阻断主体。
    """
    units: list[InformationUnit] = []
    for raw in args.get("units", []):
        source_matches: list[SourceType] = []
        for s in raw.get("sources", []):
            try:
                source_matches.append(SourceType(s))
            except ValueError:
                continue  # 忽略无法识别的数据源标识
        units.append(
            InformationUnit(
                id=str(raw["id"]),
                need=str(raw["need"]),
                source_matches=source_matches,
            )
        )
    return CoverageTable(units=units)


def citations_of(evidence: list[Evidence]) -> list[str]:
    """从一组 Evidence 抽取可回溯标注列表（去空、保序去重）。"""
    seen: set[str] = set()
    result: list[str] = []
    for ev in evidence:
        c = (ev.citation or "").strip()
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result
