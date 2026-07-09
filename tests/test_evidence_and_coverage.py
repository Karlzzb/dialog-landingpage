"""第二接缝 —— Evidence 契约 / 覆盖度表 / 知识库适配层的纯函数式单元测试。

脱离图独立测：Evidence 契约规整、覆盖度表拆解与更新语义、假检索器的命中/未命中行为。
"""

from __future__ import annotations

from dialog_agent.coverage import (
    CoverageStatus,
    CoverageTable,
    InformationUnit,
    citations_of,
    parse_plan_coverage,
)
from dialog_agent.evidence import Evidence, SourceType
from dialog_agent.knowledge_tool import FakeKnowledgeRetriever, to_evidence


# ── Evidence 契约 ──


def test_evidence_as_dict_roundtrip():
    ev = Evidence(
        source_type=SourceType.INTERNAL_KNOWLEDGE,
        content="片段",
        citation="库/文件#chunk1",
        raw={"chunk_index": 1},
    )
    d = ev.as_dict()
    assert d == {
        "source_type": "INTERNAL_KNOWLEDGE",
        "content": "片段",
        "citation": "库/文件#chunk1",
        "raw": {"chunk_index": 1},
    }


# ── 知识库适配层：chunk → Evidence 规整 ──


def test_to_evidence_builds_traceable_citation():
    ev = to_evidence(
        {
            "content": "正文",
            "file_name": "标准.pdf",
            "knowledge_name": "标准库",
            "chunk_index": 3,
        }
    )
    assert ev.source_type == SourceType.INTERNAL_KNOWLEDGE
    assert ev.citation == "标准库/标准.pdf#chunk3"
    assert ev.raw["chunk_index"] == 3


def test_fake_retriever_hit_and_miss():
    retriever = FakeKnowledgeRetriever(
        {"对口率": [{"content": "对口率 85%", "file_name": "统计.pdf", "chunk_index": 0}]}
    )
    hit = retriever.retrieve("某专业对口率")
    assert len(hit) == 1
    assert hit[0].source_type == SourceType.INTERNAL_KNOWLEDGE

    miss = retriever.retrieve("无关查询")
    assert miss == []
    # 调用被记录，供图行为断言。
    assert len(retriever.calls) == 2


# ── 覆盖度表：拆解、查询、更新 ──


def test_parse_plan_coverage_maps_sources():
    table = parse_plan_coverage(
        {
            "units": [
                {"id": "u1", "need": "补贴标准", "sources": ["INTERNAL_KNOWLEDGE", "EXTERNAL_SEARCH"]},
                {"id": "u2", "need": "对口率", "sources": ["INTERNAL_DATABASE"]},
            ]
        }
    )
    assert len(table.units) == 2
    assert table.units[0].source_matches == [
        SourceType.INTERNAL_KNOWLEDGE,
        SourceType.EXTERNAL_SEARCH,
    ]
    assert len(table.remaining_units) == 2


def test_parse_plan_coverage_ignores_unknown_source():
    """脏输出容错：无法识别的数据源标识被忽略，不炸。"""
    table = parse_plan_coverage(
        {"units": [{"id": "u1", "need": "x", "sources": ["INTERNAL_KNOWLEDGE", "BOGUS"]}]}
    )
    assert table.units[0].source_matches == [SourceType.INTERNAL_KNOWLEDGE]


def test_remaining_for_source_filters_by_layer():
    table = parse_plan_coverage(
        {
            "units": [
                {"id": "u1", "need": "a", "sources": ["INTERNAL_KNOWLEDGE"]},
                {"id": "u2", "need": "b", "sources": ["INTERNAL_DATABASE"]},
            ]
        }
    )
    kb_units = table.remaining_for_source(SourceType.INTERNAL_KNOWLEDGE)
    assert [u.id for u in kb_units] == ["u1"]


def test_mark_covered_updates_status_and_completion():
    table = parse_plan_coverage(
        {"units": [{"id": "u1", "need": "a", "sources": ["INTERNAL_KNOWLEDGE"]}]}
    )
    assert not table.is_complete
    table.mark_covered("u1", ["库/文件#chunk1"])
    assert table.units[0].status == CoverageStatus.COVERED
    assert table.units[0].citations == ["库/文件#chunk1"]
    assert table.is_complete
    assert table.remaining_units == []


def test_coverage_table_dict_roundtrip():
    table = parse_plan_coverage(
        {"units": [{"id": "u1", "need": "a", "sources": ["INTERNAL_KNOWLEDGE"]}]}
    )
    table.mark_covered("u1", ["c1"])
    restored = CoverageTable.from_dict(table.as_dict())
    assert restored.units[0].status == CoverageStatus.COVERED
    assert restored.units[0].citations == ["c1"]
    assert restored.units[0].source_matches == [SourceType.INTERNAL_KNOWLEDGE]


def test_citations_of_dedupes_and_drops_empty():
    evidence = [
        Evidence(SourceType.INTERNAL_KNOWLEDGE, "a", "c1"),
        Evidence(SourceType.INTERNAL_KNOWLEDGE, "b", "c1"),  # 重复
        Evidence(SourceType.INTERNAL_KNOWLEDGE, "c", ""),  # 空
        Evidence(SourceType.INTERNAL_KNOWLEDGE, "d", "c2"),
    ]
    assert citations_of(evidence) == ["c1", "c2"]


# ── 覆盖度表动态精修：补充新单元 / 修正数据源匹配（ADR 0002，切片 5）──


def test_add_unit_appends_new_unit():
    table = parse_plan_coverage(
        {"units": [{"id": "u1", "need": "a", "sources": ["INTERNAL_KNOWLEDGE"]}]}
    )
    table.add_unit(
        InformationUnit(id="u2", need="b", source_matches=[SourceType.INTERNAL_DATABASE])
    )
    assert [u.id for u in table.units] == ["u1", "u2"]
    assert table.units[1].status == CoverageStatus.REMAINING


def test_add_unit_is_idempotent_on_duplicate_id():
    """已存在的 id 追加被忽略，不重复、不覆盖原单元。"""
    table = parse_plan_coverage(
        {"units": [{"id": "u1", "need": "原", "sources": ["INTERNAL_KNOWLEDGE"]}]}
    )
    table.add_unit(
        InformationUnit(id="u1", need="新", source_matches=[SourceType.EXTERNAL_SEARCH])
    )
    assert len(table.units) == 1
    assert table.units[0].need == "原"


def test_reassign_sources_replaces_matches():
    table = parse_plan_coverage(
        {"units": [{"id": "u1", "need": "a", "sources": ["INTERNAL_KNOWLEDGE"]}]}
    )
    table.reassign_sources("u1", [SourceType.INTERNAL_DATABASE, SourceType.EXTERNAL_SEARCH])
    assert table.units[0].source_matches == [
        SourceType.INTERNAL_DATABASE,
        SourceType.EXTERNAL_SEARCH,
    ]
    # 修正后该单元可被数据库层拾取。
    assert table.remaining_for_source(SourceType.INTERNAL_DATABASE) == [table.units[0]]


def test_reassign_sources_unknown_unit_raises():
    import pytest

    table = parse_plan_coverage(
        {"units": [{"id": "u1", "need": "a", "sources": ["INTERNAL_KNOWLEDGE"]}]}
    )
    with pytest.raises(KeyError):
        table.reassign_sources("nope", [SourceType.INTERNAL_DATABASE])


def test_parse_refine_coverage_parses_add_and_reassign():
    from dialog_agent.coverage import parse_refine_coverage

    new_units, reassigns = parse_refine_coverage(
        {
            "add_units": [
                {"id": "u2", "need": "b", "sources": ["INTERNAL_DATABASE", "BOGUS"]}
            ],
            "reassign": [
                {"id": "u1", "sources": ["EXTERNAL_SEARCH"]},
                {"id": "u3", "sources": ["BOGUS"]},  # 全部未知 → 视为不修正，丢弃
            ],
        }
    )
    assert len(new_units) == 1
    assert new_units[0].source_matches == [SourceType.INTERNAL_DATABASE]
    assert reassigns == [("u1", [SourceType.EXTERNAL_SEARCH])]
