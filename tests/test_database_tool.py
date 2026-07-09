"""第二接缝 —— 参数化查询能力集（数据库工具）的纯函数式单元测试（ADR 0004）。

脱离图独立测：给定 `{capability, params}` 断言只读、参数强类型校验、注入型输入被拒、自由 SQL
无入口、Evidence 溯源（raw 保留原始行）。安全属性是数据层的结构性性质，在此层验证。
"""

from __future__ import annotations

import pytest

from dialog_agent.database_tool import (
    INTERNSHIP_PLACEMENT_RATE,
    DatabaseTool,
    DatabaseToolError,
    FakeDatabaseBackend,
    ParamSpec,
    ParamType,
    QueryCapability,
    build_default_database_tool,
)
from dialog_agent.evidence import SourceType

# 内置能力 internship_placement_rate 的假数据：某学院某专业一行对口率统计。
ROWS = {
    "internship_placement_rate": [
        {
            "college": "机电学院",
            "major": "数控技术",
            "rate": 87.5,
            "sample_size": 120,
            "term": "2025-秋",
        }
    ]
}


def _tool() -> DatabaseTool:
    return build_default_database_tool(FakeDatabaseBackend(ROWS))


# ── 端到端能力调用：{capability, params} → INTERNAL_DATABASE Evidence，raw 溯源 ──


def test_capability_returns_internal_database_evidence_with_raw():
    tool = _tool()
    evidence = tool.run(
        "internship_placement_rate", {"college": "机电学院", "major": "数控技术"}
    )
    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.source_type == SourceType.INTERNAL_DATABASE
    # content 含渲染后的对口率；citation 可回溯到表与维度。
    assert "87.5%" in ev.content
    assert "internship_stats" in ev.citation
    # 数字溯源：raw 保留原始行，关键数字取自 raw 字段而非文本。
    assert ev.raw["rate"] == 87.5
    assert ev.raw["sample_size"] == 120


def test_backend_receives_parameterized_sql_not_interpolated():
    """后端收到的是人写死 SQL + 参数字典（参数化），SQL 内不含用户值拼接。"""
    backend = FakeDatabaseBackend(ROWS)
    tool = build_default_database_tool(backend)
    tool.run("internship_placement_rate", {"college": "机电学院", "major": "数控技术"})

    call = backend.calls[-1]
    assert ":college" in call["sql"] and ":major" in call["sql"]
    # 用户值只经参数字典传递，不出现在 SQL 文本里。
    assert "机电学院" not in call["sql"]
    assert call["params"]["college"] == "机电学院"


# ── 只读性：模板 SQL 必须只读，写/DDL 语句注册期即被拒 ──


def test_write_sql_rejected_at_registration():
    with pytest.raises(DatabaseToolError):
        QueryCapability(
            name="bad_write",
            description="非法写操作",
            sql="UPDATE internship_stats SET rate = 100",
            params=(),
            content_template="x",
            citation_template="y",
        )


def test_select_with_embedded_ddl_rejected():
    """即便以 SELECT 开头，夹带写/DDL 关键字也被拒（对人写模板的防呆）。"""
    with pytest.raises(DatabaseToolError):
        QueryCapability(
            name="sneaky",
            description="夹带 DDL",
            sql="SELECT * FROM t; DROP TABLE t",
            params=(),
            content_template="x",
            citation_template="y",
        )


def test_cte_select_is_allowed():
    """WITH（CTE）开头的只读查询允许注册。"""
    cap = QueryCapability(
        name="cte_ok",
        description="CTE 只读",
        sql="WITH s AS (SELECT 1 AS n) SELECT n FROM s",
        params=(),
        content_template="{n}",
        citation_template="c",
    )
    assert cap.name == "cte_ok"


# ── 自由 SQL 无入口：工具只认已注册能力名 ──


def test_free_sql_has_no_entry_unknown_capability_rejected():
    tool = _tool()
    # 试图把自由 SQL 当能力名传入——被拒，因为它不是注册能力。
    with pytest.raises(DatabaseToolError):
        tool.run("SELECT * FROM internship_stats", {})
    with pytest.raises(DatabaseToolError):
        tool.run("not_a_capability", {"college": "x", "major": "y"})


# ── 参数强类型校验 ──


def test_missing_required_param_rejected():
    tool = _tool()
    with pytest.raises(DatabaseToolError):
        tool.run("internship_placement_rate", {"college": "机电学院"})


def test_unknown_param_rejected():
    tool = _tool()
    with pytest.raises(DatabaseToolError):
        tool.run(
            "internship_placement_rate",
            {"college": "机电学院", "major": "数控技术", "evil": "1"},
        )


def test_integer_type_enforced():
    spec = ParamSpec(name="year", type=ParamType.INTEGER)
    assert spec.validate("2025") == 2025
    assert spec.validate(2025) == 2025
    with pytest.raises(DatabaseToolError):
        spec.validate("二零二五")
    # bool 不被当作整数。
    with pytest.raises(DatabaseToolError):
        spec.validate(True)


def test_choices_whitelist_enforced():
    spec = ParamSpec(name="scope", type=ParamType.STRING, choices=("校级", "院级"))
    assert spec.validate("校级") == "校级"
    with pytest.raises(DatabaseToolError):
        spec.validate("班级")


def test_overlong_string_rejected():
    spec = ParamSpec(name="x", type=ParamType.STRING, max_length=4)
    with pytest.raises(DatabaseToolError):
        spec.validate("超过四个字的输入")


# ── 注入型输入被拒（含 SQL 元字符的字符串参数）──


@pytest.mark.parametrize(
    "payload",
    [
        "机电学院'; DROP TABLE internship_stats; --",
        "' OR '1'='1",
        "数控技术) UNION SELECT * FROM users",
        "x; DELETE FROM t",
    ],
)
def test_injection_payloads_rejected(payload):
    tool = _tool()
    with pytest.raises(DatabaseToolError):
        tool.run("internship_placement_rate", {"college": payload, "major": "数控技术"})


def test_empty_string_param_rejected():
    tool = _tool()
    with pytest.raises(DatabaseToolError):
        tool.run("internship_placement_rate", {"college": "  ", "major": "数控技术"})


# ── 能力清单自描述（供内核选能力）──


def test_describe_capabilities_lists_registered():
    tool = _tool()
    desc = tool.describe_capabilities()
    assert "internship_placement_rate" in desc
    assert "college" in desc and "major" in desc


def test_default_tool_registers_builtin_capability():
    tool = build_default_database_tool()
    assert INTERNSHIP_PLACEMENT_RATE.name in tool.capability_names
