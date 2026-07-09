"""数据库工具 —— 参数化查询能力集（ADR 0004），非 Text2SQL。

数据库查询压着多条确定性红线（注入防护、账号只读、数字反向溯源）。开放式 Text2SQL 注入面
最大、越权最难防，这些红线几乎不可能 100% 保证。故本工具把数据库能力做成**参数化查询
能力集（语义层）**：

- 预定义一组业务查询能力，每个是带参数、只读的安全查询；SQL 由人写死。
- LLM 的工作退化为 `{capability, params}` 的结构化工具调用——只选能力、填参数，不写 SQL。
- 注入、只读、参数强类型校验在模板层做结构性根治（权限注入层暂缓，见 ADR 0005）。

新增一类业务查询 = 新增一个 `QueryCapability` 模板，扩展可控可维护（ADR 0004）。

分层（与知识库工具同构的可插拔接缝）：
- `DatabaseBackend` 协议：只读查询执行器的稳定接口（人写死 SQL + 参数 → 结果行）。
- `FakeDatabaseBackend`：假实现，可编排返回，供图桩化与离线联调（真实只读库到位后替换）。
- `QueryCapability`：一条能力模板（人写死只读 SQL + 参数规格 + 行→Evidence 规整）。
- `DatabaseTool`：能力注册表，`run(capability, params)` 校验后执行，返回 `INTERNAL_DATABASE`
  的 Evidence（`raw` 保留原始行供数字溯源）。
- `query_capability`：绑定强模型的结构化工具 schema（LLM 只输出 capability + params）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .evidence import Evidence, SourceType

# 只读白名单：能力模板注册时校验其 SQL 以此开头（大小写不敏感、跳过前导空白）。
# 任何非 SELECT / WITH（CTE 最终仍是 SELECT）语句在注册期即被拒，杜绝写操作进入能力集。
_READ_ONLY_PREFIXES = ("select", "with")

# 危险 SQL 关键字：即便前缀是 SELECT，也不允许模板 SQL 里夹带写/DDL 语句（如子查询后接分号
# 再跟 UPDATE）。这是对人写模板的注册期防呆，不是对 LLM 输入的防护（LLM 碰不到 SQL）。
_FORBIDDEN_SQL = re.compile(
    r"(?i)\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|merge|replace|"
    r"attach|pragma|commit|rollback)\b"
)


class DatabaseToolError(Exception):
    """数据库工具的用户输入侧错误（未知能力、参数非法/缺失/注入）。

    与「后端执行异常」区分：本类表示 `{capability, params}` 不合法，属可预期的拒绝路径，
    由内核降级处理（标记该源不可用 / 落盲区），绝不外泄技术堆栈。
    """


class ParamType(str, Enum):
    """能力参数的强类型。LLM 填入的每个参数按其声明类型强校验，类型不符即拒。"""

    STRING = "string"
    INTEGER = "integer"


# 参数值的字符白名单：中英文、数字、下划线、连字符、空格。刻意不含引号/分号/括号/等号等
# SQL 元字符——注入型输入（如 `1; DROP TABLE`、`' OR '1'='1`）因此被结构性拒绝。
_SAFE_STRING = re.compile(r"^[\w一-鿿\- ]+$")


@dataclass(frozen=True)
class ParamSpec:
    """单个能力参数的规格：强类型校验 + 可选白名单 + 长度上限。

    校验在能力执行前完成，非法值抛 `DatabaseToolError`；通过校验的值才作为参数绑定进
    人写死的 SQL（参数化占位，绝不字符串拼接）。
    """

    name: str
    type: ParamType
    required: bool = True
    # 枚举白名单（如学院/专业代码集）；非空时值必须在其中，进一步收窄注入面。
    choices: tuple[str, ...] | None = None
    # 字符串最大长度，防超长输入。
    max_length: int = 64

    def validate(self, value: Any) -> Any:
        """校验并规范化单个参数值，返回可安全绑定的值；非法即抛 DatabaseToolError。"""
        if self.type is ParamType.INTEGER:
            return self._validate_integer(value)
        return self._validate_string(value)

    def _validate_integer(self, value: Any) -> int:
        # 明确拒绝 bool（Python 中 bool 是 int 子类，避免 True 被当 1 混入）。
        if isinstance(value, bool):
            raise DatabaseToolError(f"参数 {self.name} 需为整数，得到布尔值")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip())
        raise DatabaseToolError(f"参数 {self.name} 需为整数，得到非法值：{value!r}")

    def _validate_string(self, value: Any) -> str:
        if not isinstance(value, str):
            raise DatabaseToolError(f"参数 {self.name} 需为字符串，得到 {type(value).__name__}")
        text = value.strip()
        if not text:
            raise DatabaseToolError(f"参数 {self.name} 不能为空")
        if len(text) > self.max_length:
            raise DatabaseToolError(f"参数 {self.name} 超出长度上限 {self.max_length}")
        if self.choices is not None:
            if text not in self.choices:
                raise DatabaseToolError(
                    f"参数 {self.name} 取值须在白名单内：{self.choices}，得到 {text!r}"
                )
            return text
        if not _SAFE_STRING.match(text):
            # 命中此路径的典型是注入型输入（含引号/分号/SQL 元字符）。
            raise DatabaseToolError(f"参数 {self.name} 含非法字符（疑似注入），已拒绝：{text!r}")
        return text


@runtime_checkable
class DatabaseBackend(Protocol):
    """只读查询执行器接口（可插拔）。

    真实实现连接只读账号的数据库，按人写死 SQL + 已校验参数做参数化查询，返回结果行；
    真实只读库是挂起依赖，切片 3 用 `FakeDatabaseBackend` 打桩，主体图零改动即可替换。
    """

    def run_query(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...


class FakeDatabaseBackend:
    """只读查询执行器的假实现（打桩）。

    以「能力名 → 结果行列表」预置返回；回放时按能力的人写死 SQL 匹配（每能力 SQL 唯一），
    故与真实后端签名一致（只收 sql + 已校验参数，不含额外约定键）。记录每次执行（SQL/参数），
    供测试断言参数化调用与只读性。
    """

    def __init__(
        self,
        rows_by_capability: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        # key 为能力名，value 为该能力返回的结果行；未预置的能力返回空列表（模拟无数据）。
        # 内部转成「SQL → 行」以便按后端真实入参（SQL）回放。
        self._rows_by_sql: dict[str, list[dict[str, Any]]] = {}
        for name, rows in (rows_by_capability or {}).items():
            cap = _BUILTIN_SQL_BY_NAME.get(name)
            if cap is not None:
                self._rows_by_sql[cap] = rows
        self.calls: list[dict[str, Any]] = []  # 记录执行，供测试断言

    def run_query(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        # 后端只认参数化 SQL + 已校验参数字典；不解析 SQL，仅按其回放预置行。
        # 真实后端在此用只读连接执行参数化查询。
        self.calls.append({"sql": sql, "params": dict(params)})
        return [dict(r) for r in self._rows_by_sql.get(sql, [])]


@dataclass(frozen=True)
class QueryCapability:
    """一条参数化业务查询能力（ADR 0004 的最小单元）。

    - `sql`：人写死的只读查询（注册期校验只读性），仅含命名参数占位（`:name`），绝不拼接。
    - `params`：参数规格清单，LLM 填入的 params 逐项强类型校验。
    - `to_evidence`：把结果行规整成 `INTERNAL_DATABASE` 的 Evidence（`raw` 保留原始行）。
    """

    name: str
    description: str
    sql: str
    params: tuple[ParamSpec, ...]
    # 结果行 → 一句可回溯的证据文本（content）与来源标注（citation）的渲染规格。
    content_template: str
    citation_template: str

    def __post_init__(self) -> None:
        self._assert_read_only()

    def _assert_read_only(self) -> None:
        """注册期防呆：模板 SQL 必须只读（SELECT/WITH 开头且不含写/DDL 关键字）。"""
        stripped = self.sql.lstrip().lower()
        if not stripped.startswith(_READ_ONLY_PREFIXES):
            raise DatabaseToolError(
                f"能力 {self.name} 的 SQL 非只读（须以 SELECT/WITH 开头）：{self.sql!r}"
            )
        if _FORBIDDEN_SQL.search(self.sql):
            raise DatabaseToolError(
                f"能力 {self.name} 的 SQL 含写/DDL 关键字，拒绝注册：{self.sql!r}"
            )

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def validate_params(self, raw_params: dict[str, Any]) -> dict[str, Any]:
        """强类型校验全部参数：缺参、多余参、类型/白名单/注入违规均抛 DatabaseToolError。"""
        if not isinstance(raw_params, dict):
            raise DatabaseToolError(f"能力 {self.name} 的 params 需为对象，得到 {type(raw_params).__name__}")
        unknown = set(raw_params) - set(self.param_names)
        if unknown:
            raise DatabaseToolError(f"能力 {self.name} 收到未知参数：{sorted(unknown)}")
        validated: dict[str, Any] = {}
        for spec in self.params:
            if spec.name not in raw_params:
                if spec.required:
                    raise DatabaseToolError(f"能力 {self.name} 缺少必填参数：{spec.name}")
                continue
            validated[spec.name] = spec.validate(raw_params[spec.name])
        return validated

    def row_to_evidence(self, row: dict[str, Any]) -> Evidence:
        """把单个结果行规整成 Evidence：content/citation 按模板渲染，raw 保留原始行供数字溯源。"""
        return Evidence(
            source_type=SourceType.INTERNAL_DATABASE,
            content=self._safe_format(self.content_template, row),
            citation=self._safe_format(self.citation_template, row),
            raw=dict(row),
        )

    @staticmethod
    def _safe_format(template: str, row: dict[str, Any]) -> str:
        """按结果行字段渲染模板；缺字段不炸，占位保留原样，保证脏数据不阻断主体。"""

        class _Default(dict):
            def __missing__(self, key: str) -> str:  # noqa: D401
                return "{" + key + "}"

        return template.format_map(_Default(row))


class query_capability(BaseModel):  # noqa: N801 —— 类名即工具名，须与领域动作一致
    """从数据库查询一条预定义业务能力：只选能力名 + 填参数，不写 SQL。

    需要校内统计数据（对口率/考勤率/实习状态/薪资等）时，内核调用本工具，`capability` 取
    可用能力名之一，`params` 按该能力要求填入参数。SQL 由系统写死、只读，你无法也无需提供 SQL。
    """

    capability: str = Field(description="要调用的业务查询能力名（须为系统已注册的能力之一）")
    params: dict[str, Any] = Field(
        default_factory=dict, description="该能力所需的参数键值对（按能力声明填写）"
    )


@dataclass
class DatabaseTool:
    """数据库工具：参数化查询能力的注册表 + 执行入口。

    只认已注册的能力名——自由 SQL 无从进入（工具入口只接受 `{capability, params}`，未注册
    能力名直接拒），从结构上杜绝 Text2SQL 与注入。执行流程：查表取能力 → 强类型校验参数 →
    经只读后端参数化执行 → 结果行规整为 `INTERNAL_DATABASE` Evidence。
    """

    backend: DatabaseBackend
    capabilities: dict[str, QueryCapability] = field(default_factory=dict)

    def register(self, capability: QueryCapability) -> None:
        """注册一条能力模板（注册期已在 QueryCapability 内校验只读性）。"""
        self.capabilities[capability.name] = capability

    @property
    def capability_names(self) -> list[str]:
        return sorted(self.capabilities)

    def describe_capabilities(self) -> str:
        """把可用能力清单渲染成给内核选能力用的说明（名称 + 描述 + 参数）。"""
        lines: list[str] = []
        for name in self.capability_names:
            cap = self.capabilities[name]
            param_desc = "、".join(
                f"{p.name}({p.type.value}{'，必填' if p.required else ''})" for p in cap.params
            )
            lines.append(f"- {name}：{cap.description}；参数：{param_desc or '无'}")
        return "\n".join(lines)

    def run(self, capability: str, params: dict[str, Any]) -> list[Evidence]:
        """执行一次参数化能力调用，返回 Evidence 列表；非法入参抛 DatabaseToolError。

        - 未注册能力名 → 拒（自由 SQL 无入口）。
        - 参数强类型校验：缺参/多参/类型错/白名单外/注入型输入 → 拒。
        - 只读后端参数化执行；结果行经能力的规整逻辑产出 Evidence（raw 保留原始行）。
        """
        cap = self.capabilities.get(capability)
        if cap is None:
            raise DatabaseToolError(
                f"未知的查询能力：{capability!r}；可用能力：{self.capability_names}"
            )
        validated = cap.validate_params(params)
        # 只把已校验参数传给只读后端做参数化执行（绝不字符串拼接）。
        rows = self.backend.run_query(cap.sql, validated)
        return [cap.row_to_evidence(row) for row in rows]


# ── 内置能力模板（ADR 0004：至少落地一个端到端可用能力）──

# 「查某学院某专业实习对口率」：参数化、只读、数字取自结果行 `rate`（供 raw 溯源）。
INTERNSHIP_PLACEMENT_RATE = QueryCapability(
    name="internship_placement_rate",
    description="查某学院某专业的实习对口率（岗位与所学专业相符的实习占比）",
    sql=(
        "SELECT college, major, placement_rate AS rate, sample_size, term "
        "FROM internship_stats "
        "WHERE college = :college AND major = :major"
    ),
    params=(
        ParamSpec(name="college", type=ParamType.STRING),
        ParamSpec(name="major", type=ParamType.STRING),
    ),
    content_template="{college}{major}实习对口率为 {rate}%（{term}，样本 {sample_size} 人）。",
    citation_template="校内数据库/internship_stats/{college}-{major}",
)


# 内置能力名 → 其人写死 SQL 的索引，供 FakeDatabaseBackend 按 SQL 回放预置行。
_BUILTIN_SQL_BY_NAME: dict[str, str] = {
    INTERNSHIP_PLACEMENT_RATE.name: INTERNSHIP_PLACEMENT_RATE.sql,
}


def build_default_database_tool(backend: DatabaseBackend | None = None) -> DatabaseTool:
    """构造带内置能力集的数据库工具。

    backend 省略则用 `FakeDatabaseBackend`（打桩）；真实只读库到位后注入真实后端即可，
    能力模板与主体图零改动。新增业务查询 = 在此注册一条新的 QueryCapability。
    """
    tool = DatabaseTool(backend=backend or FakeDatabaseBackend())
    tool.register(INTERNSHIP_PLACEMENT_RATE)
    return tool
