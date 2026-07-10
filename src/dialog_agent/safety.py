"""loop 安全阀 + 优雅兜底（切片 8）。

内核 loop 有三条退出路径（覆盖完成已在切片 5；本切片接入另两条）：

1. 盲区（Blind Spot）：知识库→数据库→联网三层穷尽后仍未覆盖的信息单元，是 loop 的**一等状态**
   （非异常）。结论生成时坦诚告知边界、给补救/下一步指引，绝不编造。
2. 安全阀（Safety Caps）：最大工具调用 `max_tool_calls`（初值 6）/ 最大迭代 `max_iterations`
   （初值 8），均可经 .env 配置；触顶即兜底强制作答，不死循环烧 token。
3. 优雅兜底：任意异常被捕获 → 人性化兜底话术，技术堆栈绝不外泄。

本模块提供：
- `SafetyCaps`：两阈值的不可变容器，从 `Settings` 构造，注入图与各检索/精修节点。
- `safety_valve_tripped`：判定当前是否已超阈值，供条件路由使用。
- `FALLBACK_REPLY`：全局异常兜底的固定人性化话术（零技术细节）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings


@dataclass(frozen=True)
class SafetyCaps:
    """loop 安全阀的两条阈值（可经 .env 配置，初值取 PRD/CONTEXT.md 的 6 / 8）。

    - `max_tool_calls`：检索工具实际调用次数上限（知识库 retrieve / 数据库 query_capability /
      联网 search 各计一次）。触顶即在当前层 per-unit 循环内 break、并经条件边短路到 final_answer。
    - `max_iterations`：检索层 + 精修步的节点访问数上限。当前有界管线正常一轮≤5，此值作结构兜底。
    """

    max_tool_calls: int = 6
    max_iterations: int = 8

    @classmethod
    def from_settings(cls, settings: Settings) -> "SafetyCaps":
        return cls(
            max_tool_calls=settings.max_tool_calls,
            max_iterations=settings.max_iterations,
        )


def safety_valve_tripped(
    tool_call_count: int, iteration_count: int, caps: SafetyCaps
) -> bool:
    """是否已触发安全阀（任一阈值达上限即兜底强制作答）。"""
    return tool_call_count >= caps.max_tool_calls or iteration_count >= caps.max_iterations


# 全局异常兜底的固定人性化话术。零技术细节——不拼异常文本、不含堆栈/报错关键字，
# 确保用户侧绝不外泄技术堆栈。异常本身经 `logger.exception` 落运维日志，不进回复。
FALLBACK_REPLY = (
    "抱歉，我在处理您的问题时遇到了一点状况，暂时无法给出答复。"
    "请稍后重试，或换一种方式描述您的问题，我会尽快为您处理。"
)
