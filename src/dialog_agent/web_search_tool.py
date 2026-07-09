"""联网工具适配层 —— 边界内置内容过滤，返回统一 Evidence 契约（ADR 0007）。

对接外部检索（企业工商/财报、外部新闻、未收录的实时政策等校内绝对没有的数据）。
真实检索后端是**挂起依赖**（与 bisheng 检索端点、真实只读库同性质），故本切片以假实现
打桩；真实后端到位后仅替换 `WebSearchBackend` 实现，主体图与其余工具零改动。

**边界内容安全过滤**是本工具的硬约束（ADR 0007 / PRD User Story 26）：外部内容进 State 前
先经 `ContentFilter` 清洗/拦截，**绝不把未过滤原文喂给 LLM**。这是数据层职责，不是独立
质检节点——过滤发生在 `WebSearchTool.search` 内，产出的 `Evidence.content` 与 `Evidence.raw`
均已是脱敏后的副本，原始外部文本从不持久化进 State。

分层（与知识库/数据库工具同构的可插拔接缝）：
- `WebSearchBackend` 协议：外部检索执行器的稳定接口（query → 原始条目列表，未过滤）。
- `FakeWebSearchBackend`：假实现，可编排返回，供图桩化与离线联调。
- `ContentFilter`：边界敏感词/合规过滤（纯函数式）。block 词命中整条拦截、mask 词命中清洗脱敏。
- `to_evidence`：原始条目（须已清洗）→ `EXTERNAL_SEARCH` Evidence 的纯函数规整。
- `WebSearchTool`：后端 + 过滤器编排入口，对外只暴露 `search(query) -> list[Evidence]`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .evidence import Evidence, SourceType

# 清洗后替换敏感文本的占位标记。结论生成据此识别「此处被过滤」，不还原原文。
_MASK_PLACEHOLDER = "[已过滤]"


@runtime_checkable
class WebSearchBackend(Protocol):
    """外部检索执行器接口（可插拔）。

    真实实现对接联网检索后端（如 web search API），按 query 返回原始条目列表；返回的条目
    **未过滤**——过滤是 `WebSearchTool` 的边界职责，后端只负责取数。真实后端是挂起依赖，
    切片 4 用 `FakeWebSearchBackend` 打桩，主体图零改动即可替换。
    """

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ContentFilter:
    """联网工具边界的内容安全过滤（纯函数式）。

    两级策略，覆盖 PRD「涉政/违法等基础敏感词/合规过滤，清洗/拦截」：
    - `block_terms`：高严重度词。任一字符串字段命中 → 整条条目拦截（不产 Evidence）。
    - `mask_terms`：一般敏感词。命中 → 该敏感子串替换为 `[已过滤]`（数字与非敏感文本保留），
      条目仍进 State 但已脱敏。

    默认词表为空——**真实敏感词词典是部署数据项**（与 bisheng 检索端点、真实只读库同列为
    挂起依赖），交付的是过滤机制；生产部署时注入真实词表（或由配置/外部合规服务加载）。
    测试通过构造时注入词表验证清洗/拦截机制。
    """

    block_terms: tuple[str, ...] = ()
    mask_terms: tuple[str, ...] = ()

    def filter_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """对一条外部条目做边界过滤，返回脱敏后的副本；命中 block 词则返回 None（拦截）。

        先判 block（任一字符串字段含任一 block 词即整条拦截），再对所有字符串字段做 mask
        脱敏。返回的是新 dict，原 item 不被修改；数字等非字符串字段原样保留供溯源。
        """
        for value in _iter_text_values(item):
            if any(term in value for term in self.block_terms):
                return None  # 拦截：绝不进 State、绝不喂 LLM
        return self._mask_item(item)

    def clean_text(self, text: str) -> str:
        """对单段文本做 mask 脱敏（block 词不在此处理，由 filter_item 整条拦截）。"""
        cleaned = text
        for term in self.mask_terms:
            cleaned = cleaned.replace(term, _MASK_PLACEHOLDER)
        return cleaned

    def _mask_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """复制 item 并对其所有字符串字段做 mask 脱敏；数字等非字符串字段原样保留。"""
        masked: dict[str, Any] = {}
        for key, value in item.items():
            masked[key] = self.clean_text(value) if isinstance(value, str) else value
        return masked


def to_evidence(item: dict[str, Any]) -> Evidence:
    """把已清洗的单个外部条目规整成 Evidence（纯函数）。

    调用方须先用 `ContentFilter.filter_item` 清洗——本函数只做契约规整，不重复过滤。
    citation 取可回溯的来源标注（URL > 来源 > 域名 > 占位），raw 保留（清洗后的）原始字段
    供数字溯源；content 取条目正文。
    """
    citation = (
        item.get("url")
        or item.get("source")
        or item.get("domain")
        or "互联网公开信息"
    )
    return Evidence(
        source_type=SourceType.EXTERNAL_SEARCH,
        content=str(item.get("content", "")),
        citation=str(citation),
        raw=dict(item),
    )


@dataclass
class WebSearchTool:
    """联网工具：后端 + 边界过滤的编排入口。

    对外只暴露 `search(query) -> list[Evidence]`：取后端原始条目 → 逐条经 `ContentFilter`
    过滤（拦截/脱敏）→ 非 None 者规整为 `EXTERNAL_SEARCH` Evidence。产出的 Evidence.content
    与 raw 均为脱敏副本，原始外部文本从不进 State、从不喂 LLM。记录每次调用供测试断言。
    """

    backend: WebSearchBackend
    content_filter: ContentFilter = field(default_factory=ContentFilter)
    # 每次检索的记录，供测试断言命中即停/未向下穿透等行为。
    calls: list[dict[str, Any]] = field(default_factory=list)

    def search(self, query: str, *, top_k: int = 5) -> list[Evidence]:
        self.calls.append({"query": query, "top_k": top_k})
        raw_items = self.backend.search(query, top_k=top_k)
        evidence: list[Evidence] = []
        for item in raw_items:
            filtered = self.content_filter.filter_item(item)
            if filtered is None:
                continue  # 整条拦截：不产 Evidence，原文不进 State
            evidence.append(to_evidence(filtered))
        return evidence


class FakeWebSearchBackend:
    """外部检索后端的假实现（打桩）。

    以「查询关键子串 → 原始条目列表」预置返回，命中即返回（模拟原始外部检索结果，**未过滤**，
    过滤在 `WebSearchTool` 边界发生）；未命中返回空列表（模拟盲区/降级）。记录每次检索供测试断言。
    """

    def __init__(self, corpus: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self._corpus = corpus or {}
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "top_k": top_k})
        for key, items in self._corpus.items():
            if key in query:
                return [dict(item) for item in items[:top_k]]
        return []


def build_default_web_search_tool(
    backend: WebSearchBackend | None = None,
    content_filter: ContentFilter | None = None,
) -> WebSearchTool:
    """构造带假后端 + 默认过滤器的联网工具。

    backend 省略则用 `FakeWebSearchBackend`（打桩）；content_filter 省略则用空词表
    `ContentFilter`（机制就位，真实词表由部署注入）。真实检索后端 + 真实合规词表到位后
    注入即可，主体图零改动。
    """
    return WebSearchTool(
        backend=backend or FakeWebSearchBackend(),
        content_filter=content_filter or ContentFilter(),
    )


def _iter_text_values(item: dict[str, Any]):
    """遍历 item 的所有字符串字段值（用于 block 判定）。"""
    for value in item.values():
        if isinstance(value, str):
            yield value
