"""知识库工具适配层 —— 检索(retrieval)侧，返回统一 Evidence 契约。

对接 bisheng 知识库后端的检索端点（query → 相关 chunk + 出处）。真实检索端点是**阻塞
依赖**（`bisheng_rag_api.py` 只覆盖入库侧），故本切片以假实现打桩：`FakeKnowledgeRetriever`
返回符合契约的假 chunk。检索端点到位后，仅替换 `KnowledgeRetriever` 实现，主体图与其余工具
零改动（可插拔接缝）。

分层：
- `KnowledgeRetriever` 协议：适配层的稳定接口（query → Evidence[]）。
- `FakeKnowledgeRetriever`：假实现，可编排返回，供图桩化与离线联调。
- `to_evidence`：bisheng 检索响应 → Evidence 的纯函数规整（真实端点到位后复用）。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .evidence import Evidence, SourceType


@runtime_checkable
class KnowledgeRetriever(Protocol):
    """知识库检索适配层接口。

    输入自包含查询（可选目标知识库 id / 过滤条件），返回 `INTERNAL_KNOWLEDGE` 的
    Evidence 列表。真实实现负责携带 Cookie+JWT 调 bisheng 检索端点、解析统一信封、
    经 `to_evidence` 规整；本接口对上游只暴露 query → Evidence[]。
    """

    def retrieve(
        self, query: str, *, knowledge_id: str | None = None, top_k: int = 3
    ) -> list[Evidence]: ...


def to_evidence(chunk: dict[str, Any]) -> Evidence:
    """把 bisheng 检索返回的单个 chunk 规整成 Evidence（纯函数）。

    真实检索端点到位后，用其响应结构复用本函数即可产出契约化 Evidence——citation 组装
    文件名/知识库/chunk_index 等可回溯信息，raw 保留原始 chunk 供溯源。
    """
    file_name = chunk.get("file_name") or chunk.get("source") or "未知文件"
    kb_name = chunk.get("knowledge_name") or chunk.get("knowledge_id") or "知识库"
    chunk_index = chunk.get("chunk_index")
    citation = f"{kb_name}/{file_name}"
    if chunk_index is not None:
        citation = f"{citation}#chunk{chunk_index}"
    return Evidence(
        source_type=SourceType.INTERNAL_KNOWLEDGE,
        content=str(chunk.get("content", "")),
        citation=citation,
        raw=dict(chunk),
    )


class FakeKnowledgeRetriever:
    """知识库检索的假实现（打桩）。

    以「查询关键片段 → chunk 列表」的映射预置返回，命中即产 Evidence；未命中返回空列表
    （模拟盲区，供后续切片验证降级/盲区路径）。检索端点到位前用于图桩化与离线联调。
    """

    def __init__(self, corpus: dict[str, list[dict[str, Any]]] | None = None) -> None:
        # key 为查询里的关键子串，value 为该查询命中的原始 chunk 列表。
        self._corpus = corpus or {}
        self.calls: list[dict[str, Any]] = []  # 记录调用，供测试断言

    def retrieve(
        self, query: str, *, knowledge_id: str | None = None, top_k: int = 3
    ) -> list[Evidence]:
        self.calls.append({"query": query, "knowledge_id": knowledge_id, "top_k": top_k})
        for key, chunks in self._corpus.items():
            if key in query:
                return [to_evidence(c) for c in chunks[:top_k]]
        return []
