"""产教融合专家助理 —— 确定性骨架 + ReAct 内核（LangGraph 重构）。

对外主接缝：`invoke(session_id, user_input)`。
"""

from __future__ import annotations

from .coverage import CoverageTable, InformationUnit
from .evidence import Evidence, SourceType
from .graph import build_graph, invoke
from .knowledge_tool import FakeKnowledgeRetriever, KnowledgeRetriever
from .models import Models, build_models
from .session_memory import SessionMemory
from .state import TurnState
from .web_search_tool import (
    ContentFilter,
    FakeWebSearchBackend,
    WebSearchBackend,
    WebSearchTool,
)

__all__ = [
    "invoke",
    "build_graph",
    "build_models",
    "Models",
    "TurnState",
    "SessionMemory",
    "Evidence",
    "SourceType",
    "CoverageTable",
    "InformationUnit",
    "KnowledgeRetriever",
    "FakeKnowledgeRetriever",
    "WebSearchBackend",
    "WebSearchTool",
    "FakeWebSearchBackend",
    "ContentFilter",
]
