"""Langfuse 可观测性接入 —— 挂到图执行层的横切基础设施。

在 `invoke` 边界注入 LangGraph callback handler，使 ReAct 循环 / 工具调用 / LLM 调用被
自动 trace，后续切片无需逐处补埋点。此处仅接 tracing，不含告警/看板。

配置经 .env 注入；凭据缺失或未安装 langfuse 时静默关闭（返回空 callbacks），不影响图行为
——测试环境即据此关闭 tracing。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .config import Settings, get_settings


@lru_cache(maxsize=1)
def _init_langfuse(secret_key: str, public_key: str, base_url: str) -> Any:
    """初始化并缓存 Langfuse 客户端与 langchain CallbackHandler。

    失败（未安装 / 版本不兼容）时返回 None，让调用方降级为无 tracing。
    """
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return None

    # 显式构造客户端，避免依赖进程环境变量是否已导出。
    Langfuse(secret_key=secret_key, public_key=public_key, host=base_url)
    return CallbackHandler()


def build_langfuse_callbacks(settings: Settings | None = None) -> list[Any]:
    """返回注入 LangGraph 的 callback 列表；无凭据/不可用时返回 []（tracing 关闭）。"""
    settings = settings or get_settings()
    if not settings.has_langfuse:
        return []
    handler = _init_langfuse(
        settings.langfuse_secret_key,  # type: ignore[arg-type]
        settings.langfuse_public_key,  # type: ignore[arg-type]
        settings.langfuse_base_url,  # type: ignore[arg-type]
    )
    return [handler] if handler is not None else []
