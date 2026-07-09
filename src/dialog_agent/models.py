"""两档模型容器 —— 图的依赖注入接缝。

生产环境用 `build_models()` 从 .env 构造真实 ChatOpenAI（共用 base_url+key，按角色切
model 名）；测试注入实现同样接口的桩，`build_graph(models=...)` 即可确定性驱动全图。

节点对模型的最小依赖面：
- `.invoke(messages) -> AIMessage`
- `.bind_tools(tools) -> Runnable`（返回物同样支持 `.invoke`）
ChatOpenAI 与测试桩都满足此面。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings, get_settings


@dataclass(frozen=True)
class Models:
    """按角色分档的模型集合。

    strong：ReAct 规划循环 / 工具编排 / 结论生成。
    fast：  问题改写 / 对话流闲聊。
    """

    strong: Any
    fast: Any


def build_models(settings: Settings | None = None) -> Models:
    """从配置构造真实的两档 ChatOpenAI。缺凭据即显式报错（不静默降级）。"""
    settings = settings or get_settings()
    if not settings.has_model_credentials:
        raise RuntimeError(
            "缺少模型凭据 MODEL_BASE_URL / MODEL_API_KEY，请在 .env 配置（参考 .env.example）"
        )

    # 延迟导入：仅真实构造时才需要 langchain_openai，测试注入桩时无此依赖负担。
    from langchain_openai import ChatOpenAI

    common: dict[str, Any] = {
        "base_url": settings.model_base_url,
        "api_key": settings.model_api_key,
        "timeout": 60,
        "max_retries": 2,
    }
    return Models(
        strong=ChatOpenAI(model=settings.strong_model, temperature=0.3, **common),
        fast=ChatOpenAI(model=settings.fast_model, temperature=0.5, **common),
    )
