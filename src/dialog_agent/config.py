"""集中配置：全部凭据经 .env 注入，代码零硬编码。

- Agent 两档模型共用同一 OpenAI 兼容 base_url + key，按角色切 model 名。
- Langfuse tracing 凭据可选：缺失时 tracing 关闭，不影响图行为。
真实值由用户在 .env 填写（见 .env.example）；测试环境注入桩，不依赖此配置。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从 .env / 进程环境变量读取的运行时配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Agent 两档模型（共用 base_url + key，按角色切 model 名）──
    model_base_url: str | None = Field(default=None, alias="MODEL_BASE_URL")
    model_api_key: str | None = Field(default=None, alias="MODEL_API_KEY")
    strong_model: str = Field(default="kimi-2.6", alias="STRONG_MODEL")
    fast_model: str = Field(default="deepseek-v4-pro", alias="FAST_MODEL")

    # ── Langfuse 可观测性（tracing，可选）──
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_base_url: str | None = Field(default=None, alias="LANGFUSE_BASE_URL")

    @property
    def has_model_credentials(self) -> bool:
        return bool(self.model_base_url and self.model_api_key)

    @property
    def has_langfuse(self) -> bool:
        return bool(
            self.langfuse_secret_key
            and self.langfuse_public_key
            and self.langfuse_base_url
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例配置。"""
    return Settings()
