"""集中配置：全部凭据经 .env 注入，代码零硬编码。

- Agent 两档模型共用同一 OpenAI 兼容 base_url + key，按角色切 model 名。
- Langfuse tracing 凭据可选：缺失时 tracing 关闭，不影响图行为。
- Redis checkpointer 凭据（切片 7）：连接参数取值参考 `redis.conf`，但一律经 .env 注入、
  不写死、不进版本库；checkpointer 用的 `database` 索引与业务用途隔离（ADR/切片 7）。
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
        # 字段均带大写 alias（如 max_tool_calls → MAX_TOOL_CALLS）用于 .env / 环境变量；
        # 同时允许用字段名构造（Settings(max_tool_calls=2)），便于测试注入。
        populate_by_name=True,
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

    # ── Redis checkpointer（跨轮持久化整个 State，ADR/切片 7）──
    # 连接参数取值参考 redis.conf（host/port/password/database/timeout/ssl），
    # 真实值经 .env 注入。会话隔离仅靠 thread_id=session_id 唯一性（ADR 0005）。
    redis_host: str | None = Field(default=None, alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")
    # 通用业务 db（redis.conf 的 database=4）；checkpointer 不用此 db。
    redis_db: int = Field(default=4, alias="REDIS_DB")
    # checkpointer 专用 db 索引，须与业务 db（REDIS_DB）隔离，避免会话状态与其他用途串台。
    redis_checkpointer_db: int = Field(default=0, alias="REDIS_CHECKPOINTER_DB")
    redis_ssl: bool = Field(default=False, alias="REDIS_SSL")
    # 秒；对应 redis.conf 的 timeout（10000ms）。
    redis_timeout: float = Field(default=10.0, alias="REDIS_TIMEOUT")
    # 会话记忆 TTL（分钟），可配置；占位默认值，待业务确认实际取值后覆盖。
    session_ttl_minutes: float = Field(default=60.0, alias="SESSION_TTL_MINUTES")

    # ── loop 安全阀（切片 8）──
    # 内核 loop 的两条硬上限，触发即兜底强制作答，防止死循环烧 token。
    # 最大工具调用数（检索工具实际调用次数：知识库/数据库能力/联网）。初值 6，上线后按真实分布调。
    max_tool_calls: int = Field(default=6, alias="MAX_TOOL_CALLS")
    # 最大迭代步数（检索层 + 精修步的节点访问数）。初值 8；当前有界管线正常轮次≤5，
    # 此阈值作为结构兜底，自由 ReAct 回边循环引入后才会自然触顶。
    max_iterations: int = Field(default=8, alias="MAX_ITERATIONS")

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

    @property
    def has_redis_config(self) -> bool:
        """是否配置了 Redis host（据此决定默认 checkpointer 用 Redis 还是内存版）。"""
        return bool(self.redis_host)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例配置。"""
    return Settings()
