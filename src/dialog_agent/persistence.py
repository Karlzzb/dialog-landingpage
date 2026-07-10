"""Redis 版 LangGraph Checkpointer（切片 7）。

把切片 1 的内存版 `MemorySaver` 换成 `RedisSaver`：持久化整个 State（含会话记忆/滚动实体
摘要），`thread_id = session_id`。会话隔离仅靠 session id 唯一性（ADR 0005），后端每轮只传
session id + 本轮 user input；改写读记忆、每轮末更新，全在图内闭环。

连接参数（host/port/database/password/timeout/ssl）一律经 `.env` 注入（取值参考 `redis.conf`，
但 `redis.conf` 不入库、不直接读取）。checkpointer 用的 `database` 索引与业务用途隔离
（见 `Settings.redis_checkpointer_db`）。会话记忆设 TTL 避免无限增长，取值可配置。

`RedisSaver` 需要 Redis Stack（RedisJSON + RediSearch）才能建索引与存 checkpoint；
`build_redis_checkpointer` 构造 saver 并调用 `setup()` 建索引。生产环境应有可达的 Redis Stack；
本地无 Redis 时 `build_graph` 的默认 checkpointer 会优雅降级回内存版（见 `graph.py`）。
"""

from __future__ import annotations

import logging

import redis
from langgraph.checkpoint.redis import RedisSaver

from .config import Settings

logger = logging.getLogger(__name__)


def build_redis_checkpointer(settings: Settings) -> RedisSaver:
    """按 .env 配置构造 Redis 版 checkpointer 并建好索引。

    - `redis_client` 显式构造，完整控制 host/port/password/ssl/timeout，避免把口令拼进 URL。
    - `db` 取 `redis_checkpointer_db`（与业务 db 隔离）。
    - `ttl={"default_ttl": session_ttl_minutes, "refresh_on_read": True}`：会话记忆 TTL 可配置，
      读时刷新使活跃会话不因过期丢失（TTL 单位为分钟，由 RedisSaver 内部换算成秒）。
    - `setup()` 在 Redis 建 RediSearch/RedisJSON 索引；需要 Redis Stack，否则抛错交由调用方降级。
    """
    client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_checkpointer_db,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        socket_timeout=settings.redis_timeout,
        socket_connect_timeout=settings.redis_timeout,
    )
    saver = RedisSaver(
        redis_client=client,
        ttl={
            "default_ttl": settings.session_ttl_minutes,
            "refresh_on_read": True,
        },
    )
    saver.setup()
    return saver
