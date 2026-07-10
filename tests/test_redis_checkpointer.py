"""切片 7 —— Redis checkpointer 接线（配置 + 构造）的单元测试（第二接缝：工具/适配层）。

不连真实 Redis：只断言「凭据经 .env 注入、TTL 可配、checkpointer db 与业务 db 隔离、
`build_redis_checkpointer` 据配置构造出带正确 `ttl_config` 的 `RedisSaver`」。真实跨轮持久化/隔离
由 `test_redis_persistence_isolation.py` 用可达 Redis Stack 实例验证（不可达时 skip）。
"""

from __future__ import annotations

from langgraph.checkpoint.redis import RedisSaver

from dialog_agent.config import Settings
from dialog_agent.persistence import build_redis_checkpointer


def test_redis_connection_params_injected_from_env():
    """Redis 连接参数（host/port/password/database/timeout/ssl）经环境变量/.env 注入。"""
    settings = Settings(
        _env_file=None,
        REDIS_HOST="redis.example.com",
        REDIS_PORT=6380,
        REDIS_PASSWORD="s3cret",
        REDIS_DB=4,
        REDIS_CHECKPOINTER_DB=2,
        REDIS_SSL=True,
        REDIS_TIMEOUT=7,
        SESSION_TTL_MINUTES=120,
    )
    assert settings.redis_host == "redis.example.com"
    assert settings.redis_port == 6380
    assert settings.redis_password == "s3cret"
    assert settings.redis_db == 4
    assert settings.redis_checkpointer_db == 2
    assert settings.redis_ssl is True
    assert settings.redis_timeout == 7
    # TTL 可配置（取值来自配置，不是写死）。
    assert settings.session_ttl_minutes == 120
    assert settings.has_redis_config is True


def test_no_redis_host_means_no_redis_config():
    """未配 REDIS_HOST 时 has_redis_config 为 False —— 默认走内存版 checkpointer。"""
    settings = Settings(_env_file=None)
    assert settings.has_redis_config is False


def test_checkpointer_db_isolated_from_business_db():
    """checkpointer 用的 db 索引须与业务 db（REDIS_DB）隔离，避免会话状态与业务用途串台。"""
    settings = Settings(
        _env_file=None,
        REDIS_HOST="redis.example.com",
        REDIS_DB=4,
        REDIS_CHECKPOINTER_DB=0,
    )
    assert settings.redis_checkpointer_db != settings.redis_db


def test_build_redis_checkpointer_constructs_saver_with_ttl(monkeypatch):
    """`build_redis_checkpointer` 据配置构造 `RedisSaver`，TTL 取自可配置的 session_ttl_minutes。

    打桩 `RedisSaver.setup`（建索引需连真实 Redis Stack），只验证构造产物与 ttl_config——
    即「换成了 Redis 后端、TTL 可配置、凭据经配置注入」这条接线，不在此连 Redis。
    """
    constructed: dict = {}

    class FakeRedisClient:
        """记录构造参数的 redis 客户端替身（不连网）。"""

        def __init__(self, **kwargs):
            constructed["client_kwargs"] = kwargs

    import dialog_agent.persistence as persistence_mod

    monkeypatch.setattr(persistence_mod, "redis", type("R", (), {"Redis": FakeRedisClient}))
    monkeypatch.setattr(persistence_mod.RedisSaver, "setup", lambda self: None, raising=False)

    settings = Settings(
        _env_file=None,
        REDIS_HOST="redis.example.com",
        REDIS_PORT=6380,
        REDIS_PASSWORD="s3cret",
        REDIS_CHECKPOINTER_DB=3,
        REDIS_SSL=True,
        REDIS_TIMEOUT=9,
        SESSION_TTL_MINUTES=45,
    )

    saver = build_redis_checkpointer(settings)

    assert isinstance(saver, RedisSaver)
    # TTL 经配置注入：default_ttl 取自 session_ttl_minutes；读时刷新保活跃会话不丢。
    assert saver.ttl_config["default_ttl"] == 45
    assert saver.ttl_config["refresh_on_read"] is True
    # 客户端用 checkpointer 专用 db（与业务 db 隔离），ssl/timeout/host/port/password 全部来自配置。
    ck = constructed["client_kwargs"]
    assert ck["host"] == "redis.example.com"
    assert ck["port"] == 6380
    assert ck["password"] == "s3cret"
    assert ck["db"] == 3
    assert ck["ssl"] is True
    assert ck["socket_timeout"] == 9
    assert ck["socket_connect_timeout"] == 9
