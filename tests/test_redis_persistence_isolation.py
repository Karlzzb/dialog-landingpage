"""切片 7 —— Redis checkpointer 跨轮持久化 + 会话隔离的行为集成测试（第一接缝：图入口）。

用真实 `RedisSaver`（需可达 Redis Stack：RedisJSON + RediSearch）跨两轮驱动同一会话，注入桩化
知识库检索器 + 桩化 LLM，断言：
- 整个 State（含会话记忆/滚动实体摘要）跨轮持久化：轮2继承轮1的实体摘要、省略句改写为自包含
  查询、内核轮2收到的正是自包含查询而非省略句原文；
- 不同 session id 的会话完全隔离：另一会话的首轮记忆不带首轮会话的实体（若 checkpointer 把
  A 的 State 串进 B 的 thread，B 的 `session_memory` 会保留 A 的实体——见 `merged_with_entities`
  「新值覆盖、旧值保留」；故 B 记忆为空即证明隔离）。

本机无可达 Redis Stack 时整体 skip（ping / setup 失败即 skip）——部署/CI 环境跑真。这是
`test_session_memory_rewrite.py`（内存版 checkpointer）的 Redis 后端对照：同一行为契约，换持久化
后端。只测外部行为，不测内部函数如何被调用。
"""

from __future__ import annotations

import os
import uuid

import pytest
import redis as redis_lib
from langchain_core.messages import AIMessage
from langgraph.checkpoint.redis import RedisSaver

from conftest import make_stub_models, plan_coverage_call, rewrite_call
from dialog_agent.config import Settings
from dialog_agent.graph import build_graph, invoke
from dialog_agent.knowledge_tool import FakeKnowledgeRetriever
from dialog_agent.session_memory import SessionMemory

# 知识库假语料：查询含「校企合作补贴」即命中一条带出处的 chunk（各轮均覆盖）。
CORPUS = {
    "校企合作补贴": [
        {
            "content": "武汉市对校企合作企业给予每生每年 2000 元补贴。",
            "file_name": "校企合作补贴办法.pdf",
            "knowledge_name": "武汉市产教融合政策库",
            "chunk_index": 7,
        }
    ]
}

# 轮1 / 轮2 的强制作答产物。
TURN1_FINAL = (
    "据武汉市产教融合政策库《校企合作补贴办法.pdf》，武汉市对开展校企合作的企业，"
    "按每生每年 2000 元的标准给予补贴。"
)
TURN2_FINAL = (
    "据武汉市产教融合政策库，江夏区校企合作补贴按市级标准每生每年 2000 元执行。"
)


def _test_settings_from_env() -> Settings:
    """从进程环境变量构造 Redis 测试配置（不读 .env，避免依赖本地密钥）。

    CI/部署环境把 REDIS_HOST 等注入环境即可跑真；本机无 REDIS_HOST 则 has_redis_config 为 False。
    """
    return Settings(_env_file=None)


@pytest.fixture
def redis_checkpointer():
    """构造一个可达的真实 RedisSaver；不可达 / 非 Redis Stack 时 skip。

    用独立 `checkpoint_prefix` 与该测试实例绑定，避免与同库其他用途的 checkpoint 键碰撞；
    teardown 对用到的 thread_id 逐个 `delete_thread` 清理。
    """
    settings = _test_settings_from_env()
    if not settings.has_redis_config:
        pytest.skip("未配置 REDIS_HOST —— Redis checkpointer 集成测试跳过（需可达 Redis Stack）")

    # 探测用短超时，避免本机不可达时长时间挂起。
    probe_timeout = float(os.environ.get("TEST_REDIS_CONNECT_TIMEOUT", "2"))
    client = redis_lib.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_checkpointer_db,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        socket_timeout=probe_timeout,
        socket_connect_timeout=probe_timeout,
    )
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001 —— 不可达即 skip，不作失败。
        pytest.skip(f"Redis 不可达（{settings.redis_host}:{settings.redis_port}）：{exc}")

    # 独立前缀：键空间与生产 / 其他用例隔离。
    prefix = f"test_cp_{uuid.uuid4().hex}"
    saver = RedisSaver(
        redis_client=client,
        ttl={
            "default_ttl": settings.session_ttl_minutes,
            "refresh_on_read": True,
        },
        checkpoint_prefix=prefix,
        checkpoint_write_prefix=f"{prefix}_w",
    )
    try:
        saver.setup()
    except Exception as exc:  # noqa: BLE001 —— 非 Redis Stack（缺 RediSearch/RedisJSON）即 skip。
        pytest.skip(f"RedisSaver.setup 失败（需 Redis Stack：RediSearch + RedisJSON）：{exc}")

    used_threads: list[str] = []
    yield saver, used_threads

    for tid in used_threads:
        try:
            saver.delete_thread(tid)
        except Exception:  # noqa: BLE001 —— 清理不阻断用例结果。
            pass
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass


def _build_graph_with(saver: RedisSaver, models) -> object:
    return build_graph(
        models,
        knowledge_retriever=FakeKnowledgeRetriever(CORPUS),
        checkpointer=saver,
    )


def test_state_persists_across_turns_via_redis(redis_checkpointer, test_settings):
    """跨两轮持久化：轮1建立「武汉市/校企合作」→ 轮2「江夏区有额外的吗」继承记忆改写为自包含查询。

    两个独立 invoke（同 session_id）之间 State 完全由 Redis checkpointer 承载——若持久化失效，
    轮2的 session_memory 为空、改写无法继承轮1实体、内核收到的将是省略句原文。
    """
    saver, used_threads = redis_checkpointer
    session_id = f"redis-persist-{uuid.uuid4().hex}"
    used_threads.append(session_id)

    models = make_stub_models(
        strong_responses=[
            # 轮1：内核规划 + 结论。
            plan_coverage_call(
                [{"id": "u1", "need": "武汉市校企合作补贴标准", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
            AIMessage(content=TURN1_FINAL),
            # 轮2：内核规划 + 结论。
            plan_coverage_call(
                [{"id": "u1", "need": "武汉市江夏区校企合作补贴", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
            AIMessage(content=TURN2_FINAL),
        ],
        fast_responses=[
            # 轮1改写：原样自包含，落地实体摘要（地域/主题）。
            rewrite_call(
                "武汉市校企合作补贴是多少",
                entities={"地域": "武汉市", "主题": "校企合作"},
            ),
            # 轮2改写：读会话记忆，把省略句补全为自包含查询。
            rewrite_call(
                "武汉市江夏区校企合作补贴",
                entities={"地域": "武汉市江夏区", "主题": "校企合作"},
            ),
        ],
    )
    graph = _build_graph_with(saver, models)

    # 跨两轮驱动同一会话（thread_id = session id）；两轮间 State 经 Redis 持久化。
    invoke(session_id, "武汉市校企合作补贴是多少", graph=graph, settings=test_settings)
    reply2, state2 = invoke(
        session_id, "江夏区有额外的吗", graph=graph, settings=test_settings
    )

    # 轮2被改写为自包含查询（继承轮1的地域/主题）。
    assert state2["rewritten_query"] == "武汉市江夏区校企合作补贴"
    assert reply2 == TURN2_FINAL

    # 会话记忆跨轮持久并更新：地域精修为含「江夏区」，主题继承。
    memory = SessionMemory.from_dict(state2["session_memory"])
    assert memory.entities["主题"] == "校企合作"
    assert "江夏区" in memory.entities["地域"]

    # 内核轮2收到的正是自包含查询（而非省略句原文「江夏区有额外的吗」）。
    turn2_core_input = models.strong.invocations[2]  # 轮2的 plan_coverage 调用
    assert turn2_core_input[-1].content == "武汉市江夏区校企合作补贴"
    assert "江夏区有额外的吗" not in turn2_core_input[-1].content

    # 单轮消息轨迹本轮用完即弃：轮2的内核不被喂轮1的全量历史（轮1结论不进轮2内核输入）。
    turn1_final_fragment = "按每生每年 2000 元的标准给予补贴"
    assert all(
        turn1_final_fragment not in getattr(m, "content", "")
        for m in turn2_core_input
    )


def test_sessions_isolated_no_context_bleed(redis_checkpointer, test_settings):
    """会话隔离：会话 B 的首轮记忆不带会话 A 的实体（thread_id 唯一性保证不串台）。

    会话 A 首轮落地「武汉市/校企合作」实体；会话 B 首轮改写产出空实体摘要。若 checkpointer 把
    A 的 State 串进 B 的 thread，B 的 `merged_with_entities({})` 会保留 A 的实体（旧值保留），
    故 B 的 session_memory.entities 非空即隔离失效。据此断言 B 记忆为空 = 隔离成立。
    """
    saver, used_threads = redis_checkpointer
    session_a = f"redis-iso-a-{uuid.uuid4().hex}"
    session_b = f"redis-iso-b-{uuid.uuid4().hex}"
    used_threads.extend([session_a, session_b])

    models = make_stub_models(
        strong_responses=[
            # A 首轮：规划 + 结论。
            plan_coverage_call(
                [{"id": "u1", "need": "武汉市校企合作补贴标准", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
            AIMessage(content=TURN1_FINAL),
            # B 首轮：规划 + 结论。
            plan_coverage_call(
                [{"id": "u1", "need": "江夏区校企合作补贴", "sources": ["INTERNAL_KNOWLEDGE"]}]
            ),
            AIMessage(content="据武汉市产教融合政策库，江夏区校企合作补贴按市级标准执行。"),
        ],
        fast_responses=[
            # A 首轮改写：落地实体摘要（地域/主题）。
            rewrite_call(
                "武汉市校企合作补贴是多少",
                entities={"地域": "武汉市", "主题": "校企合作"},
            ),
            # B 首轮改写：空实体摘要——若隔离失效，B 会继承 A 的实体（旧值保留）。
            rewrite_call("江夏区校企合作补贴是多少", entities={}),
        ],
    )
    graph = _build_graph_with(saver, models)

    # A 先写记忆，B 再起一轮——若 thread 串台，B 会读到 A 的 State。
    _, state_a = invoke(
        session_a, "武汉市校企合作补贴是多少", graph=graph, settings=test_settings
    )
    _, state_b = invoke(
        session_b, "江夏区校企合作补贴是多少", graph=graph, settings=test_settings
    )

    # A 的会话记忆含其本轮实体（持久化写入）。
    mem_a = SessionMemory.from_dict(state_a["session_memory"])
    assert mem_a.entities["地域"] == "武汉市"
    assert mem_a.entities["主题"] == "校企合作"

    # B 的会话记忆为空——证明未继承 A 的实体（隔离成立）。
    mem_b = SessionMemory.from_dict(state_b["session_memory"])
    assert mem_b.entities == {}
    assert "武汉市" not in mem_b.entities
