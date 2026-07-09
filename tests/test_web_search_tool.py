"""第二接缝 —— 联网工具 + 边界内容过滤的纯函数式单元测试（ADR 0007）。

脱离图独立测：外部条目 → Evidence(EXTERNAL_SEARCH) 契约规整、ContentFilter 清洗/拦截、
WebSearchTool 端到端过滤。核心安全属性——「含敏感词的外部原文绝不透传进 State/LLM」
在此层结构性验证，而非事后质检。
"""

from __future__ import annotations

from dialog_agent.evidence import SourceType
from dialog_agent.web_search_tool import (
    ContentFilter,
    FakeWebSearchBackend,
    WebSearchTool,
    build_default_web_search_tool,
    to_evidence,
)

# 外部检索返回的原始条目（未过滤）：企业工商信息，含可溯源的数字字段。
EXTERNAL_ITEM = {
    "content": "某科技有限公司 2024 年营业收入 1.2 亿元，参保人数 80 人。",
    "url": "https://example.com/firm/123",
    "domain": "example.com",
    "revenue_yi": 1.2,  # 亿元，数字字段供 raw 溯源
    "headcount": 80,
}


# ── 契约规整：外部条目 → EXTERNAL_SEARCH Evidence ──


def test_to_evidence_external_search_contract():
    ev = to_evidence(EXTERNAL_ITEM)
    assert ev.source_type == SourceType.EXTERNAL_SEARCH
    assert ev.citation == "https://example.com/firm/123"
    # content 取条目正文。
    assert "1.2 亿元" in ev.content
    # raw 保留原始字段，数字可溯源。
    assert ev.raw["revenue_yi"] == 1.2
    assert ev.raw["headcount"] == 80


def test_to_evidence_citation_falls_back_when_no_url():
    """无 url 时 citation 回落到来源/域名，最终回落到弱化标注占位。"""
    ev = to_evidence({"content": "x", "source": "某新闻网"})
    assert ev.citation == "某新闻网"
    ev2 = to_evidence({"content": "x"})
    assert ev2.citation == "互联网公开信息"


# ── ContentFilter：清洗（mask）/ 拦截（block）/ 干净透传 ──


def test_mask_term_cleaned_number_preserved():
    """mask 词命中：敏感文本被替换为占位，数字与非敏感文本保留。"""
    flt = ContentFilter(mask_terms=("某敏感词",))
    cleaned = flt.clean_text("某敏感词出现在这里，营收 1.2 亿元。")
    assert "某敏感词" not in cleaned
    assert "[已过滤]" in cleaned
    # 数字保留（数字溯源不能被过滤误伤）。
    assert "1.2 亿元" in cleaned


def test_filter_item_masks_all_string_fields_keeps_numbers():
    flt = ContentFilter(mask_terms=("敏感",))
    item = {
        "content": "这里有敏感内容 1.2 亿元",
        "url": "https://x.example/sensitive",
        "revenue_yi": 1.2,
    }
    filtered = flt.filter_item(item)
    assert filtered is not None
    # 所有字符串字段均脱敏，原始 item 不被修改。
    assert "敏感" not in filtered["content"]
    assert "敏感" not in filtered["url"]
    assert item["content"] == "这里有敏感内容 1.2 亿元"  # 原件未被破坏
    # 数字字段原样保留。
    assert filtered["revenue_yi"] == 1.2


def test_block_term_drops_item_entirely():
    """block 词命中任一字符串字段 → 整条拦截（filter_item 返回 None，不产 Evidence）。"""
    flt = ContentFilter(block_terms=("严重违规词",))
    item = {"content": "正常内容", "url": "https://x/严重违规词/path", "n": 5}
    assert flt.filter_item(item) is None


def test_clean_item_passes_through_unchanged():
    flt = ContentFilter(block_terms=("违禁",), mask_terms=("敏感",))
    item = {"content": "干净内容 80 人", "url": "https://x/y", "n": 80}
    filtered = flt.filter_item(item)
    assert filtered == item  # 干净条目原样通过（仍是新 dict）


def test_default_filter_empty_terms_passes_all():
    """默认空词表：机制就位但无词，所有内容通过（真实词表由部署注入，挂起项）。"""
    flt = ContentFilter()
    assert flt.filter_item(EXTERNAL_ITEM) == EXTERNAL_ITEM


# ── WebSearchTool 端到端：后端取数 → 边界过滤 → Evidence ──


def test_web_search_tool_cleans_sensitive_external_content():
    """工具边界生效：含 mask 词的外部原文 → Evidence.content 已脱敏，敏感词不进产出。"""
    backend = FakeWebSearchBackend(
        {
            "企业": [
                {
                    "content": "敏感违规表述：某公司营收 1.2 亿元。",
                    "url": "https://x/y",
                    "revenue_yi": 1.2,
                }
            ]
        }
    )
    tool = WebSearchTool(backend=backend, content_filter=ContentFilter(mask_terms=("敏感违规表述",)))
    evidence = tool.search("某企业营收")

    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.source_type == SourceType.EXTERNAL_SEARCH
    # 敏感词已从 content 与 raw 中清除，绝不透传。
    assert "敏感违规表述" not in ev.content
    assert "敏感违规表述" not in str(ev.raw)
    assert "[已过滤]" in ev.content
    # 数字保留供溯源。
    assert ev.raw["revenue_yi"] == 1.2


def test_web_search_tool_blocks_item_no_evidence():
    """含 block 词的条目被整条拦截：不产出 Evidence，原始外部文本不进 State。"""
    backend = FakeWebSearchBackend(
        {
            "企业": [
                {"content": "正常 A 公司 1.2 亿元", "url": "https://a"},
                {"content": "严重违规词 出现", "url": "https://b"},
            ]
        }
    )
    tool = WebSearchTool(
        backend=backend, content_filter=ContentFilter(block_terms=("严重违规词",))
    )
    evidence = tool.search("企业营收")

    # 只有第一条通过；第二条被拦截。
    assert len(evidence) == 1
    assert evidence[0].citation == "https://a"
    assert all("严重违规词" not in str(ev.raw) for ev in evidence)


def test_web_search_tool_miss_returns_empty():
    """假后端未命中 → 空列表（模拟盲区/无外部数据）。"""
    tool = build_default_web_search_tool(FakeWebSearchBackend({}))
    assert tool.search("无关查询") == []
    assert tool.calls  # 调用被记录供断言


def test_build_default_tool_uses_fake_backend_and_empty_filter():
    tool = build_default_web_search_tool()
    assert isinstance(tool.backend, FakeWebSearchBackend)
    # 默认空词表过滤器（机制就位，真实词表由部署注入）。
    assert tool.content_filter.block_terms == ()
    assert tool.content_filter.mask_terms == ()
