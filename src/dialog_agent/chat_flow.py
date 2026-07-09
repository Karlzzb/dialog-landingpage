"""对话流（Chat Flow）约束。

对话流 = 寒暄/问候/情绪/主观闲聊，目的是拟人化承接，不触发检索。
行为约束（专家口吻、禁越界、尾部业务引导拉回）靠 system prompt 保证；其中唯一可确定性
度量的「≤50 字」再加一次纯代码后处理截断作廉价兜底（CONTEXT.md「对话流行为约束」）。
"""

from __future__ import annotations

# 对话流回复长度上限（字符数）。可确定性度量，故加代码兜底。
CHAT_REPLY_MAX_CHARS = 50

# 内核判定「无需工具」后，对话流作答的 system prompt。
CHAT_FLOW_SYSTEM_PROMPT = (
    "你是「产教融合专家助理」，服务职业教育一线的教师与院系/校级业务人员。"
    "当前是寒暄或与业务无关的闲聊，请用专家口吻简短、礼貌地承接，"
    "不要展开、不要越界回答专业问题，并在结尾自然地把话题引导回产教融合业务。"
    f"整段回复必须不超过 {CHAT_REPLY_MAX_CHARS} 个字。"
)


def truncate_reply(text: str, max_chars: int = CHAT_REPLY_MAX_CHARS) -> str:
    """对话流回复的纯代码后处理：规整空白并硬截断到 ≤max_chars 字。

    这是廉价兜底，不替代 prompt 的自律；仅保证「≤50 字」这条硬指标不被突破。
    """
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars]
