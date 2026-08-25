"""CloudEnricher —— 真 Anthropic Claude API 打标实现。

把 OCR 文本 + 元数据发给 Claude，返回一个精简的搜索 tags 数组（真实语义标签，
**不带 `mock:` 前缀**）。离线测试请用 MockCloudEnricher。

密钥：只通过 pipeline.env.get_anthropic_api_key() 从 os.environ 读（复用已有基础设施），
**绝不硬编码**。客户端惰性构造：实例化本类不需要密钥，首次调用 enrich() 时才要。

错误语义（对齐阶段一"断网/失败→pending"逻辑）：
- 瞬时失败（网络 / 限流 429 / 5xx / 过载）→ 抛 EnricherError → 上层入 pending 队列，联网后重试。
- 配置错误（密钥无效 401 / 无权限 403 / 模型 ID 错误 404 / 请求非法 400）→ 抛 EnricherConfigError
  → 不入队（重试无用），直接向上抛让用户修。
- 调用成功但返回内容无法解析成 tags → 容错：记 warning、返回空 tags [](不崩、不入队死循环)。

模型：默认 claude-sonnet-4-6（当前 Sonnet；轻任务快且便宜）。可通过构造参数覆盖。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ..env import get_anthropic_api_key
from .base import EnricherConfigError, EnricherError, EnricherInterface

logger = logging.getLogger(__name__)

# 默认模型：claude-haiku-4-5（$1/$5 每百万 token）。生成标签是极轻任务，
# Haiku 质量足够、便宜快约 1/3，符合项目省成本理念。留作配置项，复杂任务可换 sonnet/opus。
# 注意：不存在 "claude-sonnet-5" 这个 ID；"5" 系列是 Fable 5 / Mythos 5，当前 Sonnet 为 claude-sonnet-4-6。
DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are a tagging assistant for a visual memory system. The user gives you "
    "text OCR-extracted from a whiteboard/document/screen (may contain OCR errors). "
    "Output ONLY a JSON array of 3-6 concise search tags (lowercase, specific nouns/topics/"
    "type like 'meeting-notes','budget','todo','architecture-diagram'). "
    "Always include at least one tag naming the content type "
    "(e.g. 'meeting-notes','slides','invoice','code-snippet','document','whiteboard'). "
    "No prose, no markdown, just the JSON array. If text is garbled/empty, return []."
)


def _strip_code_fence(text: str) -> str:
    """去掉可能包裹的 ```json ... ``` 或 ``` ... ``` 代码块。"""
    s = text.strip()
    if s.startswith("```"):
        # 去掉首行围栏（```json / ```）
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        # 去掉尾部围栏
        s = s.rstrip()
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _parse_tags(text: str, max_tags: int = 6) -> list[str]:
    """把模型输出解析成 tags 数组。解析失败容错返回 []（记 warning，不抛）。

    容忍两种形状：裸数组 `["a","b"]`，或（json_object 模式下）对象里包一个数组
    如 `{"tags": ["a","b"]}` —— 取对象里第一个 list 值。
    """
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("enrich: tags JSON 解析失败，返回空标签。原始输出=%r", text[:200])
        return []
    if isinstance(data, dict):
        # json_object 模式：模型把数组包在对象里，取第一个 list 值
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list):
        logger.warning("enrich: 模型未返回 JSON 数组（或对象内无数组），返回空标签。")
        return []
    tags: list[str] = []
    for item in data:
        if isinstance(item, str):
            t = item.strip().lower()
            if t and t not in tags:
                tags.append(t)
    return tags[:max_tags]


class CloudEnricher(EnricherInterface):
    """真 Claude API enricher。"""

    name = "cloud"  # 真实现，标签不带 mock: 前缀

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 300,
        max_tags: int = 6,
        client: Optional[Any] = None,   # 可注入（测试用）；否则惰性构造
        api_key: Optional[str] = None,  # 可显式传；否则从 env 读
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_tags = max_tags
        self._client = client
        self._api_key = api_key

    @property
    def client(self):
        """惰性构造 Anthropic 客户端。缺密钥时抛 EnricherConfigError。"""
        if self._client is None:
            import anthropic  # 局部 import：没装 SDK 也能 import 本模块

            key = self._api_key or get_anthropic_api_key()
            if not key:
                raise EnricherConfigError(
                    "未找到 ANTHROPIC_API_KEY。请在项目根 .env 写入 ANTHROPIC_API_KEY=...，"
                    "或在环境里 export（.env 已 gitignore）。"
                )
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def _build_user_prompt(self, ocr_text: str, metadata: dict[str, Any]) -> str:
        ts = metadata.get("timestamp", "unknown")
        conf = metadata.get("trigger_confidence", "unknown")
        text = ocr_text.strip() or "(empty)"
        return (
            f"Scene metadata: captured_at={ts}, gatekeeper_confidence={conf}, "
            f"first-launch scenario = useful-text screen (whiteboard/document/slides/"
            f"code screen/projector).\n\n"
            f"OCR text:\n{text}"
        )

    def enrich(self, ocr_text: str, metadata: dict[str, Any]) -> list[str]:
        import anthropic

        client = self.client  # 缺密钥 → EnricherConfigError（不入队）
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self._build_user_prompt(ocr_text, metadata)}],
            )
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ) as e:
            # 配置类错误：重试无用，抛 EnricherConfigError（不入队）
            raise EnricherConfigError(f"Claude API 配置错误（不重试）: {e}") from e
        except anthropic.APIError as e:
            # 瞬时错误（连接/限流/5xx/过载）：抛 EnricherError → 上层入 pending 重试
            raise EnricherError(f"Claude API 瞬时失败（可重试）: {e}") from e

        # 安全拒答：不是网络故障，也不该入队死循环 → 记 warning、返回空 tags
        if getattr(resp, "stop_reason", None) == "refusal":
            logger.warning("CloudEnricher: 模型安全拒答，返回空标签。")
            return []

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return _parse_tags(text, max_tags=self.max_tags)
