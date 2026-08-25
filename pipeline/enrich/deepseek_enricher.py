"""DeepSeekEnricher —— 用 DeepSeek 打标（默认实现）。

DeepSeek 走 OpenAI 兼容 SDK：
    from openai import OpenAI
    client = OpenAI(api_key=..., base_url="https://api.deepseek.com")
    client.chat.completions.create(model="deepseek-v4-flash", messages=[...], ...)

复用 CloudEnricher(Claude) 那套 system prompt 与解析逻辑（strip 围栏 + json.loads +
容错返回 []）。真标签**不带** `mock:` 前缀。密钥只经 pipeline.env.get_deepseek_api_key()
从 os.environ 读（复用现有 python-dotenv 加载），**绝不硬编码**。

错误语义与 Claude 版一致：
- 瞬时（网络 / 限流 429 / 5xx / 超时）→ EnricherError → 入 pending 重试。
- 配置（密钥无效 401 / 无权限 403 / 模型 ID 错误 404 / 请求非法 400）→ EnricherConfigError → 不入队。
- 调用成功但无法解析成 tags → 返回 []（记 warning，不崩）。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..env import get_deepseek_api_key
from .base import EnricherConfigError, EnricherError, EnricherInterface
# 复用 Claude 版的 prompt 与解析逻辑（单一真源）
from .cloud_enricher import SYSTEM_PROMPT, _parse_tags

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v4-flash"  # 轻任务、便宜快。注意：deepseek-chat 2026-07-24 退役，勿用。
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekEnricher(EnricherInterface):
    """基于 DeepSeek（OpenAI 兼容 API）的真 enricher。"""

    name = "deepseek"  # 真实现，标签不带 mock: 前缀

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 300,
        max_tags: int = 6,
        base_url: str = DEEPSEEK_BASE_URL,
        client: Optional[Any] = None,   # 可注入（测试用）；否则惰性构造
        api_key: Optional[str] = None,  # 可显式传；否则从 env 读
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_tags = max_tags
        self.base_url = base_url
        self._client = client
        self._api_key = api_key

    @property
    def client(self):
        """惰性构造 OpenAI 客户端（指向 DeepSeek）。缺密钥时抛 EnricherConfigError。"""
        if self._client is None:
            from openai import OpenAI  # 局部 import：没装 SDK 也能 import 本模块

            key = self._api_key or get_deepseek_api_key()
            if not key:
                raise EnricherConfigError(
                    "未找到 DEEPSEEK_API_KEY。请在项目根 .env 写入 DEEPSEEK_API_KEY=...，"
                    "或在环境里 export（.env 已 gitignore）。"
                )
            self._client = OpenAI(api_key=key, base_url=self.base_url)
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
        import openai

        client = self.client  # 缺密钥 → EnricherConfigError（不入队）
        try:
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_user_prompt(ocr_text, metadata)},
                ],
            )
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.BadRequestError,
        ) as e:
            # 配置类错误：重试无用，抛 EnricherConfigError（不入队）
            raise EnricherConfigError(f"DeepSeek API 配置错误（不重试）: {e}") from e
        except openai.APIError as e:
            # 瞬时错误（连接/限流/5xx/超时）：抛 EnricherError → 上层入 pending 重试
            raise EnricherError(f"DeepSeek API 瞬时失败（可重试）: {e}") from e

        choice = resp.choices[0] if resp.choices else None
        # 内容过滤等导致无内容 → 空标签（不入队死循环）
        if choice is None or choice.message is None or choice.message.content is None:
            logger.warning("DeepSeekEnricher: 无返回内容（可能被过滤），返回空标签。")
            return []
        return _parse_tags(choice.message.content, max_tags=self.max_tags)
