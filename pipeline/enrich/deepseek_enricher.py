"""DeepSeekEnricher: tagging via DeepSeek. This is the default implementation.

DeepSeek is reached through the OpenAI-compatible SDK:
    from openai import OpenAI
    client = OpenAI(api_key=..., base_url="https://api.deepseek.com")
    client.chat.completions.create(model="deepseek-v4-flash", messages=[...], ...)

It reuses CloudEnricher's system prompt and parsing logic (strip the code fence, json.loads,
tolerate failure by returning []). Real tags carry no `mock:` prefix. The key is read only
through pipeline.env.get_deepseek_api_key() from os.environ, reusing the existing python-dotenv
loading, and is never hardcoded.

Error semantics match the Claude version:
- Transient (network, 429 rate limiting, 5xx, timeout) raises EnricherError and the card is
  queued as pending for retry.
- Configuration (401 invalid key, 403 no permission, 404 wrong model id, 400 malformed request)
  raises EnricherConfigError and is not queued.
- A successful call whose content cannot be parsed into tags returns [], logging a warning
  rather than crashing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..env import get_deepseek_api_key
from .base import EnricherConfigError, EnricherError, EnricherInterface
# Reuse the Claude version's prompt and parsing logic, so there is a single source of truth
from .cloud_enricher import SYSTEM_PROMPT, _parse_tags

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v4-flash"  # cheap and fast for a light task. Note that deepseek-chat
                                     # was retired on 2026-07-24 and must not be used.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekEnricher(EnricherInterface):
    """The real enricher backed by DeepSeek through its OpenAI-compatible API."""

    name = "deepseek"  # a real implementation, so tags carry no mock: prefix

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 300,
        max_tags: int = 6,
        base_url: str = DEEPSEEK_BASE_URL,
        client: Optional[Any] = None,   # injectable for tests; otherwise constructed lazily
        api_key: Optional[str] = None,  # may be passed explicitly; otherwise read from the environment
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_tags = max_tags
        self.base_url = base_url
        self._client = client
        self._api_key = api_key

    @property
    def client(self):
        """Construct the OpenAI client, pointed at DeepSeek, lazily. Raises
        EnricherConfigError if the key is missing."""
        if self._client is None:
            from openai import OpenAI  # local import, so this module imports without the SDK installed

            key = self._api_key or get_deepseek_api_key()
            if not key:
                raise EnricherConfigError(
                    "DEEPSEEK_API_KEY not found. Put DEEPSEEK_API_KEY=... in a .env at the "
                    "project root, or export it in the environment. .env is gitignored."
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

        client = self.client  # a missing key raises EnricherConfigError and is not queued
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
            # Configuration error: retrying is pointless, so raise EnricherConfigError and do
            # not queue
            raise EnricherConfigError(f"DeepSeek API configuration error, not retried: {e}") from e
        except openai.APIError as e:
            # Transient error (connection, rate limit, 5xx, timeout): raise EnricherError so
            # the layer above queues it as pending and retries
            raise EnricherError(f"DeepSeek API transient failure, retryable: {e}") from e

        choice = resp.choices[0] if resp.choices else None
        # No content, for instance because of content filtering: return empty tags rather than
        # looping forever in the queue
        if choice is None or choice.message is None or choice.message.content is None:
            logger.warning("DeepSeekEnricher: no content returned, possibly filtered; returning empty tags.")
            return []
        return _parse_tags(choice.message.content, max_tags=self.max_tags)
