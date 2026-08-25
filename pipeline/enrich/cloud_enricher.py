"""CloudEnricher: tagging via the real Anthropic Claude API.

Sends OCR text plus metadata to Claude and returns a compact array of search tags. These are
semantically real tags with no `mock:` prefix. Use MockCloudEnricher for offline testing.

Keys are read only through pipeline.env.get_anthropic_api_key(), from os.environ, reusing the
existing infrastructure. They are never hardcoded. The client is constructed lazily: creating
an instance of this class needs no key, and one is only required on the first enrich() call.

Error semantics, matching the phase-one "offline or failed means pending" logic:
- Transient failures (network, 429 rate limiting, 5xx, overload) raise EnricherError, so the
  layer above queues the card as pending and retries once connectivity returns.
- Configuration errors (401 invalid key, 403 no permission, 404 wrong model id, 400 malformed
  request) raise EnricherConfigError. These are not queued, because retrying is pointless, and
  are raised to the caller to fix.
- A successful call whose content cannot be parsed into tags is tolerated: log a warning and
  return an empty tag list, so nothing crashes and nothing loops forever in the queue.

Model: defaults to a current Claude model, which is fast and cheap for a light task. It can be
overridden through the constructor.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ..env import get_anthropic_api_key
from .base import EnricherConfigError, EnricherError, EnricherInterface

logger = logging.getLogger(__name__)

# Default model: claude-haiku-4-5. Generating tags is a very light task, Haiku's quality is
# sufficient, and it is cheaper and faster, which fits the project's cost discipline. It stays
# a configuration option so a heavier model can be used for harder tasks.
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
    """Strip a surrounding ```json ... ``` or ``` ... ``` code fence, if present."""
    s = text.strip()
    if s.startswith("```"):
        # Drop the opening fence line (```json or ```)
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        # Drop the closing fence
        s = s.rstrip()
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _parse_tags(text: str, max_tags: int = 6) -> list[str]:
    """Parse model output into a tags array. On a parse failure this returns [] and logs a
    warning rather than raising.

    Two shapes are accepted: a bare array `["a","b"]`, and, in json_object mode, an object
    wrapping an array such as `{"tags": ["a","b"]}`, in which case the first list value in the
    object is used.
    """
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("enrich: could not parse tags JSON, returning empty tags. raw output=%r", text[:200])
        return []
    if isinstance(data, dict):
        # json_object mode: the model wraps the array in an object, so take the first list value
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list):
        logger.warning("enrich: the model did not return a JSON array, or the object contained no "
                       "array. Returning empty tags.")
        return []
    tags: list[str] = []
    for item in data:
        if isinstance(item, str):
            t = item.strip().lower()
            if t and t not in tags:
                tags.append(t)
    return tags[:max_tags]


class CloudEnricher(EnricherInterface):
    """The real Claude API enricher."""

    name = "cloud"  # a real implementation, so tags carry no mock: prefix

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 300,
        max_tags: int = 6,
        client: Optional[Any] = None,   # injectable for tests; otherwise constructed lazily
        api_key: Optional[str] = None,  # may be passed explicitly; otherwise read from the environment
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_tags = max_tags
        self._client = client
        self._api_key = api_key

    @property
    def client(self):
        """Construct the Anthropic client lazily, raising EnricherConfigError if the key is
        missing."""
        if self._client is None:
            import anthropic  # local import, so this module imports without the SDK installed

            key = self._api_key or get_anthropic_api_key()
            if not key:
                raise EnricherConfigError(
                    "ANTHROPIC_API_KEY not found. Put ANTHROPIC_API_KEY=... in a .env at the "
                    "project root, or export it in the environment. .env is gitignored."
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

        client = self.client  # a missing key raises EnricherConfigError and is not queued
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
            # Configuration error: retrying is pointless, so raise EnricherConfigError and do
            # not queue
            raise EnricherConfigError(f"Claude API configuration error, not retried: {e}") from e
        except anthropic.APIError as e:
            # Transient error (connection, rate limit, 5xx, overload): raise EnricherError so
            # the layer above queues it as pending and retries
            raise EnricherError(f"Claude API transient failure, retryable: {e}") from e

        # A safety refusal is not a network failure and must not loop forever in the queue, so
        # log a warning and return empty tags
        if getattr(resp, "stop_reason", None) == "refusal":
            logger.warning("CloudEnricher: the model declined on safety grounds; returning empty tags.")
            return []

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return _parse_tags(text, max_tags=self.max_tags)
