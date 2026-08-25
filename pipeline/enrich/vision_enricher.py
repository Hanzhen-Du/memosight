"""VisionEnricher, path Y: send the image itself to a cloud multimodal model and get a
complete memory card back in one step.

How it differs from the existing enrichers, which is the entire reason this module exists:

    Path X (existing):    image -> local Tesseract OCR -> upload TEXT  -> DeepSeek/Claude produces tags only
    Path Y (this module): image -> resize -> base64     -> upload IMAGE -> Claude multimodal produces everything
                     {description, tags, extracted_text}

In other words, on path X what an image *is* depends entirely on whether OCR can read the
text. If OCR returns noise, the system is blind. Path Y lets the model look at the image
directly, so it may still understand images OCR failed on. That is the claim being tested.

This module is evaluation material and does not change the pipeline's default behaviour; the
default enricher is still DeepSeekEnricher. The architecture decision is still open.

Interface:
- The main method is `enrich_image(image, metadata) -> VisionCard`, the real interface of this
  module, returning the complete card content.
- It also implements `EnricherInterface.enrich(ocr_text, metadata) -> list[str]` as an adapter:
  take the image from `metadata["image_path"]`, call enrich_image, and return only the tags.
  That lets this class slot straight into the existing IngestService pipeline, so path Y's tags
  are comparable with path X's.

Error semantics are identical to CloudEnricher and DeepSeekEnricher:
- Transient (network, 429 rate limiting, 5xx, overload) raises EnricherError and the card is
  queued as pending for retry.
- Configuration (401 invalid key, 403 no permission, 404 wrong model id, 400 malformed request)
  raises EnricherConfigError and is not queued.
- A successful call whose output cannot be parsed returns an empty VisionCard with
  `parse_ok=False`, logging a warning rather than crashing.

Cost: images are resized so the longest side is at most 1024 before base64 encoding, which
controls both tokens and memory. Real token usage per call comes back in the API response's
`usage` field and is recorded in `VisionCard.usage`, so cost can be tallied.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

from ..env import get_anthropic_api_key
from .base import EnricherConfigError, EnricherError, EnricherInterface

logger = logging.getLogger(__name__)

# Multimodal model. claude-haiku-4-5 also supports vision, but this evaluation is about how
# well the image is understood, and Sonnet better represents the ceiling of multimodal
# capability that the argument rests on. Cost is still measured per call rather than estimated.
DEFAULT_MODEL = "claude-sonnet-4-6"

# Pricing in USD per million tokens, used only to convert measured token counts into a dollar
# estimate. Source: the official Anthropic pricing table ($3 input / $15 output per million
# tokens for this model). If pricing changes, update it here: every dollar figure in the
# reports is computed from these two constants.
PRICE_IN_PER_MTOK = 3.00
PRICE_OUT_PER_MTOK = 15.00

# Maximum image side. Anything larger is scaled down proportionally before base64 encoding.
# 1024 is the compromise between text staying legible and keeping the token count down.
DEFAULT_MAX_SIDE = 1024
JPEG_QUALITY = 85

SYSTEM_PROMPT = (
    "You are the perception module of a wearable visual memory assistant. "
    "The user gives you ONE photo captured by a head-mounted camera. "
    "Produce a searchable memory card for it.\n\n"
    "Reply with ONLY a JSON object, no prose and no markdown fences:\n"
    '{"description": "...", "tags": ["...", "..."], "extracted_text": "..."}\n\n'
    "Rules:\n"
    "- description: ONE sentence saying what this image is (English), concrete and specific.\n"
    "- tags: 3-6 concise lowercase search tags (nouns/topics/content-type, e.g. "
    "'slides','whiteboard','dashboard','textbook','code-screen','chemistry').\n"
    "  Always include one tag naming the content type.\n"
    "- extracted_text: the text visible in the image, verbatim, newline-separated. "
    "Use an empty string \"\" if there is no legible text. Do NOT invent text you cannot read.\n"
    "- If the image is unreadable or contains nothing useful, still return the object with "
    "your best-effort description and an empty tags array."
)

USER_INSTRUCTION = "Produce the memory card JSON for this image."

ImageInput = Union[str, Path, "np.ndarray"]


@dataclass
class VisionCard:
    """One path-Y result: the complete memory card content plus the real usage for that
    call."""

    description: str = ""
    tags: list[str] = field(default_factory=list)
    extracted_text: str = ""
    parse_ok: bool = False              # whether the model output parsed into a JSON object
    refusal: bool = False               # whether the model declined on safety grounds
    raw_output: str = ""                # the raw text output, for diagnosing parse failures
    usage: dict[str, int] = field(default_factory=dict)  # input/output tokens, as measured by the API
    image_bytes_sent: int = 0           # bytes actually sent after resize and JPEG encoding

    @property
    def has_content(self) -> bool:
        """Whether anything usable came back: a description or some tags."""
        return bool(self.description.strip() or self.tags)

    def cost_usd(self) -> float:
        """Dollar estimate for this call, from the measured token counts and the pricing
        constants above."""
        return (
            self.usage.get("input_tokens", 0) / 1e6 * PRICE_IN_PER_MTOK
            + self.usage.get("output_tokens", 0) / 1e6 * PRICE_OUT_PER_MTOK
        )


# ---------- image preprocessing ----------

def resize_and_encode(image: ImageInput, max_side: int = DEFAULT_MAX_SIDE) -> tuple[str, int]:
    """Read the image, scale the longest side down to at most max_side, JPEG encode, base64.

    Returns (base64 string, encoded byte count). The image is read only; the source file is
    never modified.
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"cannot read image: {image}")
    else:
        img = image
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError(f"JPEG encoding failed: {image}")
    raw = buf.tobytes()
    return base64.b64encode(raw).decode("ascii"), len(raw)


# ---------- output parsing ----------

def _strip_code_fence(text: str) -> str:
    """Strip a surrounding ```json ... ``` or ``` ... ``` code fence, if present."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        s = s.rstrip()
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _coerce_text(value: Any) -> str:
    """Normalise whatever the model returned, str, list or None, into a string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def parse_vision_output(text: str, max_tags: int = 6) -> tuple[bool, str, list[str], str]:
    """Parse model output into (parse_ok, description, tags, extracted_text).

    Tolerant by design; it never raises:
    - strip the code fence then json.loads. On failure, parse_ok is False and everything is
      empty.
    - if the top level is an array, take the first dict element, since the model occasionally
      wraps its output one level deeper.
    - a missing field or a wrong type degrades that field to empty and the rest is returned as
      normal.
    """
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("VisionEnricher: JSON parse failed, returning an empty card. raw output=%r", text[:200])
        return False, "", [], ""
    if isinstance(data, list):
        data = next((d for d in data if isinstance(d, dict)), None)
    if not isinstance(data, dict):
        logger.warning("VisionEnricher: the model did not return a JSON object; returning an empty card.")
        return False, "", [], ""

    description = _coerce_text(data.get("description"))
    extracted_text = _coerce_text(data.get("extracted_text"))

    tags: list[str] = []
    raw_tags = data.get("tags")
    if isinstance(raw_tags, str):          # the model occasionally returns a comma-separated string
        raw_tags = [t for t in raw_tags.split(",")]
    if isinstance(raw_tags, list):
        for item in raw_tags:
            if isinstance(item, str):
                t = item.strip().lower()
                if t and t not in tags:
                    tags.append(t)
    return True, description, tags[:max_tags], extracted_text


class VisionEnricher(EnricherInterface):
    """Path Y: image to a Claude multimodal model to a complete memory card."""

    name = "vision"  # a real implementation, so tags carry no mock: prefix

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 500,
        max_tags: int = 6,
        max_side: int = DEFAULT_MAX_SIDE,
        client: Optional[Any] = None,   # injectable for tests; otherwise constructed lazily
        api_key: Optional[str] = None,  # may be passed explicitly; otherwise read from the environment
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_tags = max_tags
        self.max_side = max_side
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

    def _build_user_text(self, metadata: dict[str, Any]) -> str:
        ts = metadata.get("timestamp", "unknown")
        conf = metadata.get("trigger_confidence", "unknown")
        return (
            f"Scene metadata: captured_at={ts}, gatekeeper_confidence={conf}, "
            f"first-launch scenario = useful-text screen (whiteboard/document/slides/"
            f"code screen/projector).\n\n{USER_INSTRUCTION}"
        )

    # ---- main method: image to a complete card ----
    def enrich_image(self, image: ImageInput, metadata: Optional[dict[str, Any]] = None) -> VisionCard:
        """Hand one image to the multimodal model and return the complete card content plus
        the measured token usage."""
        import anthropic

        metadata = metadata or {}
        client = self.client  # a missing key raises EnricherConfigError and is not queued
        b64, nbytes = resize_and_encode(image, max_side=self.max_side)

        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": self._build_user_text(metadata)},
                    ],
                }],
            )
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ) as e:
            raise EnricherConfigError(f"Claude multimodal API configuration error, not retried: {e}") from e
        except anthropic.APIError as e:
            raise EnricherError(f"Claude multimodal API transient failure, retryable: {e}") from e

        usage = {}
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = {
                "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
            }

        # A safety refusal is not a network failure and must not loop forever in the queue, so
        # log a warning and return an empty card
        if getattr(resp, "stop_reason", None) == "refusal":
            logger.warning("VisionEnricher: the model declined on safety grounds; returning an empty card.")
            return VisionCard(refusal=True, usage=usage, image_bytes_sent=nbytes)

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        ok, description, tags, extracted = parse_vision_output(text, max_tags=self.max_tags)
        return VisionCard(
            description=description,
            tags=tags,
            extracted_text=extracted,
            parse_ok=ok,
            raw_output=text,
            usage=usage,
            image_bytes_sent=nbytes,
        )

    # ---- adapter satisfying EnricherInterface, so this drops into the existing pipeline ----
    def enrich(self, ocr_text: str, metadata: dict[str, Any]) -> list[str]:
        """Interface adapter: take the image from metadata['image_path'], run enrich_image,
        and return only the tags.

        Note that `ocr_text` is not used at all on path Y. That is exactly where the two paths
        diverge.
        A missing image_path raises EnricherConfigError, since it is a caller error and
        retrying would not help.
        """
        image_path = (metadata or {}).get("image_path")
        if not image_path:
            raise EnricherConfigError(
                "VisionEnricher.enrich() requires metadata['image_path']: path Y sends the "
                "image, not text. To produce a complete card directly, call enrich_image()."
            )
        return self.enrich_image(image_path, metadata).tags
