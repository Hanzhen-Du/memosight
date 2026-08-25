"""Packaging: assemble OCR text plus metadata into a payload that can be uploaded or queued.

payload = ocr_text plus metadata (timestamp, trigger_confidence, raw_image_policy)

This is the part the gatekeeper and OCR provide for free. tags are not in the payload; they are
added after the cloud enricher generates them, which completes the memory card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import config


@dataclass
class Payload:
    """Packaged data awaiting enrichment; it has no tags yet."""

    ocr_text: str
    timestamp: str
    trigger_confidence: float
    raw_image_policy: str = config.POLICY_DELETE

    def metadata(self) -> dict[str, Any]:
        """The metadata handed to the enricher, excluding ocr_text itself."""
        return {
            "timestamp": self.timestamp,
            "trigger_confidence": self.trigger_confidence,
            "raw_image_policy": self.raw_image_policy,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"ocr_text": self.ocr_text, **self.metadata()}


def build_payload(
    ocr_text: str,
    timestamp: str,
    trigger_confidence: float,
    raw_image_policy: str = config.POLICY_DELETE,
) -> Payload:
    """Assemble a payload, validating raw_image_policy."""
    if raw_image_policy not in config.VALID_RAW_IMAGE_POLICIES:
        raise ValueError(f"invalid raw_image_policy: {raw_image_policy!r}")
    return Payload(
        ocr_text=ocr_text,
        timestamp=timestamp,
        trigger_confidence=trigger_confidence,
        raw_image_policy=raw_image_policy,
    )
