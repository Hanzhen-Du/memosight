"""DirectUploadMock: a mock transport for the Pi uploading directly to the cloud.

In direct mode the Pi sends the payload straight to the cloud enricher. Here the injected
enricher generates the tags and the result is assembled into a memory card with status=done,
which is returned rather than persisted.

A phone-relay implementation later is just another UploadInterface; the pipeline above is
unchanged.
"""

from __future__ import annotations

from ..enrich.base import EnricherInterface
from ..models import MemoryCard, utc_now_iso
from ..packaging import Payload
from .. import config
from .base import UploadInterface


class DirectUploadMock(UploadInterface):
    name = "direct-mock"

    def __init__(self, enricher: EnricherInterface):
        self.enricher = enricher

    def upload(self, payload: Payload) -> MemoryCard:
        tags = self.enricher.enrich(payload.ocr_text, payload.metadata())
        return MemoryCard(
            timestamp=payload.timestamp,
            trigger_confidence=payload.trigger_confidence,
            ocr_text=payload.ocr_text,
            tags=list(tags),
            raw_image_policy=payload.raw_image_policy,
            status=config.STATUS_DONE,
            enriched_at=utc_now_iso(),
        )
