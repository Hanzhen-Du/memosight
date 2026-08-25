"""DirectUploadMock —— 树莓派"直连云端"模式的 mock 传输。

直连模式下，Pi 直接把 payload 发给云端 enricher；这里用注入的 enricher 完成
tags 生成，组装成一张 status=done 的 memory card 返回（未入库）。

未来"经手机中转"的实现只需换成另一个 UploadInterface，上层管线不变。
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
