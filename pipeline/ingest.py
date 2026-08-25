"""断网待处理队列的核心编排。

三条路径（用可切换的 is_online() mock 测试）：
1. 联网      → transport.upload(payload) 生成 tags → 完整 memory card → status=done 存库。
2. 断网      → {ocr_text + 元数据} 存 pending（tags 暂空）。
3. 恢复联网  → 批量取 pending → 逐条 upload 补 tags → 回填 → status=done，填 enriched_at。

不做规则降级：断网时绝不用本地规则伪造 tags，只入队等待真正（本阶段 mock）的云端。
云端调用失败（EnricherError）等价于"这条先留在 pending"。
"""

from __future__ import annotations

from .connectivity import Connectivity
from .db import CardStore
from .enrich.base import EnricherError
from .models import MemoryCard
from .packaging import Payload, build_payload
from .transport.base import UploadInterface
from . import config


class IngestService:
    def __init__(
        self,
        store: CardStore,
        transport: UploadInterface,
        connectivity: Connectivity,
    ):
        self.store = store
        self.transport = transport
        self.connectivity = connectivity

    # ---- 路径 1 & 2：接收一个 payload ----
    def ingest(self, payload: Payload) -> MemoryCard:
        """联网直存(done) / 断网入队(pending)。云端失败也回退入队。"""
        if self.connectivity.is_online():
            try:
                card = self.transport.upload(payload)   # 已是 done 态
            except EnricherError:
                return self._queue(payload)             # 云端挂了 → 入队
            self.store.insert(card)
            return card
        return self._queue(payload)

    def _queue(self, payload: Payload) -> MemoryCard:
        card = MemoryCard(
            timestamp=payload.timestamp,
            trigger_confidence=payload.trigger_confidence,
            ocr_text=payload.ocr_text,
            tags=None,
            raw_image_policy=payload.raw_image_policy,
            status=config.STATUS_PENDING,
        )
        self.store.insert(card)
        return card

    # ---- 路径 3：恢复联网后批量补传 ----
    def process_pending(self) -> list[MemoryCard]:
        """把 pending 队列逐条补 tags 回填。断网时直接返回空。"""
        if not self.connectivity.is_online():
            return []
        completed: list[MemoryCard] = []
        for card in self.store.list_pending():
            payload = build_payload(
                ocr_text=card.ocr_text,
                timestamp=card.timestamp,
                trigger_confidence=card.trigger_confidence,
                raw_image_policy=card.raw_image_policy,
            )
            try:
                enriched = self.transport.upload(payload)
            except EnricherError:
                continue  # 这条留在 pending，下次再试
            updated = self.store.enrich_card(card.id, enriched.tags or [])
            completed.append(updated)
        return completed
