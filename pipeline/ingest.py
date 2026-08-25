"""Orchestration of the offline pending queue.

Three paths, tested with the switchable is_online() mock:
1. Online:   transport.upload(payload) generates tags, producing a complete memory card stored
             with status=done.
2. Offline:  ocr_text plus metadata is stored as pending, with tags left empty.
3. Recovery: fetch the pending rows in bulk, upload each to obtain tags, backfill them, set
             status=done and fill in enriched_at.

There is no rule-based fallback. When offline, tags are never fabricated from local rules; the
card simply waits in the queue for the real cloud call, mocked in this phase. A failed cloud
call (EnricherError) means the same thing as offline: this one stays pending.
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

    # ---- paths 1 and 2: accept one payload ----
    def ingest(self, payload: Payload) -> MemoryCard:
        """Store directly as done when online, queue as pending when offline. A cloud failure
        also falls back to queuing."""
        if self.connectivity.is_online():
            try:
                card = self.transport.upload(payload)   # already in the done state
            except EnricherError:
                return self._queue(payload)             # cloud failed, so queue it
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

    # ---- path 3: bulk backfill once connectivity returns ----
    def process_pending(self) -> list[MemoryCard]:
        """Backfill tags for each row in the pending queue. Returns empty when offline."""
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
                continue  # leave this one pending and try again next time
            updated = self.store.enrich_card(card.id, enriched.tags or [])
            completed.append(updated)
        return completed
