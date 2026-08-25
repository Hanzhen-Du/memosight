"""End-to-end orchestration: the whole data flow in one place.

Gatekeeper trigger (mocked) -> full-resolution frame (a test image) -> local OCR -> packaging
-> ingest (store directly when online, queue when offline) -> apply raw_image_policy to the raw
frame -> return the memory card.

`build_pipeline()` assembles the default components. OCR prefers Tesseract when the binary is
installed and otherwise falls back to StubOCR, so the loop still demonstrates end to end
without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import config as config_mod
from .env import load_env
from .capture import MockGatekeeper, grab_frame
from .connectivity import Connectivity, ConnectivityMock
from .db import CardStore
from .enrich import MockCloudEnricher
from .enrich.base import EnricherInterface
from .ingest import IngestService
from .models import MemoryCard, utc_now_iso
from .ocr.base import OCRInterface
from .ocr.stub_ocr import StubOCR
from .ocr.tesseract_ocr import TesseractOCR, tesseract_available
from .packaging import build_payload
from .privacy import apply_raw_image_policy
from .transport import DirectUploadMock


class MemoSightPipeline:
    def __init__(
        self,
        store: CardStore,
        ocr: OCRInterface,
        ingest: IngestService,
        cfg: config_mod.Config,
        gatekeeper: Optional[MockGatekeeper] = None,
    ):
        self.store = store
        self.ocr = ocr
        self.ingest = ingest
        self.cfg = cfg
        self.gatekeeper = gatekeeper or MockGatekeeper()

    def capture(
        self,
        source_image: Path,
        trigger_confidence: float = 0.9,
        timestamp: Optional[str] = None,
        raw_image_policy: Optional[str] = None,
    ) -> Optional[MemoryCard]:
        """Run one full capture. Returns None if the gatekeeper does not fire."""
        policy = raw_image_policy or self.cfg.raw_image_policy
        ts = timestamp or utc_now_iso()

        # 1. gatekeeper (mocked trigger signal)
        signal = self.gatekeeper.trigger(trigger_confidence)
        if not signal.should_record:
            return None

        # 2. full-resolution grab (a test image stands in)
        frame_path = grab_frame(source_image, self.cfg.frames_dir, ts=ts)
        try:
            # 3. local OCR
            text = self.ocr.ocr(frame_path)
            # 4. package ocr_text plus metadata
            payload = build_payload(
                ocr_text=text,
                timestamp=ts,
                trigger_confidence=signal.confidence,
                raw_image_policy=policy,
            )
            # 5. ingest: store directly when online, queue when offline
            card = self.ingest.ingest(payload)
        finally:
            # 6. privacy: apply the policy to the raw frame, even if something above failed
            apply_raw_image_policy(frame_path, policy, self.cfg.cache_dir)
        return card

    def process_pending(self) -> list[MemoryCard]:
        return self.ingest.process_pending()

    def close(self) -> None:
        self.store.close()


def build_pipeline(
    cfg: Optional[config_mod.Config] = None,
    connectivity: Optional[Connectivity] = None,
    ocr: Optional[OCRInterface] = None,
    enricher: Optional[EnricherInterface] = None,
) -> MemoSightPipeline:
    """Assemble the default pipeline. When no OCR is given, use TesseractOCR if the binary is
    present, otherwise StubOCR."""
    load_env()  # load environment variables and keys from the project-root .env, for the real enricher
    cfg = cfg or config_mod.default_config()
    cfg.ensure_dirs()
    store = CardStore(cfg.db_path)
    if ocr is None:
        if tesseract_available():
            ocr = TesseractOCR(lang=cfg.ocr_lang, max_side=cfg.ocr_max_side)
        else:
            ocr = StubOCR()
    # Default to the mock, which is offline-safe so tests and keyless environments still run.
    # The CLI passes a real enricher explicitly.
    enricher = enricher or MockCloudEnricher()
    transport = DirectUploadMock(enricher)
    connectivity = connectivity or ConnectivityMock(online=True)
    ingest = IngestService(store, transport, connectivity)
    return MemoSightPipeline(store, ocr, ingest, cfg)
