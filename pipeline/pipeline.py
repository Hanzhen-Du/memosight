"""端到端编排：串起全数据流。

守门员触发(mock) → 高清帧(测试图) → 本地 OCR → 打包 → ingest(联网直存/断网入队)
→ 按 raw_image_policy 处理原始帧 → 返回 memory card。

`build_pipeline()` 组装默认组件：OCR 优先用 Tesseract（装了二进制才用），
否则回退 StubOCR，保证在二进制缺失时闭环仍可演示。
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
        """跑一次完整捕捉。守门员不触发则返回 None。"""
        policy = raw_image_policy or self.cfg.raw_image_policy
        ts = timestamp or utc_now_iso()

        # 1. 守门员（mock 触发信号）
        signal = self.gatekeeper.trigger(trigger_confidence)
        if not signal.should_record:
            return None

        # 2. 高清抓帧（测试图替代）
        frame_path = grab_frame(source_image, self.cfg.frames_dir, ts=ts)
        try:
            # 3. 本地 OCR
            text = self.ocr.ocr(frame_path)
            # 4. 打包 {ocr_text + 元数据}
            payload = build_payload(
                ocr_text=text,
                timestamp=ts,
                trigger_confidence=signal.confidence,
                raw_image_policy=policy,
            )
            # 5. ingest：联网直存 / 断网入队
            card = self.ingest.ingest(payload)
        finally:
            # 6. 隐私：按策略处理原始帧（即便出错也执行）
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
    """组装默认管线。OCR 未指定时：有 tesseract 二进制→TesseractOCR，否则→StubOCR。"""
    load_env()  # 从项目根 .env 加载环境变量/密钥（供未来真 enricher 用）
    cfg = cfg or config_mod.default_config()
    cfg.ensure_dirs()
    store = CardStore(cfg.db_path)
    if ocr is None:
        if tesseract_available():
            ocr = TesseractOCR(lang=cfg.ocr_lang, max_side=cfg.ocr_max_side)
        else:
            ocr = StubOCR()
    # 默认用 mock（离线安全，保证测试/无密钥环境可跑）；CLI 显式传真 CloudEnricher。
    enricher = enricher or MockCloudEnricher()
    transport = DirectUploadMock(enricher)
    connectivity = connectivity or ConnectivityMock(online=True)
    ingest = IngestService(store, transport, connectivity)
    return MemoSightPipeline(store, ocr, ingest, cfg)
