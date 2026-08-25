"""M5: end-to-end smoke tests.

Mocked trigger, test image, StubOCR (so this runs without the tesseract binary), mocked
enrichment, store, then recall. Also covers the offline-queue to recovery-backfill path end to
end, and the delete and cache behaviour of raw_image_policy.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipeline import config as config_mod
from pipeline.connectivity import ConnectivityMock
from pipeline.ocr import StubOCR
from pipeline.pipeline import build_pipeline


def make_config(tmpdir: Path) -> config_mod.Config:
    return config_mod.Config(
        db_path=tmpdir / "e2e.db",
        frames_dir=tmpdir / "frames",
        cache_dir=tmpdir / "cache",
    )


def write_test_image(path: Path, text="WHITEBOARD NOTES") -> Path:
    img = np.full((200, 800, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    cv2.imwrite(str(path), img)
    return path


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cfg = make_config(self.dir)
        self.img = write_test_image(self.dir / "src.png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_online_capture_and_recall(self):
        pipe = build_pipeline(
            cfg=self.cfg,
            connectivity=ConnectivityMock(online=True),
            ocr=StubOCR(fixed_text="quarterly plan whiteboard roadmap"),
        )
        try:
            card = pipe.capture(self.img, trigger_confidence=0.92)
            self.assertIsNotNone(card)
            self.assertEqual(card.status, "done")
            self.assertTrue(card.tags)                 # mock tags were filled in
            self.assertIsNotNone(card.enriched_at)
            # Recall works: searching a keyword from the OCR text finds it
            hits = pipe.store.search("roadmap")
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].id, card.id)
        finally:
            pipe.close()

    def test_gatekeeper_no_trigger_records_nothing(self):
        pipe = build_pipeline(cfg=self.cfg, ocr=StubOCR(fixed_text="x"))
        try:
            card = pipe.capture(self.img, trigger_confidence=0.1)  # below the 0.5 threshold
            self.assertIsNone(card)
            self.assertEqual(pipe.store.count(), 0)
        finally:
            pipe.close()

    def test_offline_then_recovery_e2e(self):
        conn = ConnectivityMock(online=False)
        pipe = build_pipeline(cfg=self.cfg, connectivity=conn,
                              ocr=StubOCR(fixed_text="whiteboard captured while offline"))
        try:
            card = pipe.capture(self.img, trigger_confidence=0.88)
            self.assertEqual(card.status, "pending")
            self.assertIsNone(card.tags)
            self.assertEqual(pipe.store.count("pending"), 1)

            conn.go_online()
            done = pipe.process_pending()
            self.assertEqual(len(done), 1)
            self.assertEqual(pipe.store.count("pending"), 0)
            self.assertEqual(pipe.store.count("done"), 1)
            self.assertTrue(pipe.store.get(card.id).tags)
        finally:
            pipe.close()

    def test_raw_image_deleted_by_default(self):
        pipe = build_pipeline(cfg=self.cfg, ocr=StubOCR(fixed_text="t"))
        try:
            pipe.capture(self.img, trigger_confidence=0.9)  # the policy defaults to delete
            frames = list(Path(self.cfg.frames_dir).glob("frame_*"))
            self.assertEqual(frames, [])                    # the raw frame was deleted
            cached = list(Path(self.cfg.cache_dir).glob("*"))
            self.assertEqual(cached, [])
        finally:
            pipe.close()

    def test_raw_image_cached_when_policy_cache(self):
        pipe = build_pipeline(cfg=self.cfg, ocr=StubOCR(fixed_text="t"))
        try:
            pipe.capture(self.img, trigger_confidence=0.9, raw_image_policy="cache")
            cached = list(Path(self.cfg.cache_dir).glob("frame_*"))
            self.assertEqual(len(cached), 1)                # one frame was kept in the cache
            frames = list(Path(self.cfg.frames_dir).glob("frame_*"))
            self.assertEqual(frames, [])                    # and moved out of frames_dir
        finally:
            pipe.close()


if __name__ == "__main__":
    unittest.main()
