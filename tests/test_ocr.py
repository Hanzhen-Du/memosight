"""M2: tests for the OCR interface, the real Tesseract engine and StubOCR.

- The StubOCR tests always run; they have no system dependency.
- The TesseractOCR tests run only when the tesseract binary is installed, and skip otherwise.
  They synthesise an image containing English text and assert the recognised text contains the
  expected keywords.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipeline.ocr import StubOCR, TesseractOCR, tesseract_available
from pipeline.ocr.base import (
    OCRInterface, preprocess, preprocess_enhanced, estimate_skew_angle, _deskew,
)


def make_text_image(text: str, size=(240, 900)) -> np.ndarray:
    """Synthesise a BGR image with black text on white. size is (h, w)."""
    h, w = size
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(
        img, text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
        3.0, (0, 0, 0), 6, cv2.LINE_AA,
    )
    return img


class TestPreprocess(unittest.TestCase):
    def test_resizes_down_and_grayscales(self):
        img = make_text_image("HELLO", size=(2000, 4000))
        gray = preprocess(img, max_side=1600)
        self.assertEqual(gray.ndim, 2)              # greyscale
        self.assertEqual(max(gray.shape), 1600)     # the longest side was reduced to 1600

    def test_no_upscale(self):
        img = make_text_image("HI", size=(100, 200))
        gray = preprocess(img, max_side=1600)
        self.assertEqual(gray.shape[:2], (100, 200))  # small images are not enlarged

    def test_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            preprocess("/no/such/file.png")


class TestEnhancedPreprocess(unittest.TestCase):
    def _skewed_text(self, deg):
        img = np.full((300, 900), 255, np.uint8)
        cv2.putText(img, "THE QUICK BROWN FOX", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)
        m = cv2.getRotationMatrix2D((450, 150), deg, 1.0)
        return cv2.warpAffine(img, m, (900, 300), borderValue=255)

    def test_returns_2d_uint8(self):
        out = preprocess_enhanced(make_text_image("HELLO"))
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.dtype, np.uint8)

    def test_deskew_reduces_angle(self):
        for deg in (8.0, -6.0, 12.0):
            sk = self._skewed_text(deg)
            before = abs(estimate_skew_angle(sk))
            after = abs(estimate_skew_angle(_deskew(sk)))
            self.assertLess(after, before, f"deskew failed for {deg} deg")

    def test_deskew_skips_large_angle(self):
        # Large angles are usually mis-estimates and are skipped, returning the image unchanged
        sk = self._skewed_text(30.0)
        self.assertEqual(_deskew(sk).shape, sk.shape)

    def test_upscales_small_image(self):
        small = make_text_image("HI", size=(80, 160))
        out = preprocess_enhanced(small, max_side=1600, min_side=1000)
        self.assertGreaterEqual(max(out.shape), 1000)  # small images are upscaled

    def test_blank_image_does_not_crash(self):
        blank = np.full((200, 400), 255, np.uint8)
        out = preprocess_enhanced(blank)
        self.assertEqual(out.ndim, 2)


@unittest.skipUnless(tesseract_available(), "the tesseract binary is not installed")
class TestTesseractEnhance(unittest.TestCase):
    def test_reads_slightly_rotated_text(self):
        img = np.full((300, 1000, 3), 255, np.uint8)
        cv2.putText(img, "ROADMAP", (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 6)
        m = cv2.getRotationMatrix2D((500, 150), 7.0, 1.0)
        rotated = cv2.warpAffine(img, m, (1000, 300), borderValue=(255, 255, 255))
        text = TesseractOCR(lang="eng", enhance=True).ocr(rotated).upper()
        self.assertIn("ROADMAP", text.replace(" ", ""))


class TestStubOCR(unittest.TestCase):
    def test_is_ocr_interface(self):
        self.assertIsInstance(StubOCR(), OCRInterface)

    def test_fixed_text(self):
        ocr = StubOCR(fixed_text="whiteboard meeting notes")
        self.assertEqual(ocr.ocr("anything.png"), "whiteboard meeting notes")

    def test_mapping(self):
        ocr = StubOCR(mapping={"meeting": "meeting notes", "menu": "lunch menu"})
        self.assertEqual(ocr.ocr("/x/meeting_01.png"), "meeting notes")
        self.assertEqual(ocr.ocr("/x/menu_02.png"), "lunch menu")

    def test_default_from_stem(self):
        ocr = StubOCR()
        self.assertIn("whiteboard", ocr.ocr("/frames/whiteboard.png"))

    def test_ocr_result_carries_engine(self):
        res = StubOCR(fixed_text="hi").ocr_result("a.png")
        self.assertEqual(res.text, "hi")
        self.assertEqual(res.engine, "stub")


@unittest.skipUnless(
    tesseract_available(), "the tesseract binary is not installed via apt; skipping the real-engine tests"
)
class TestTesseractOCR(unittest.TestCase):
    def test_reads_english_text(self):
        ocr = TesseractOCR(lang="eng")
        img = make_text_image("MEMOSIGHT")
        text = ocr.ocr(img).upper()
        # OCR is imperfect, so require only that the main characters of the key words are read
        self.assertIn("MEMOSIGHT", text.replace(" ", ""))

    def test_reads_from_file_path(self):
        ocr = TesseractOCR(lang="eng")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "board.png"
            cv2.imwrite(str(p), make_text_image("ROADMAP"))
            text = ocr.ocr(p).upper()
            self.assertIn("ROADMAP", text.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
