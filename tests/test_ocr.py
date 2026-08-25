"""M2: OCR 接口 + Tesseract 真引擎 + StubOCR 测试。

- StubOCR 测试始终跑（无系统依赖）。
- TesseractOCR 测试仅在系统装了 tesseract 二进制时跑（否则 skip），
  合成一张写有英文文字的图，断言识别文本包含关键词。
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
    """白底黑字合成图（BGR）。size=(h, w)。"""
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
        self.assertEqual(gray.ndim, 2)              # 灰度
        self.assertEqual(max(gray.shape), 1600)     # 最长边被缩到 1600

    def test_no_upscale(self):
        img = make_text_image("HI", size=(100, 200))
        gray = preprocess(img, max_side=1600)
        self.assertEqual(gray.shape[:2], (100, 200))  # 小图不放大

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
        # 大角度多为误估，跳过（返回原图，形状不变）
        sk = self._skewed_text(30.0)
        self.assertEqual(_deskew(sk).shape, sk.shape)

    def test_upscales_small_image(self):
        small = make_text_image("HI", size=(80, 160))
        out = preprocess_enhanced(small, max_side=1600, min_side=1000)
        self.assertGreaterEqual(max(out.shape), 1000)  # 小图被放大

    def test_blank_image_does_not_crash(self):
        blank = np.full((200, 400), 255, np.uint8)
        out = preprocess_enhanced(blank)
        self.assertEqual(out.ndim, 2)


@unittest.skipUnless(tesseract_available(), "tesseract 二进制未安装")
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
        ocr = StubOCR(fixed_text="白板 会议纪要")
        self.assertEqual(ocr.ocr("anything.png"), "白板 会议纪要")

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
    tesseract_available(), "tesseract 二进制未安装（apt），跳过真引擎测试"
)
class TestTesseractOCR(unittest.TestCase):
    def test_reads_english_text(self):
        ocr = TesseractOCR(lang="eng")
        img = make_text_image("MEMOSIGHT")
        text = ocr.ocr(img).upper()
        # OCR 容错：至少认出核心词的主要字符
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
