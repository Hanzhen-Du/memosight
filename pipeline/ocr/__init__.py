"""OCR 接口层。

抽象基类 `OCRInterface`；本阶段实现真引擎 `TesseractOCR` 与测试用 `StubOCR`。
未来可在同一接口后换手机/云端 OCR，管线其余部分无需改动。
"""

from .base import OCRInterface, OCRResult
from .stub_ocr import StubOCR
from .tesseract_ocr import TesseractOCR, tesseract_available

__all__ = ["OCRInterface", "OCRResult", "StubOCR", "TesseractOCR", "tesseract_available"]
