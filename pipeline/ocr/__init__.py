"""The OCR interface layer.

`OCRInterface` is the abstract base class. This phase implements the real engine
`TesseractOCR` and `StubOCR` for tests. On-phone or cloud OCR can be swapped in behind the same
interface later without touching the rest of the pipeline.
"""

from .base import OCRInterface, OCRResult
from .stub_ocr import StubOCR
from .tesseract_ocr import TesseractOCR, tesseract_available

__all__ = ["OCRInterface", "OCRResult", "StubOCR", "TesseractOCR", "tesseract_available"]
