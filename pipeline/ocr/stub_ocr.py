"""StubOCR: a fake OCR for tests and offline development.

It depends on no system binary and its behaviour is controllable:
- pass fixed_text to always return that text;
- pass a mapping of {path fragment: text} to return text based on the image path;
- by default, derive a placeholder sentence from the image filename.

This lets the whole loop (M3 to M5) run end to end before the tesseract binary is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import ImageInput, OCRInterface


class StubOCR(OCRInterface):
    name = "stub"

    def __init__(self, fixed_text: Optional[str] = None, mapping: Optional[dict] = None):
        self.fixed_text = fixed_text
        self.mapping = mapping or {}

    def ocr(self, image: ImageInput) -> str:
        if self.fixed_text is not None:
            return self.fixed_text
        if isinstance(image, np.ndarray):
            return "stub ocr text from ndarray"
        key = str(image)
        for frag, text in self.mapping.items():
            if frag in key:
                return text
        return f"stub ocr text for {Path(key).stem}"
