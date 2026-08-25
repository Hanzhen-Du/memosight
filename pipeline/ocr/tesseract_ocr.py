"""The real Tesseract engine, behind OCRInterface.

Dependencies:
- the Python package `pytesseract`, already installed in the venv;
- the `tesseract` system binary plus language data, installed with apt and needing sudo:
    sudo apt install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng

If the binary is missing, construction still succeeds so the module stays importable, and
calling ocr() raises a clear error instead. `tesseract_available()` probes for it in advance,
which is how the pipeline decides to fall back to StubOCR.
"""

from __future__ import annotations

import shutil
from typing import Optional

import pytesseract

from .base import ImageInput, OCRInterface, preprocess, preprocess_enhanced


def tesseract_available() -> bool:
    """Whether the tesseract binary is installed on this system."""
    if shutil.which("tesseract"):
        return True
    # pytesseract may also have an explicit path configured
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


class TesseractOCR(OCRInterface):
    """Tesseract-based OCR. Defaults to simplified Chinese plus English."""

    name = "tesseract"

    def __init__(
        self,
        lang: str = "chi_sim+eng",
        max_side: int = 1600,
        psm: int = 6,
        cmd: Optional[str] = None,
        enhance: bool = False,
    ):
        # lang is the tesseract language code. psm 6 assumes a single uniform block of text,
        # which suits whiteboards and documents.
        # enhance turns on the enhanced preprocessing (deskew, adaptive threshold, upscaling
        #   small text). It is OFF by default. Measured on 2026-07-06 across 10 Pexels
        #   positives, enhancement made clean screenshots and scans worse, not better:
        #   Tesseract already binarises, and forcing an adaptive threshold plus deskew plus
        #   upscaling introduces noise. Net negative. It may still help low-quality, angled or
        #   glare-heavy frames, so it stays available as opt-in pending a test on real captured
        #   frames (see the backlog in `docs/pipeline-architecture.md`).
        self.lang = lang
        self.max_side = max_side
        self.psm = psm
        self.enhance = enhance
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

    def ocr(self, image: ImageInput) -> str:
        if not tesseract_available():
            raise RuntimeError(
                "the tesseract binary is not installed. Run:\n"
                "  sudo apt install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng"
            )
        if self.enhance:
            img = preprocess_enhanced(image, max_side=self.max_side)
        else:
            img = preprocess(image, max_side=self.max_side)
        config = f"--psm {self.psm}"
        text = pytesseract.image_to_string(img, lang=self.lang, config=config)
        return text.strip()
