"""The OCR abstract interface plus shared image preprocessing.

Interface contract: `ocr(image) -> str`. `image` may be a file path (str or Path) or an already
decoded numpy array, greyscale or colour. It returns the recognised plain text, possibly an
empty string.

Preprocessing strategy, aligned with how the gatekeeper is trained: resize on load before doing
anything else, and work in low-resolution greyscale. That saves memory, steadies OCR, and stays
close to the low-resolution greyscale input the Pi and the gatekeeper use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np

ImageInput = Union[str, Path, np.ndarray]


@dataclass
class OCRResult:
    """An OCR result. text is the main output; engine records which engine produced it, for
    auditing."""

    text: str
    engine: str


def load_image(image: ImageInput) -> np.ndarray:
    """Normalise a path or an array into a BGR or greyscale ndarray."""
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"image does not exist: {path}")
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"cannot decode image: {path}")
    return arr


def preprocess(image: ImageInput, max_side: int = 1600) -> np.ndarray:
    """Load, convert to greyscale, and resize by the longest side. Shrinks only, never
    enlarges. Returns a greyscale ndarray."""
    arr = load_image(image)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr
    h, w = gray.shape[:2]
    longest = max(h, w)
    if max_side and longest > max_side:
        scale = max_side / float(longest)
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return gray


# ---- Optional enhanced preprocessing: deskew, adaptive thresholding, upscaling small text ----
# Intended to help OCR and never hurt it, and useful against the slight tilt and uneven lighting
# of a real head-mounted camera. Every step is implemented conservatively with a fallback on
# exception.
# Measured result: it is net negative on clean images, so it is off by default. See
# tesseract_ocr.py.

def _upscale_if_small(gray: np.ndarray, min_side: int) -> np.ndarray:
    """Upscale to min_side when the image is small, which helps Tesseract read small text."""
    h, w = gray.shape[:2]
    longest = max(h, w)
    if min_side and longest < min_side:
        scale = min_side / float(longest)
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )
    return gray


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate the text skew angle in degrees, normalised to (-45, 45]. Returns 0 when there
    are too few text pixels."""
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.shape[0] < 50:
        return 0.0
    angle = float(cv2.minAreaRect(coords.astype(np.float32))[-1])
    # minAreaRect's angle range differs between OpenCV versions, so normalise to (-45, 45]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return angle


def _deskew(gray: np.ndarray, max_correct_deg: float = 15.0) -> np.ndarray:
    """Correct a small skew. Only angles between 0.5 degrees and max_correct_deg are handled;
    larger angles are usually mis-estimates and are skipped."""
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.5 or abs(angle) > max_correct_deg:
        return gray
    h, w = gray.shape[:2]
    # The estimated angle is the text's counter-clockwise tilt, so correcting means rotating by -angle
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive thresholding, for uneven lighting and glare."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )


def preprocess_enhanced(
    image: ImageInput, max_side: int = 1600, min_side: int = 1000
) -> np.ndarray:
    """Enhanced preprocessing: the base greyscale and downscale, then upscale small text,
    deskew, and adaptive threshold. Returns a binary ndarray.

    Any step that raises falls back to the base greyscale image, so preprocessing can never
    bring down the whole chain.
    """
    gray = preprocess(image, max_side=max_side)
    try:
        gray = _upscale_if_small(gray, min_side)
        gray = _deskew(gray)
        return _binarize(gray)
    except cv2.error:
        return gray


class OCRInterface(ABC):
    """Abstract base class for every OCR engine."""

    #: Engine name, used in auditing and logs
    name: str = "abstract"

    @abstractmethod
    def ocr(self, image: ImageInput) -> str:
        """Run OCR on one image and return the text."""
        raise NotImplementedError

    def ocr_result(self, image: ImageInput) -> OCRResult:
        """Convenience wrapper that also carries the source engine."""
        return OCRResult(text=self.ocr(image), engine=self.name)
