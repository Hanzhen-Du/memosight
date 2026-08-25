"""OCR 抽象接口 + 共享的图像预处理。

接口约定：`ocr(image) -> str`。`image` 可以是文件路径(str/Path) 或 已解码的
numpy 灰度/彩色数组。返回识别出的纯文本（可能为空字符串）。

预处理策略（与守门员训练口径对齐）：加载图片先 resize 再处理、低分辨率灰度，
省内存、稳 OCR、贴近 Pi/守门员的低清灰度输入取向。
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
    """OCR 结果。text 为主输出；engine 记录来源引擎便于审计。"""

    text: str
    engine: str


def load_image(image: ImageInput) -> np.ndarray:
    """把路径或数组统一成 BGR/灰度 ndarray。"""
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {path}")
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"无法解码图片: {path}")
    return arr


def preprocess(image: ImageInput, max_side: int = 1600) -> np.ndarray:
    """加载 → 转灰度 → 按最长边 resize（仅缩小，不放大）。返回灰度 ndarray。"""
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


# ---- 可选增强预处理（deskew 纠斜 + 自适应二值化 + 小字放大）----
# 只会让 OCR 更好不更差；对真实头戴摄像头的轻微倾斜/光照也有用。全部保守实现，异常兜底。

def _upscale_if_small(gray: np.ndarray, min_side: int) -> np.ndarray:
    """图太小则放大到 min_side（小字放大，帮 Tesseract 认清）。"""
    h, w = gray.shape[:2]
    longest = max(h, w)
    if min_side and longest < min_side:
        scale = min_side / float(longest)
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )
    return gray


def estimate_skew_angle(gray: np.ndarray) -> float:
    """估计文本倾斜角（度）。归一化到 (-45,45]。文本像素太少则返回 0。"""
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.shape[0] < 50:
        return 0.0
    angle = float(cv2.minAreaRect(coords.astype(np.float32))[-1])
    # OpenCV 版本间 minAreaRect 角度值域不同，统一归一到 (-45,45]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return angle


def _deskew(gray: np.ndarray, max_correct_deg: float = 15.0) -> np.ndarray:
    """纠正小幅倾斜。只处理 0.5°~max_correct_deg 的角度（大角度多为误估，跳过）。"""
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.5 or abs(angle) > max_correct_deg:
        return gray
    h, w = gray.shape[:2]
    # 估计角为文本的 CCW 倾斜；纠正需反向旋转 -angle
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _binarize(gray: np.ndarray) -> np.ndarray:
    """自适应二值化（应对不均光照/眩光）。"""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )


def preprocess_enhanced(
    image: ImageInput, max_side: int = 1600, min_side: int = 1000
) -> np.ndarray:
    """增强预处理：基础(灰度+缩小) → 小字放大 → 纠斜 → 自适应二值化。返回二值 ndarray。

    任何一步出异常都兜底回退到基础灰度图，绝不因预处理让整条链崩。
    """
    gray = preprocess(image, max_side=max_side)
    try:
        gray = _upscale_if_small(gray, min_side)
        gray = _deskew(gray)
        return _binarize(gray)
    except cv2.error:
        return gray


class OCRInterface(ABC):
    """所有 OCR 引擎的抽象基类。"""

    #: 引擎名，用于审计/日志
    name: str = "abstract"

    @abstractmethod
    def ocr(self, image: ImageInput) -> str:
        """对单张图片做 OCR，返回文本。"""
        raise NotImplementedError

    def ocr_result(self, image: ImageInput) -> OCRResult:
        """带来源信息的便捷封装。"""
        return OCRResult(text=self.ocr(image), engine=self.name)
