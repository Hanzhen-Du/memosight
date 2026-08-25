"""StubOCR：测试/离线开发用的假 OCR。

不依赖任何系统二进制。行为可控：
- 传入 fixed_text → 恒定返回该文本；
- 传入 mapping{路径关键字: 文本} → 按图片路径匹配返回；
- 默认从图片文件名推断一句占位文本。

它让整个闭环（M3/M4/M5）在 tesseract 二进制落地前就能端到端跑通。
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
