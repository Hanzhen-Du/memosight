"""Tesseract 真引擎（封装在 OCRInterface 之后）。

依赖：
- Python 包 `pytesseract`（已在 venv 安装）。
- 系统二进制 `tesseract` + 语言数据（apt 安装，需 sudo）：
    sudo apt install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng

若二进制缺失，构造时不报错（便于导入模块），调用 ocr() 时才抛出清晰错误；
`tesseract_available()` 可先行探测，管线据此回退到 StubOCR。
"""

from __future__ import annotations

import shutil
from typing import Optional

import pytesseract

from .base import ImageInput, OCRInterface, preprocess, preprocess_enhanced


def tesseract_available() -> bool:
    """系统是否装了 tesseract 二进制。"""
    if shutil.which("tesseract"):
        return True
    # pytesseract 也可能配了显式路径
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


class TesseractOCR(OCRInterface):
    """基于 Tesseract 的 OCR。中文简体+英文默认。"""

    name = "tesseract"

    def __init__(
        self,
        lang: str = "chi_sim+eng",
        max_side: int = 1600,
        psm: int = 6,
        cmd: Optional[str] = None,
        enhance: bool = False,
    ):
        # lang: tesseract 语言代码；psm 6 = 假设为一整块统一文本（白板/文档友好）
        # enhance: 增强预处理（deskew + 自适应二值化 + 小字放大）。**默认关**：
        #   2026-07-06 在 10 张 Pexels 正例图上实测，增强对"干净截图/扫描"反而变差
        #   （Tesseract 自带二值化，强加自适应阈值+纠斜+放大引入噪声），净负面。
        #   仅对低质/斜角/眩光的脏图可能有用，留作 opt-in，等真实脏帧再验证（见 `docs/pipeline-architecture.md` 的 backlog）。
        self.lang = lang
        self.max_side = max_side
        self.psm = psm
        self.enhance = enhance
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

    def ocr(self, image: ImageInput) -> str:
        if not tesseract_available():
            raise RuntimeError(
                "tesseract 二进制未安装。请先执行：\n"
                "  sudo apt install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng"
            )
        if self.enhance:
            img = preprocess_enhanced(image, max_side=self.max_side)
        else:
            img = preprocess(image, max_side=self.max_side)
        config = f"--psm {self.psm}"
        text = pytesseract.image_to_string(img, lang=self.lang, config=config)
        return text.strip()
