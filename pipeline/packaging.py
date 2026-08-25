"""打包逻辑：把 OCR 文本 + 元数据组装成一个可上传/可入队的 payload。

payload = { ocr_text + 元数据(timestamp / trigger_confidence / raw_image_policy) }
这是"守门员+OCR 免费提供"的部分；tags 不在 payload 里——tags 由云端 enricher 生成后
才补上，构成完整 memory card。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import config


@dataclass
class Payload:
    """待 enrich 的打包数据（尚无 tags）。"""

    ocr_text: str
    timestamp: str
    trigger_confidence: float
    raw_image_policy: str = config.POLICY_DELETE

    def metadata(self) -> dict[str, Any]:
        """交给 enricher 的元数据（不含 ocr_text 本身）。"""
        return {
            "timestamp": self.timestamp,
            "trigger_confidence": self.trigger_confidence,
            "raw_image_policy": self.raw_image_policy,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"ocr_text": self.ocr_text, **self.metadata()}


def build_payload(
    ocr_text: str,
    timestamp: str,
    trigger_confidence: float,
    raw_image_policy: str = config.POLICY_DELETE,
) -> Payload:
    """组装 payload。校验 raw_image_policy 合法。"""
    if raw_image_policy not in config.VALID_RAW_IMAGE_POLICIES:
        raise ValueError(f"raw_image_policy 非法: {raw_image_policy!r}")
    return Payload(
        ocr_text=ocr_text,
        timestamp=timestamp,
        trigger_confidence=trigger_confidence,
        raw_image_policy=raw_image_policy,
    )
