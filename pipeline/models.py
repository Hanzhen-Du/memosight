"""Memory card 数据模型。

字段定死（与 SQLite 表一一对应）：
id / timestamp / trigger_confidence / ocr_text / tags(JSON数组) /
raw_image_policy(默认"delete") / status(pending|done) / created_at / enriched_at

除 tags 外全部由系统/OCR/守门员免费提供；tags 是唯一由云端大模型生成的字段。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from . import config


def utc_now_iso() -> str:
    """统一的时间戳格式：UTC ISO-8601（秒精度，带 Z）。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class MemoryCard:
    """一张记忆卡片。

    - `tags` 为 None 表示尚未由云端 enricher 生成（pending 态）。
      生成后是一个字符串数组（JSON 存库）。
    - `enriched_at` 在 tags 回填时写入。
    """

    timestamp: str                          # 抓帧/触发时刻（ISO）
    trigger_confidence: float               # 守门员置信度
    ocr_text: str                           # 本地 OCR 文本
    tags: Optional[list[str]] = None        # 云端生成的标签数组；None=未生成
    raw_image_policy: str = config.POLICY_DELETE
    status: str = config.STATUS_PENDING
    created_at: str = field(default_factory=utc_now_iso)
    enriched_at: Optional[str] = None
    id: Optional[int] = None                # 入库后由 SQLite 分配

    def __post_init__(self) -> None:
        if self.raw_image_policy not in config.VALID_RAW_IMAGE_POLICIES:
            raise ValueError(
                f"raw_image_policy 非法: {self.raw_image_policy!r}，"
                f"合法值 {config.VALID_RAW_IMAGE_POLICIES}"
            )
        if self.status not in config.VALID_STATUSES:
            raise ValueError(
                f"status 非法: {self.status!r}，合法值 {config.VALID_STATUSES}"
            )

    # ---- 序列化 helpers（tags 在 DB 里是 JSON 文本）----
    def tags_json(self) -> Optional[str]:
        return None if self.tags is None else json.dumps(self.tags, ensure_ascii=False)

    @staticmethod
    def tags_from_json(raw: Optional[str]) -> Optional[list[str]]:
        if raw is None:
            return None
        return json.loads(raw)

    def to_row(self) -> dict[str, Any]:
        """转成可直接写入 SQLite 的 dict（tags 已 JSON 化）。"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "trigger_confidence": self.trigger_confidence,
            "ocr_text": self.ocr_text,
            "tags": self.tags_json(),
            "raw_image_policy": self.raw_image_policy,
            "status": self.status,
            "created_at": self.created_at,
            "enriched_at": self.enriched_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "MemoryCard":
        """从 sqlite3.Row / dict 还原。"""
        d = dict(row)
        return cls(
            id=d["id"],
            timestamp=d["timestamp"],
            trigger_confidence=d["trigger_confidence"],
            ocr_text=d["ocr_text"],
            tags=cls.tags_from_json(d["tags"]),
            raw_image_policy=d["raw_image_policy"],
            status=d["status"],
            created_at=d["created_at"],
            enriched_at=d["enriched_at"],
        )
