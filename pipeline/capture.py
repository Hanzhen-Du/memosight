"""守门员触发 + 高清抓帧（本阶段全 mock）。

真系统里：常开的轻量守门员判断"是否值得记录"，触发才唤醒高清抓帧。
本阶段：`MockGatekeeper` 给一个 mock 触发信号 + 置信度；高清帧用测试图片替代。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class TriggerSignal:
    should_record: bool
    confidence: float


class MockGatekeeper:
    """mock 守门员：对给定输入返回触发信号。

    默认恒触发（用于演示）；可设 threshold + 每次 confidence 决定是否触发。
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def trigger(self, confidence: float = 0.9) -> TriggerSignal:
        return TriggerSignal(should_record=confidence >= self.threshold, confidence=confidence)


def grab_frame(source_image: Path, frames_dir: Path, ts: Optional[str] = None) -> Path:
    """模拟"高清抓帧"：把测试图片拷进 frames_dir，命名带时间戳。返回帧路径。"""
    source_image = Path(source_image)
    if not source_image.exists():
        raise FileNotFoundError(f"测试图片不存在: {source_image}")
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    stamp = (ts or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")).replace(":", "").replace("+", "")
    dest = frames_dir / f"frame_{stamp}{source_image.suffix or '.png'}"
    shutil.copy(str(source_image), str(dest))
    return dest
