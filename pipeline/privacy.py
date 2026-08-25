"""隐私：raw_image_policy 落地。

默认 "delete"：OCR 完成、payload 打包后立即删除原始高清帧（隐私优先）。
"cache"：移入短期缓存目录（cache_ttl_seconds 仅登记，本阶段不起后台清理线程）。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from . import config


def apply_raw_image_policy(
    frame_path: Path, policy: str, cache_dir: Path
) -> Optional[Path]:
    """按策略处理原始帧。返回帧的最终路径（delete → None）。"""
    frame_path = Path(frame_path)
    if policy == config.POLICY_DELETE:
        if frame_path.exists():
            frame_path.unlink()
        return None
    if policy == config.POLICY_CACHE:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / frame_path.name
        if frame_path.exists():
            shutil.move(str(frame_path), str(dest))
        return dest
    raise ValueError(f"raw_image_policy 非法: {policy!r}")
