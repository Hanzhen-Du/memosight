"""Privacy: implementing raw_image_policy.

The default is "delete": once OCR is done and the payload is packaged, the original
full-resolution frame is deleted immediately. Privacy comes first.
"cache" moves it into a short-term cache directory instead. cache_ttl_seconds is recorded but
no background cleanup thread runs in this phase.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from . import config


def apply_raw_image_policy(
    frame_path: Path, policy: str, cache_dir: Path
) -> Optional[Path]:
    """Apply the policy to the raw frame and return its final path, or None when deleted."""
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
    raise ValueError(f"invalid raw_image_policy: {policy!r}")
