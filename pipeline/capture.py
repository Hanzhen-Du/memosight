"""Gatekeeper trigger and full-resolution frame grab. Fully mocked in this phase.

In the real system a lightweight always-on gatekeeper decides whether something is worth
recording, and only a trigger wakes the full-resolution grab.
In this phase, `MockGatekeeper` supplies a mocked trigger signal and confidence, and a test
image stands in for the full-resolution frame.
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
    """Mocked gatekeeper: returns a trigger signal for a given input.

    Triggers unconditionally by default, which suits the demo. Setting a threshold plus a
    per-call confidence makes the trigger conditional.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def trigger(self, confidence: float = 0.9) -> TriggerSignal:
        return TriggerSignal(should_record=confidence >= self.threshold, confidence=confidence)


def grab_frame(source_image: Path, frames_dir: Path, ts: Optional[str] = None) -> Path:
    """Simulate a full-resolution grab by copying a test image into frames_dir with a
    timestamped name. Returns the frame path."""
    source_image = Path(source_image)
    if not source_image.exists():
        raise FileNotFoundError(f"test image does not exist: {source_image}")
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    stamp = (ts or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")).replace(":", "").replace("+", "")
    dest = frames_dir / f"frame_{stamp}{source_image.suffix or '.png'}"
    shutil.copy(str(source_image), str(dest))
    return dest
