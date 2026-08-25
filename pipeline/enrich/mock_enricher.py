"""MockCloudEnricher: the mocked version of cloud-model tagging, for offline testing.

It calls no real API. It is a placeholder tag generator for unit tests, offline development and
CI:
- it returns a structurally valid but fake tags array, not semantically real tags derived from
  rules;
- every tag carries a `mock:` prefix so its origin is obvious and nothing pretends to be real;
- to keep tests reproducible, tags are derived from a stable hash of the input text, so the same
  input always yields the same fake tags;
- `simulate_latency_s` and `fail` simulate cloud latency and errors, which exercises the queue
  and retry paths.

The production path uses the real `CloudEnricher` (see cloud_enricher.py).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

from .base import EnricherError, EnricherInterface

# A small pool of tag-like fake words. Placeholder only, with no semantic guarantee.
_MOCK_VOCAB = [
    "meeting", "whiteboard", "document", "slides", "code-screen",
    "diagram", "notes", "roadmap", "lecture", "todo",
]


class MockCloudEnricher(EnricherInterface):
    """The mocked cloud enricher. Returns three `mock:`-prefixed fake tags by default."""

    name = "cloud-mock"

    def __init__(
        self,
        canned_tags: Optional[list[str]] = None,
        num_tags: int = 3,
        simulate_latency_s: float = 0.0,
        fail: bool = False,
    ):
        # canned_tags, when given, is returned unconditionally. For tests.
        self.canned_tags = canned_tags
        self.num_tags = num_tags
        self.simulate_latency_s = simulate_latency_s
        self.fail = fail

    def enrich(self, ocr_text: str, metadata: dict[str, Any]) -> list[str]:
        if self.simulate_latency_s > 0:
            time.sleep(self.simulate_latency_s)
        if self.fail:
            raise EnricherError("simulated cloud call failure")

        if self.canned_tags is not None:
            return list(self.canned_tags)

        # Derive fake tags from a stable hash of the input: reproducible, but not real semantics.
        digest = hashlib.sha256(ocr_text.encode("utf-8")).digest()
        picks: list[str] = []
        for i in range(self.num_tags):
            word = _MOCK_VOCAB[digest[i] % len(_MOCK_VOCAB)]
            tag = f"mock:{word}"
            if tag not in picks:
                picks.append(tag)
        # Append a fake tag reflecting the confidence band, to show metadata can be used
        conf = float(metadata.get("trigger_confidence", 0.0))
        picks.append("mock:high-conf" if conf >= 0.8 else "mock:low-conf")
        return picks
