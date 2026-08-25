"""The enricher abstract interface.

Contract: `enrich(ocr_text, metadata) -> list[str]`, an array of tags.
Calling a cloud model to generate tags is this interface's only responsibility. In this phase a
mock implementation stands in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EnricherError(RuntimeError):
    """A transient enrichment failure: network, rate limiting, 5xx, or a simulated error. The
    pipeline queues the card as pending and retries once connectivity returns."""


class EnricherConfigError(RuntimeError):
    """A configuration error in enrichment: invalid key, no permission, wrong model id, or a
    malformed request.

    This is not transient and retrying will not help, so it is not queued and is raised to the
    caller to fix. It deliberately does not inherit from EnricherError, so IngestService's
    `except EnricherError` cannot mistake it for something worth retrying and queue it.
    """


class EnricherInterface(ABC):
    """Abstract base class for every enricher."""

    name: str = "abstract"

    @abstractmethod
    def enrich(self, ocr_text: str, metadata: dict[str, Any]) -> list[str]:
        """Call the cloud model, mocked for now and real later, and return an array of tags."""
        raise NotImplementedError
