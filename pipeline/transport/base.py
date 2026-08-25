"""The transport and upload abstract interface.

`upload(payload) -> MemoryCard` sends a packaged payload to the cloud and returns a complete
memory card with tags filled in and status=done.

Transport is deliberately separate from enrichment. Transport owns how a payload reaches the
cloud (direct or relayed via a phone); the enricher owns how tags are generated. The direct
implementation holds a reference to an enricher to complete the round trip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import MemoryCard
from ..packaging import Payload


class UploadInterface(ABC):
    name: str = "abstract"

    @abstractmethod
    def upload(self, payload: Payload) -> MemoryCard:
        raise NotImplementedError
