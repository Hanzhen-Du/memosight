"""The transport and upload interface layer.

`UploadInterface` is the abstract base class. This phase implements `DirectUploadMock`, a mock
of the Pi uploading directly. A phone-relay implementation can replace it later without
changing the rest of the pipeline.

Contract: `upload(payload) -> MemoryCard`. The direct implementation calls the enricher to get
tags and returns a complete memory card with status=done. It does not persist the card; the
layer above does that.
"""

from .base import UploadInterface
from .direct_mock import DirectUploadMock

__all__ = ["UploadInterface", "DirectUploadMock"]
