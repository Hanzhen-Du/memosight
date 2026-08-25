"""MemoSight MVP pipeline, phase one.

Gatekeeper trigger (mocked) -> full-resolution frame (a test image) -> local OCR -> packaging
-> cloud enricher (mocked tags) -> SQLite storage -> command-line recall.

Three swappable interfaces, each its own module with an abstract base class: OCR, Enricher and
Transport.
"""

__all__ = ["models", "db", "config"]
