"""Pipeline configuration.

Central home for runtime parameters that are adjustable but have defaults. The privacy-related
raw_image_policy defaults to "delete"; this phase implements that field and the deletion logic,
with a configurable short-term cache as the alternative.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root (this file is at <root>/pipeline/config.py)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Runtime output goes to data/mvp_demo (data/ is gitignored and not committed).
_DEFAULT_DEMO_DIR = REPO_ROOT / "data" / "mvp_demo"

# Valid values for raw_image_policy
POLICY_DELETE = "delete"        # delete the raw frame as soon as it is processed (default, privacy first)
POLICY_CACHE = "cache"          # keep it in a short-term cache for debugging, governed by cache_ttl
VALID_RAW_IMAGE_POLICIES = (POLICY_DELETE, POLICY_CACHE)

# memory card states
STATUS_PENDING = "pending"
STATUS_DONE = "done"
VALID_STATUSES = (STATUS_PENDING, STATUS_DONE)


@dataclass
class Config:
    """Runtime configuration. Every field has a default, and tests can replace the whole
    object."""

    # storage
    db_path: Path = field(default_factory=lambda: _DEFAULT_DEMO_DIR / "memosight.db")
    # Where full-resolution frames land. On a mocked trigger, a test image is copied here to
    # stand in for a grab.
    frames_dir: Path = field(default_factory=lambda: _DEFAULT_DEMO_DIR / "frames")
    # Short-term cache directory, used when raw_image_policy is "cache"
    cache_dir: Path = field(default_factory=lambda: _DEFAULT_DEMO_DIR / "cache")

    # privacy
    raw_image_policy: str = POLICY_DELETE
    cache_ttl_seconds: int = 24 * 3600  # cache lifetime under the cache policy. Recorded only
                                        # in this phase; no background cleanup thread runs

    # OCR: Tesseract languages (simplified Chinese plus English). On the Pi the chi_sim and eng
    # data packages have to be installed with apt.
    ocr_lang: str = "chi_sim+eng"
    # Longest side to resize to before processing, which saves memory and steadies OCR
    ocr_max_side: int = 1600

    def ensure_dirs(self) -> None:
        """Make sure the runtime directories exist."""
        for d in (self.db_path.parent, self.frames_dir, self.cache_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


def default_config() -> Config:
    """Default configuration. The database path can be overridden with the MEMOSIGHT_DB
    environment variable, which is convenient for the CLI and for tests."""
    cfg = Config()
    env_db = os.environ.get("MEMOSIGHT_DB")
    if env_db:
        cfg.db_path = Path(env_db)
    return cfg
