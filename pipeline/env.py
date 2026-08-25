"""Environment variable and API key loading.

Uses python-dotenv to load environment variables such as ANTHROPIC_API_KEY from a `.env` at the
project root, so nothing depends on whether a shell exported them. Any process that imports this
module and calls `load_env()` can read them.

Keys are never hardcoded. They are read only from os.environ, and `.env` is gitignored so it is
never committed. In this phase the enricher is mocked and no real key is needed; this is the
infrastructure for the real API call in phase two.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from . import config

# The project-root .env, at <root>/.env
DOTENV_PATH = config.REPO_ROOT / ".env"

_loaded = False


def load_env(override: bool = False) -> None:
    """Idempotently load environment variables from the project-root .env.

    With override=False, the default, variables already exported in the process are not
    overwritten, so an export wins.
    A missing .env is not an error, so CI and environments without one still run, relying on
    the real environment or on mocks.
    """
    global _loaded
    if _loaded and not override:
        return
    if DOTENV_PATH.exists():
        load_dotenv(dotenv_path=DOTENV_PATH, override=override)
    _loaded = True


def get_anthropic_api_key() -> Optional[str]:
    """Read ANTHROPIC_API_KEY, loading .env first. Returns None if it is not set."""
    load_env()
    return os.environ.get("ANTHROPIC_API_KEY")


def require_anthropic_api_key() -> str:
    """Get ANTHROPIC_API_KEY, raising a clear error if it is missing. Used by the real API
    call in phase two."""
    key = get_anthropic_api_key()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found. Put `ANTHROPIC_API_KEY=...` in a .env at the "
            "project root (it is gitignored), or export it in the environment."
        )
    return key


def get_deepseek_api_key() -> Optional[str]:
    """Read DEEPSEEK_API_KEY, loading .env first. Returns None if it is not set."""
    load_env()
    return os.environ.get("DEEPSEEK_API_KEY")


def require_deepseek_api_key() -> str:
    """Get DEEPSEEK_API_KEY, raising a clear error if it is missing."""
    key = get_deepseek_api_key()
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not found. Put `DEEPSEEK_API_KEY=...` in a .env at the "
            "project root (it is gitignored), or export it in the environment."
        )
    return key
