"""环境变量 / 密钥加载。

用 python-dotenv 从**项目根 `.env`** 自动加载环境变量（如 ANTHROPIC_API_KEY），
这样不依赖终端是否 export 过——任何进程 import 本模块并调 `load_env()` 都能读到。

**绝不硬编码密钥。** 密钥只从 os.environ 读取，`.env` 已 gitignore（不进 git）。
本阶段 enricher 是 mock、用不到真 key；此处只把加载基础设施铺好，供阶段二真 Claude API 用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from . import config

# 项目根 .env（<root>/.env）
DOTENV_PATH = config.REPO_ROOT / ".env"

_loaded = False


def load_env(override: bool = False) -> None:
    """幂等地从项目根 .env 加载环境变量。

    override=False（默认）：不覆盖进程里已 export 的同名变量（export 优先）。
    找不到 .env 也不报错（CI/无 .env 环境照常跑，靠真实 env 或 mock）。
    """
    global _loaded
    if _loaded and not override:
        return
    if DOTENV_PATH.exists():
        load_dotenv(dotenv_path=DOTENV_PATH, override=override)
    _loaded = True


def get_anthropic_api_key() -> Optional[str]:
    """读取 ANTHROPIC_API_KEY（先确保 .env 已加载）。缺失返回 None。"""
    load_env()
    return os.environ.get("ANTHROPIC_API_KEY")


def require_anthropic_api_key() -> str:
    """取 ANTHROPIC_API_KEY；缺失则抛清晰错误（供阶段二真 API 调用用）。"""
    key = get_anthropic_api_key()
    if not key:
        raise RuntimeError(
            "未找到 ANTHROPIC_API_KEY。请在项目根 .env 写入 "
            "`ANTHROPIC_API_KEY=...`（.env 已 gitignore），或在环境里 export。"
        )
    return key


def get_deepseek_api_key() -> Optional[str]:
    """读取 DEEPSEEK_API_KEY（先确保 .env 已加载）。缺失返回 None。"""
    load_env()
    return os.environ.get("DEEPSEEK_API_KEY")


def require_deepseek_api_key() -> str:
    """取 DEEPSEEK_API_KEY；缺失则抛清晰错误。"""
    key = get_deepseek_api_key()
    if not key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。请在项目根 .env 写入 "
            "`DEEPSEEK_API_KEY=...`（.env 已 gitignore），或在环境里 export。"
        )
    return key
