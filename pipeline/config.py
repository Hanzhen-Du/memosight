"""Pipeline configuration.

集中放"可调但有默认"的运行参数。隐私相关的 raw_image_policy 默认 "delete"
（本阶段落地该字段 + 删除逻辑，可配置短期缓存）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 仓库根目录（本文件在 <root>/pipeline/config.py）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 运行期产物统一放到 data/mvp_demo（data/ 已 gitignore，不进 git）。
_DEFAULT_DEMO_DIR = REPO_ROOT / "data" / "mvp_demo"

# raw_image_policy 合法取值
POLICY_DELETE = "delete"        # 处理完立即删除原始帧（默认，隐私优先）
POLICY_CACHE = "cache"          # 保留在短期缓存目录，供后续调试；由 cache_ttl 控制
VALID_RAW_IMAGE_POLICIES = (POLICY_DELETE, POLICY_CACHE)

# memory card 状态
STATUS_PENDING = "pending"
STATUS_DONE = "done"
VALID_STATUSES = (STATUS_PENDING, STATUS_DONE)


@dataclass
class Config:
    """运行时配置。字段全部有默认值，测试里可整体替换。"""

    # 存储
    db_path: Path = field(default_factory=lambda: _DEFAULT_DEMO_DIR / "memosight.db")
    # 高清帧落地目录（mock 触发时把测试图拷进来当"抓帧"）
    frames_dir: Path = field(default_factory=lambda: _DEFAULT_DEMO_DIR / "frames")
    # 短期缓存目录（raw_image_policy == "cache" 时用）
    cache_dir: Path = field(default_factory=lambda: _DEFAULT_DEMO_DIR / "cache")

    # 隐私
    raw_image_policy: str = POLICY_DELETE
    cache_ttl_seconds: int = 24 * 3600  # cache 策略下的短期缓存时长（本阶段仅登记，不起后台清理线程）

    # OCR：Tesseract 语言（中文简体 + 英文）。Pi 上需 apt 装 chi_sim/eng 数据包。
    ocr_lang: str = "chi_sim+eng"
    # 处理前统一 resize 的最长边（省内存、稳 OCR）
    ocr_max_side: int = 1600

    def ensure_dirs(self) -> None:
        """确保运行期目录存在。"""
        for d in (self.db_path.parent, self.frames_dir, self.cache_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


def default_config() -> Config:
    """默认配置；支持用环境变量 MEMOSIGHT_DB 覆盖 db 路径（方便 CLI/测试）。"""
    cfg = Config()
    env_db = os.environ.get("MEMOSIGHT_DB")
    if env_db:
        cfg.db_path = Path(env_db)
    return cfg
