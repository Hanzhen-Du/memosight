"""传输/上传接口层。

抽象基类 `UploadInterface`；本阶段实现 `DirectUploadMock`（树莓派直连模式的 mock）。
未来可换成"经手机中转"的实现，管线其余部分不变。

约定：`upload(payload) -> MemoryCard`。直连实现内部调用 enricher 拿 tags，
组装成一张 status=done 的完整 memory card 返回（尚未入库，由上层持久化）。
"""

from .base import UploadInterface
from .direct_mock import DirectUploadMock

__all__ = ["UploadInterface", "DirectUploadMock"]
