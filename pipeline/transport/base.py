"""传输/上传抽象接口。

`upload(payload) -> MemoryCard`：把打包好的 payload 送去"云端"，
拿回一张已补 tags 的完整 memory card（status=done）。

传输层与 enricher 分离：传输层管"payload 怎么到达云端"（直连 vs 手机中转），
enricher 管"tags 怎么生成"。直连实现持有一个 enricher 引用来完成这趟往返。
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
