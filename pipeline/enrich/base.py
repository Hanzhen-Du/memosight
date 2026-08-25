"""Enricher 抽象接口。

约定：`enrich(ocr_text, metadata) -> list[str]`（tags 数组）。
唤起云端大模型来生成标签是本接口的唯一职责——本阶段用 mock 实现占位。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EnricherError(RuntimeError):
    """enrich **瞬时**失败（网络 / 限流 / 5xx / 模拟错误）。管线据此入 pending 队列，联网后重试。"""


class EnricherConfigError(RuntimeError):
    """enrich **配置**错误（密钥无效 / 无权限 / 模型 ID 错误 / 请求非法）。

    不是瞬时故障，重试无用——不入队，直接向上抛出让用户修。故意**不**继承 EnricherError，
    这样 IngestService 的 `except EnricherError` 不会把它误当成"待重试"而入队。
    """


class EnricherInterface(ABC):
    """所有 enricher 的抽象基类。"""

    name: str = "abstract"

    @abstractmethod
    def enrich(self, ocr_text: str, metadata: dict[str, Any]) -> list[str]:
        """调用（未来真、现在 mock）云端大模型，返回 tags 数组。"""
        raise NotImplementedError
