"""MockCloudEnricher —— 云端大模型打标的 **mock 版**（离线测试用）。

不接真 API。占位假标签生成器，供单元测试 / 离线开发 / CI 用：
- 返回一个 **假的** 结构化 tags 数组（不是规则解析出的真语义标签）；
- 标签用 `mock:` 前缀标注来源，诚实表明"非真实语义、仅演示占位"；
- 为测试可复现，标签基于输入文本的稳定哈希派生（同输入→同假标签）；
- 支持 `simulate_latency_s` / `fail` 模拟云端时延与错误，供队列/重试路径用。

生产路径用真 `CloudEnricher`（见 cloud_enricher.py）。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

from .base import EnricherError, EnricherInterface

# 一小撮"看起来像标签"的假词池；仅用于演示占位，无语义保证。
_MOCK_VOCAB = [
    "meeting", "whiteboard", "document", "slides", "code-screen",
    "diagram", "notes", "roadmap", "lecture", "todo",
]


class MockCloudEnricher(EnricherInterface):
    """mock 云端 enricher。默认返回 3 个带 `mock:` 前缀的假标签。"""

    name = "cloud-mock"

    def __init__(
        self,
        canned_tags: Optional[list[str]] = None,
        num_tags: int = 3,
        simulate_latency_s: float = 0.0,
        fail: bool = False,
    ):
        # canned_tags: 若给定则恒定返回它（测试用）
        self.canned_tags = canned_tags
        self.num_tags = num_tags
        self.simulate_latency_s = simulate_latency_s
        self.fail = fail

    def enrich(self, ocr_text: str, metadata: dict[str, Any]) -> list[str]:
        if self.simulate_latency_s > 0:
            time.sleep(self.simulate_latency_s)
        if self.fail:
            raise EnricherError("mock 云端调用失败（模拟）")

        if self.canned_tags is not None:
            return list(self.canned_tags)

        # 从输入稳定哈希派生假标签（可复现，但非真语义）。
        digest = hashlib.sha256(ocr_text.encode("utf-8")).digest()
        picks: list[str] = []
        for i in range(self.num_tags):
            word = _MOCK_VOCAB[digest[i] % len(_MOCK_VOCAB)]
            tag = f"mock:{word}"
            if tag not in picks:
                picks.append(tag)
        # 附一个反映置信度档位的假标签，展示元数据可被利用
        conf = float(metadata.get("trigger_confidence", 0.0))
        picks.append("mock:high-conf" if conf >= 0.8 else "mock:low-conf")
        return picks
