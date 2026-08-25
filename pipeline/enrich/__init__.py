"""Enricher 接口层。

抽象基类 `EnricherInterface`。四个实现（可配置切换，都不删）：
- `DeepSeekEnricher`：**默认**真实现，DeepSeek（OpenAI 兼容 API），标签为真实语义。
- `CloudEnricher`：真实现，Anthropic Claude（Haiku），标签为真实语义。
- `MockCloudEnricher`：**mock** 版返回 `mock:` 假标签（离线测试 / CI 用）。
- `VisionEnricher`：**路径Y 验证用**，直接传图给 Claude 多模态，一步产出完整 card
  （description + tags + extracted_text）。**不是默认**——架构方向待与导师确认。

约定：上面前三个实现里，tags 是唯一由"云端大模型"生成的字段——本层是它的唯一来源，
不写规则实现。VisionEnricher 是这条约定的**候选替代方案**（连 ocr_text 也由云端产），
本轮只作对比验证，不改管线默认。
"""

from .base import EnricherConfigError, EnricherError, EnricherInterface
from .cloud_enricher import CloudEnricher
from .deepseek_enricher import DeepSeekEnricher
from .mock_enricher import MockCloudEnricher
from .vision_enricher import VisionCard, VisionEnricher

__all__ = [
    "EnricherInterface",
    "EnricherError",
    "EnricherConfigError",
    "DeepSeekEnricher",
    "CloudEnricher",
    "MockCloudEnricher",
    "VisionEnricher",
    "VisionCard",
]
