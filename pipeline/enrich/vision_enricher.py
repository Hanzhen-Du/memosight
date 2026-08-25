"""VisionEnricher —— 路径Y：**直接传图**给云端多模态大模型，一步产出完整 memory card。

与现有 enricher 的区别（这是本模块存在的全部意义）：

    路径X（现有）：图 → 本地 Tesseract OCR → 传**文本** → DeepSeek/Claude 只产 tags
    路径Y（本模块）：图 → resize → base64 → 传**图** → Claude 多模态一步产出
                     {description, tags, extracted_text}

即：路径X 里"这张图是什么"这件事完全依赖 OCR 能不能读出字；OCR 读成乱码就彻底失明。
路径Y 直接让模型看图，OCR 失败的图它仍可能看懂（这正是本轮要验证的核心卖点）。

**本模块只是验证材料，不改管线默认行为**（默认 enricher 仍是 DeepSeekEnricher）。
架构方向由用户与导师决定。

接口约定：
- 主方法 `enrich_image(image, metadata) -> VisionCard`（本模块的真实接口，返回完整 card 内容）。
- 同时实现 `EnricherInterface.enrich(ocr_text, metadata) -> list[str]` 作为**适配器**：
  从 `metadata["image_path"]` 取图 → 调 enrich_image → 只返回 tags，
  这样本类也能直接插进现有 IngestService 管线（路径Y 的 tags 与路径X 可比）。

错误语义与 CloudEnricher/DeepSeekEnricher 完全一致：
- 瞬时（网络 / 限流 429 / 5xx / 过载）→ EnricherError → 上层入 pending 重试。
- 配置（密钥无效 401 / 无权限 403 / 模型 ID 错误 404 / 请求非法 400）→ EnricherConfigError → 不入队。
- 调用成功但输出无法解析 → 返回空 VisionCard（`parse_ok=False`，记 warning，**不崩**）。

成本：图片先 resize 到长边 ≤1024 再 base64（控 token 与内存）。每次调用的真实 token 用量
由 API 回包的 `usage` 带回，记在 `VisionCard.usage` 里，供上层统计费用（导师会问成本）。
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

from ..env import get_anthropic_api_key
from .base import EnricherConfigError, EnricherError, EnricherInterface

logger = logging.getLogger(__name__)

# 多模态模型。claude-haiku-4-5 也支持视觉但本轮要的是"理解得对不对"，用 Sonnet 更能代表
# "多模态大模型能力上限"这个论点；成本仍逐次实测记录，不靠估计。
DEFAULT_MODEL = "claude-sonnet-4-6"

# 定价（USD / 1M tokens），仅用于把实测 token 数换算成美元估算。
# 来源：Anthropic 官方定价表（claude-sonnet-4-6 = $3 输入 / $15 输出 每百万 token）。
# 若定价变动，此处需同步更新——报告里出现的美元数都由这两个常数算出。
PRICE_IN_PER_MTOK = 3.00
PRICE_OUT_PER_MTOK = 15.00

# 图片长边上限：超过则等比缩小后再 base64。1024 是"看得清字 vs 省 token"的折中。
DEFAULT_MAX_SIDE = 1024
JPEG_QUALITY = 85

SYSTEM_PROMPT = (
    "You are the perception module of a wearable visual memory assistant. "
    "The user gives you ONE photo captured by a head-mounted camera. "
    "Produce a searchable memory card for it.\n\n"
    "Reply with ONLY a JSON object, no prose and no markdown fences:\n"
    '{"description": "...", "tags": ["...", "..."], "extracted_text": "..."}\n\n'
    "Rules:\n"
    "- description: ONE sentence saying what this image is (English), concrete and specific.\n"
    "- tags: 3-6 concise lowercase search tags (nouns/topics/content-type, e.g. "
    "'slides','whiteboard','dashboard','textbook','code-screen','chemistry').\n"
    "  Always include one tag naming the content type.\n"
    "- extracted_text: the text visible in the image, verbatim, newline-separated. "
    "Use an empty string \"\" if there is no legible text. Do NOT invent text you cannot read.\n"
    "- If the image is unreadable or contains nothing useful, still return the object with "
    "your best-effort description and an empty tags array."
)

USER_INSTRUCTION = "Produce the memory card JSON for this image."

ImageInput = Union[str, Path, "np.ndarray"]


@dataclass
class VisionCard:
    """路径Y 的一次产出：完整 memory card 内容 + 本次调用的真实用量。"""

    description: str = ""
    tags: list[str] = field(default_factory=list)
    extracted_text: str = ""
    parse_ok: bool = False              # 模型输出是否成功解析成 JSON 对象
    refusal: bool = False               # 是否被安全拒答
    raw_output: str = ""                # 原始文本输出（解析失败时用于诊断）
    usage: dict[str, int] = field(default_factory=dict)  # input/output tokens（API 实测）
    image_bytes_sent: int = 0           # resize+JPEG 后实际发出的字节数

    @property
    def has_content(self) -> bool:
        """是否产出了可用内容（有描述或有标签）。"""
        return bool(self.description.strip() or self.tags)

    def cost_usd(self) -> float:
        """本次调用的美元估算（由实测 token 数 × 上面的定价常数算出）。"""
        return (
            self.usage.get("input_tokens", 0) / 1e6 * PRICE_IN_PER_MTOK
            + self.usage.get("output_tokens", 0) / 1e6 * PRICE_OUT_PER_MTOK
        )


# ---------- 图片预处理 ----------

def resize_and_encode(image: ImageInput, max_side: int = DEFAULT_MAX_SIDE) -> tuple[str, int]:
    """读图 → 长边缩到 ≤max_side → JPEG 编码 → base64。

    返回 (base64_str, 编码后字节数)。图**只读不写**，绝不改动源文件。
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"无法读取图片: {image}")
    else:
        img = image
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError(f"JPEG 编码失败: {image}")
    raw = buf.tobytes()
    return base64.b64encode(raw).decode("ascii"), len(raw)


# ---------- 输出解析 ----------

def _strip_code_fence(text: str) -> str:
    """去掉可能包裹的 ```json ... ``` 或 ``` ... ``` 代码块。"""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        s = s.rstrip()
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _coerce_text(value: Any) -> str:
    """把模型可能给的 str / list / None 统一成字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def parse_vision_output(text: str, max_tags: int = 6) -> tuple[bool, str, list[str], str]:
    """解析模型输出 → (parse_ok, description, tags, extracted_text)。

    容错策略（**任何情况都不抛异常**）：
    - strip ```` ``` ```` 围栏后 json.loads；失败 → parse_ok=False，全空。
    - 顶层若是数组，取第一个 dict 元素（模型偶尔多包一层）。
    - 字段缺失/类型不对 → 该字段退化为空，其余照常返回。
    """
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("VisionEnricher: JSON 解析失败，返回空 card。原始输出=%r", text[:200])
        return False, "", [], ""
    if isinstance(data, list):
        data = next((d for d in data if isinstance(d, dict)), None)
    if not isinstance(data, dict):
        logger.warning("VisionEnricher: 模型未返回 JSON 对象，返回空 card。")
        return False, "", [], ""

    description = _coerce_text(data.get("description"))
    extracted_text = _coerce_text(data.get("extracted_text"))

    tags: list[str] = []
    raw_tags = data.get("tags")
    if isinstance(raw_tags, str):          # 模型偶尔给逗号分隔字符串
        raw_tags = [t for t in raw_tags.split(",")]
    if isinstance(raw_tags, list):
        for item in raw_tags:
            if isinstance(item, str):
                t = item.strip().lower()
                if t and t not in tags:
                    tags.append(t)
    return True, description, tags[:max_tags], extracted_text


class VisionEnricher(EnricherInterface):
    """路径Y：图 → Claude 多模态 → 完整 memory card。"""

    name = "vision"  # 真实现，标签不带 mock: 前缀

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 500,
        max_tags: int = 6,
        max_side: int = DEFAULT_MAX_SIDE,
        client: Optional[Any] = None,   # 可注入（测试用）；否则惰性构造
        api_key: Optional[str] = None,  # 可显式传；否则从 env 读
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_tags = max_tags
        self.max_side = max_side
        self._client = client
        self._api_key = api_key

    @property
    def client(self):
        """惰性构造 Anthropic 客户端。缺密钥时抛 EnricherConfigError。"""
        if self._client is None:
            import anthropic  # 局部 import：没装 SDK 也能 import 本模块

            key = self._api_key or get_anthropic_api_key()
            if not key:
                raise EnricherConfigError(
                    "未找到 ANTHROPIC_API_KEY。请在项目根 .env 写入 ANTHROPIC_API_KEY=...，"
                    "或在环境里 export（.env 已 gitignore）。"
                )
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def _build_user_text(self, metadata: dict[str, Any]) -> str:
        ts = metadata.get("timestamp", "unknown")
        conf = metadata.get("trigger_confidence", "unknown")
        return (
            f"Scene metadata: captured_at={ts}, gatekeeper_confidence={conf}, "
            f"first-launch scenario = useful-text screen (whiteboard/document/slides/"
            f"code screen/projector).\n\n{USER_INSTRUCTION}"
        )

    # ---- 主方法：图 → 完整 card ----
    def enrich_image(self, image: ImageInput, metadata: Optional[dict[str, Any]] = None) -> VisionCard:
        """把一张图交给多模态模型，返回完整 memory card 内容 + 实测 token 用量。"""
        import anthropic

        metadata = metadata or {}
        client = self.client  # 缺密钥 → EnricherConfigError（不入队）
        b64, nbytes = resize_and_encode(image, max_side=self.max_side)

        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": self._build_user_text(metadata)},
                    ],
                }],
            )
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ) as e:
            raise EnricherConfigError(f"Claude 多模态 API 配置错误（不重试）: {e}") from e
        except anthropic.APIError as e:
            raise EnricherError(f"Claude 多模态 API 瞬时失败（可重试）: {e}") from e

        usage = {}
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = {
                "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
            }

        # 安全拒答：不是网络故障，也不该入队死循环 → 记 warning、返回空 card
        if getattr(resp, "stop_reason", None) == "refusal":
            logger.warning("VisionEnricher: 模型安全拒答，返回空 card。")
            return VisionCard(refusal=True, usage=usage, image_bytes_sent=nbytes)

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        ok, description, tags, extracted = parse_vision_output(text, max_tags=self.max_tags)
        return VisionCard(
            description=description,
            tags=tags,
            extracted_text=extracted,
            parse_ok=ok,
            raw_output=text,
            usage=usage,
            image_bytes_sent=nbytes,
        )

    # ---- 适配器：满足 EnricherInterface，便于直接插进现有管线 ----
    def enrich(self, ocr_text: str, metadata: dict[str, Any]) -> list[str]:
        """接口适配：从 metadata['image_path'] 取图跑 enrich_image，只返回 tags。

        注意 `ocr_text` 在路径Y **完全不使用**——这正是两条路径的分界点。
        缺 image_path 时抛 EnricherConfigError（是调用方用法错误，重试无用）。
        """
        image_path = (metadata or {}).get("image_path")
        if not image_path:
            raise EnricherConfigError(
                "VisionEnricher.enrich() 需要 metadata['image_path']（路径Y 传的是图不是文本）。"
                "直接产完整 card 请调用 enrich_image()。"
            )
        return self.enrich_image(image_path, metadata).tags
