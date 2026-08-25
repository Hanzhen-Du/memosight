"""VisionEnricher 测试。

- 解析 + 错误映射 + 图片缩放：fake Anthropic client，**不真调 API**（省钱、可离线、可 CI）。
- 集成测试：一条真多模态调用，需 ANTHROPIC_API_KEY + MEMOSIGHT_LIVE_VISION=1，否则 skip。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import anthropic
import httpx

from pipeline.enrich import EnricherConfigError, EnricherError, VisionEnricher
from pipeline.enrich.vision_enricher import (
    DEFAULT_MAX_SIDE, parse_vision_output, resize_and_encode,
)


# ---- fake Anthropic client / response ----
class _TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Usage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _Resp:
    def __init__(self, text, stop_reason="end_turn", usage=(100, 50)):
        self.content = [_TextBlock(text)]
        self.stop_reason = stop_reason
        self.usage = _Usage(*usage)


class _FakeMessages:
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._resp


class _FakeClient:
    """伪装 anthropic.Anthropic：只暴露 client.messages.create。"""

    def __init__(self, resp=None, exc=None):
        self.messages = _FakeMessages(resp=resp, exc=exc)


def _req():
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _resp_obj(code):
    return httpx.Response(code, request=_req())


def _write_img(path: Path, w: int, h: int) -> Path:
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


GOOD_JSON = json.dumps({
    "description": "A lecture slide about neural networks.",
    "tags": ["Slides", "neural-networks", "lecture", "slides"],  # 含大写 + 重复
    "extracted_text": "Neural Networks\nBackpropagation",
})


class TestResize(unittest.TestCase):
    def test_downscales_long_edge(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write_img(Path(d) / "big.jpg", 3000, 1500)
            b64, nbytes = resize_and_encode(p)
            self.assertGreater(len(b64), 0)
            self.assertGreater(nbytes, 0)
            # 解回来确认长边确实 ≤ 1024
            import base64 as b64mod
            arr = np.frombuffer(b64mod.b64decode(b64), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            self.assertLessEqual(max(img.shape[:2]), DEFAULT_MAX_SIDE)

    def test_small_image_not_upscaled(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write_img(Path(d) / "small.jpg", 320, 240)
            b64, _ = resize_and_encode(p)
            import base64 as b64mod
            arr = np.frombuffer(b64mod.b64decode(b64), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            self.assertEqual(img.shape[:2], (240, 320))

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            resize_and_encode(Path("/nonexistent/nope.jpg"))


class TestParse(unittest.TestCase):
    def test_good_json(self):
        ok, desc, tags, text = parse_vision_output(GOOD_JSON)
        self.assertTrue(ok)
        self.assertIn("neural networks", desc.lower())
        self.assertEqual(tags, ["slides", "neural-networks", "lecture"])  # 小写 + 去重
        self.assertIn("Backpropagation", text)

    def test_code_fence_stripped(self):
        ok, desc, tags, _ = parse_vision_output(f"```json\n{GOOD_JSON}\n```")
        self.assertTrue(ok)
        self.assertEqual(len(tags), 3)

    def test_garbage_does_not_raise(self):
        ok, desc, tags, text = parse_vision_output("sorry, I can't help with that")
        self.assertFalse(ok)
        self.assertEqual((desc, tags, text), ("", [], ""))

    def test_missing_fields_degrade(self):
        ok, desc, tags, text = parse_vision_output('{"description": "a photo"}')
        self.assertTrue(ok)
        self.assertEqual(desc, "a photo")
        self.assertEqual(tags, [])
        self.assertEqual(text, "")

    def test_wrapped_in_array(self):
        ok, desc, _, _ = parse_vision_output(f"[{GOOD_JSON}]")
        self.assertTrue(ok)
        self.assertTrue(desc)

    def test_tags_as_comma_string(self):
        ok, _, tags, _ = parse_vision_output('{"tags": "a, b, c"}')
        self.assertTrue(ok)
        self.assertEqual(tags, ["a", "b", "c"])

    def test_extracted_text_as_list(self):
        ok, _, _, text = parse_vision_output('{"extracted_text": ["line1", "line2"]}')
        self.assertTrue(ok)
        self.assertEqual(text, "line1\nline2")

    def test_max_tags_capped(self):
        many = json.dumps({"tags": [f"t{i}" for i in range(20)]})
        ok, _, tags, _ = parse_vision_output(many, max_tags=6)
        self.assertEqual(len(tags), 6)


class TestEnrichImage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.img = _write_img(Path(self._tmp.name) / "x.jpg", 1600, 900)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, resp=None, exc=None):
        enr = VisionEnricher(client=_FakeClient(resp=resp, exc=exc))
        return enr, enr.enrich_image(self.img, {"timestamp": "t", "trigger_confidence": 0.9})

    def test_happy_path(self):
        enr, card = self._run(resp=_Resp(GOOD_JSON))
        self.assertTrue(card.parse_ok)
        self.assertTrue(card.has_content)
        self.assertEqual(card.tags, ["slides", "neural-networks", "lecture"])
        self.assertEqual(card.usage, {"input_tokens": 100, "output_tokens": 50})
        self.assertGreater(card.image_bytes_sent, 0)
        # 成本 = 100/1e6*3 + 50/1e6*15
        self.assertAlmostEqual(card.cost_usd(), 100 / 1e6 * 3 + 50 / 1e6 * 15)

    def test_request_shape(self):
        """确认发出去的确实是 image block + text block，且 model/max_tokens 正确。"""
        enr, _ = self._run(resp=_Resp(GOOD_JSON))
        kwargs = enr.client.messages.calls[0]
        self.assertEqual(kwargs["model"], "claude-sonnet-4-6")
        self.assertEqual(kwargs["max_tokens"], 500)
        blocks = kwargs["messages"][0]["content"]
        self.assertEqual(blocks[0]["type"], "image")
        self.assertEqual(blocks[0]["source"]["media_type"], "image/jpeg")
        self.assertEqual(blocks[0]["source"]["type"], "base64")
        self.assertEqual(blocks[1]["type"], "text")

    def test_unparseable_output_does_not_crash(self):
        _, card = self._run(resp=_Resp("I'm not going to answer in JSON"))
        self.assertFalse(card.parse_ok)
        self.assertFalse(card.has_content)
        self.assertTrue(card.raw_output)          # 原始输出保留供诊断
        self.assertEqual(card.usage["input_tokens"], 100)  # 用量仍记（钱已花）

    def test_refusal(self):
        _, card = self._run(resp=_Resp("", stop_reason="refusal"))
        self.assertTrue(card.refusal)
        self.assertFalse(card.has_content)

    def test_transient_error_maps_to_enricher_error(self):
        exc = anthropic.APIConnectionError(request=_req())
        with self.assertRaises(EnricherError):
            self._run(exc=exc)

    def test_rate_limit_is_transient(self):
        exc = anthropic.RateLimitError("429", response=_resp_obj(429), body=None)
        with self.assertRaises(EnricherError):
            self._run(exc=exc)

    def test_auth_error_maps_to_config_error(self):
        exc = anthropic.AuthenticationError("401", response=_resp_obj(401), body=None)
        with self.assertRaises(EnricherConfigError):
            self._run(exc=exc)
        # 配置错误**不是** EnricherError 子类 → 不会被误当成"待重试"入队
        self.assertFalse(issubclass(EnricherConfigError, EnricherError))

    def test_bad_model_id_maps_to_config_error(self):
        exc = anthropic.NotFoundError("404", response=_resp_obj(404), body=None)
        with self.assertRaises(EnricherConfigError):
            self._run(exc=exc)

    def test_missing_api_key_raises_config_error(self):
        enr = VisionEnricher(api_key=None)
        with mock.patch(
            "pipeline.enrich.vision_enricher.get_anthropic_api_key", return_value=None
        ):
            with self.assertRaises(EnricherConfigError):
                _ = enr.client


class TestInterfaceAdapter(unittest.TestCase):
    """enrich(ocr_text, metadata) 适配器：满足 EnricherInterface，可插进现有管线。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.img = _write_img(Path(self._tmp.name) / "x.jpg", 800, 600)

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_tags_from_image(self):
        enr = VisionEnricher(client=_FakeClient(resp=_Resp(GOOD_JSON)))
        tags = enr.enrich("这段 OCR 文本会被忽略", {"image_path": str(self.img)})
        self.assertEqual(tags, ["slides", "neural-networks", "lecture"])

    def test_without_image_path_raises_config_error(self):
        enr = VisionEnricher(client=_FakeClient(resp=_Resp(GOOD_JSON)))
        with self.assertRaises(EnricherConfigError):
            enr.enrich("some text", {})


@unittest.skipUnless(
    os.environ.get("MEMOSIGHT_LIVE_VISION") == "1" and os.environ.get("ANTHROPIC_API_KEY"),
    "需要 MEMOSIGHT_LIVE_VISION=1 + ANTHROPIC_API_KEY（真实调用，花钱）",
)
class TestLive(unittest.TestCase):
    def test_one_real_call(self):
        img = next(
            (Path(__file__).resolve().parent.parent / "demo" / "images").glob("*.jp*g")
        )
        card = VisionEnricher().enrich_image(img, {"timestamp": "live-test"})
        self.assertTrue(card.parse_ok)
        self.assertTrue(card.description)
        print(f"\nlive: desc={card.description!r} tags={card.tags} cost=${card.cost_usd():.5f}")


if __name__ == "__main__":
    unittest.main()
