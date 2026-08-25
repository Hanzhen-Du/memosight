"""阶段二-A: 真 CloudEnricher 测试。

- 解析逻辑 + 错误映射：用 fake client，**不真调 API**（省 token、可离线、可 CI）。
- 集成测试：一条真调用，需 ANTHROPIC_API_KEY + 显式开关 MEMOSIGHT_LIVE_ENRICH=1，否则 skip。
"""

import os
import unittest
from unittest import mock

import httpx

import anthropic

from pipeline.enrich import CloudEnricher, EnricherConfigError, EnricherError
from pipeline.enrich.cloud_enricher import _parse_tags, _strip_code_fence


# ---- fake Anthropic client / response ----
class _Block:
    def __init__(self, text, type="text"):
        self.text = text
        self.type = type


class _Resp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason


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
    def __init__(self, resp=None, exc=None):
        self.messages = _FakeMessages(resp=resp, exc=exc)


def _req():
    return httpx.Request("POST", "http://test")


class TestParsing(unittest.TestCase):
    def test_strip_plain(self):
        self.assertEqual(_strip_code_fence('["a","b"]'), '["a","b"]')

    def test_strip_json_fence(self):
        self.assertEqual(_strip_code_fence('```json\n["a","b"]\n```'), '["a","b"]')

    def test_strip_bare_fence(self):
        self.assertEqual(_strip_code_fence('```\n["a"]\n```'), '["a"]')

    def test_parse_valid(self):
        self.assertEqual(_parse_tags('["meeting-notes","budget"]'),
                         ["meeting-notes", "budget"])

    def test_parse_with_fence(self):
        self.assertEqual(_parse_tags('```json\n["todo"]\n```'), ["todo"])

    def test_parse_lowercase_dedupe_cap(self):
        out = _parse_tags('["A","a","B","c","d","e","f","g"]', max_tags=6)
        self.assertEqual(out, ["a", "b", "c", "d", "e", "f"])  # 去重+小写+截断到6

    def test_parse_invalid_json_returns_empty(self):
        self.assertEqual(_parse_tags("not json at all"), [])

    def test_parse_non_array_returns_empty(self):
        self.assertEqual(_parse_tags('{"tag":"x"}'), [])  # 对象内无 list 值 → []

    def test_parse_dict_wrapped_array(self):
        # json_object 模式：数组包在对象里 → 取第一个 list 值
        self.assertEqual(_parse_tags('{"tags":["meeting-notes","budget"]}'),
                         ["meeting-notes", "budget"])

    def test_parse_empty_array(self):
        self.assertEqual(_parse_tags("[]"), [])


class TestCloudEnricherWithFakeClient(unittest.TestCase):
    def test_returns_real_tags_no_mock_prefix(self):
        client = _FakeClient(resp=_Resp('["meeting-notes","q3-roadmap","budget"]'))
        enr = CloudEnricher(client=client)
        tags = enr.enrich("Q3 roadmap budget review", {"trigger_confidence": 0.9})
        self.assertEqual(tags, ["meeting-notes", "q3-roadmap", "budget"])
        self.assertFalse(any(t.startswith("mock:") for t in tags))  # 真标签无 mock: 前缀

    def test_default_model_is_haiku(self):
        # 决策锁定：默认模型 = claude-haiku-4-5（轻任务省成本）。可配置覆盖。
        self.assertEqual(CloudEnricher().model, "claude-haiku-4-5")

    def test_sends_expected_model_and_max_tokens(self):
        client = _FakeClient(resp=_Resp('["x"]'))
        CloudEnricher(client=client, model="claude-sonnet-4-6", max_tokens=300).enrich("t", {})
        call = client.messages.calls[0]
        self.assertEqual(call["model"], "claude-sonnet-4-6")
        self.assertEqual(call["max_tokens"], 300)
        self.assertIn("system", call)

    def test_refusal_returns_empty(self):
        client = _FakeClient(resp=_Resp("", stop_reason="refusal"))
        self.assertEqual(CloudEnricher(client=client).enrich("t", {}), [])

    def test_unparseable_response_returns_empty(self):
        client = _FakeClient(resp=_Resp("sorry, here are some tags: meeting, notes"))
        self.assertEqual(CloudEnricher(client=client).enrich("t", {}), [])

    def test_transient_api_error_raises_enricher_error(self):
        exc = anthropic.APIConnectionError(message="boom", request=_req())
        client = _FakeClient(exc=exc)
        with self.assertRaises(EnricherError):
            CloudEnricher(client=client).enrich("t", {})

    def test_config_error_raises_config_error(self):
        exc = anthropic.AuthenticationError(
            message="bad key", response=httpx.Response(401, request=_req()), body=None
        )
        client = _FakeClient(exc=exc)
        with self.assertRaises(EnricherConfigError):
            CloudEnricher(client=client).enrich("t", {})

    def test_bad_model_id_raises_config_error(self):
        # 例如误用不存在的 "claude-sonnet-5" → 404 NotFound → 配置错误（不入队）
        exc = anthropic.NotFoundError(
            message="model not found", response=httpx.Response(404, request=_req()), body=None
        )
        client = _FakeClient(exc=exc)
        with self.assertRaises(EnricherConfigError):
            CloudEnricher(client=client).enrich("t", {})

    def test_missing_key_raises_config_error(self):
        with mock.patch("pipeline.enrich.cloud_enricher.get_anthropic_api_key", return_value=None):
            with self.assertRaises(EnricherConfigError):
                CloudEnricher().enrich("t", {})  # 无注入 client、无密钥


@unittest.skipUnless(
    os.environ.get("MEMOSIGHT_LIVE_ENRICH") == "1" and os.environ.get("ANTHROPIC_API_KEY"),
    "需 MEMOSIGHT_LIVE_ENRICH=1 且有 ANTHROPIC_API_KEY 才跑真调用（省 token）",
)
class TestCloudEnricherLive(unittest.TestCase):
    def test_real_call_returns_tags(self):
        from pipeline.env import load_env
        load_env()
        enr = CloudEnricher()
        tags = enr.enrich(
            "Q3 Planning\n- Migrate gatekeeper to int8\n- Power vs miss-rate Pareto\n- Freeze by Aug",
            {"timestamp": "2026-07-06T10:00:00+00:00", "trigger_confidence": 0.93},
        )
        self.assertIsInstance(tags, list)
        self.assertTrue(all(isinstance(t, str) for t in tags))
        self.assertFalse(any(t.startswith("mock:") for t in tags))


if __name__ == "__main__":
    unittest.main()
