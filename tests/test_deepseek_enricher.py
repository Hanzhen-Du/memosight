"""DeepSeekEnricher 测试。

- 解析 + 错误映射：fake OpenAI client，**不真调 API**（省 token、可离线、可 CI）。
- 集成测试：一条真 DeepSeek 调用，需 DEEPSEEK_API_KEY + MEMOSIGHT_LIVE_ENRICH=1，否则 skip。
"""

import os
import unittest
from unittest import mock

import httpx

import openai

from pipeline.enrich import DeepSeekEnricher, EnricherConfigError, EnricherError
from pipeline.enrich.deepseek_enricher import DEEPSEEK_BASE_URL


# ---- fake OpenAI client / response ----
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content, choices=None):
        self.choices = choices if choices is not None else [_Choice(content)]


class _FakeCompletions:
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._resp


class _FakeChat:
    def __init__(self, resp=None, exc=None):
        self.completions = _FakeCompletions(resp=resp, exc=exc)


class _FakeClient:
    """伪装 openai.OpenAI：只暴露 client.chat.completions.create。"""

    def __init__(self, resp=None, exc=None):
        self.chat = _FakeChat(resp=resp, exc=exc)


def _req():
    return httpx.Request("POST", "http://test")


class TestDeepSeekWithFakeClient(unittest.TestCase):
    def test_returns_real_tags_no_mock_prefix(self):
        client = _FakeClient(resp=_Resp('["meeting-notes","q3-roadmap","budget"]'))
        enr = DeepSeekEnricher(client=client)
        tags = enr.enrich("Q3 roadmap budget review", {"trigger_confidence": 0.9})
        self.assertEqual(tags, ["meeting-notes", "q3-roadmap", "budget"])
        self.assertFalse(any(t.startswith("mock:") for t in tags))

    def test_parses_json_object_wrapped_array(self):
        # json_object 模式常返回 {"tags":[...]} —— 解析层已容忍
        client = _FakeClient(resp=_Resp('{"tags": ["invoice","billing"]}'))
        tags = DeepSeekEnricher(client=client).enrich("INVOICE ...", {})
        self.assertEqual(tags, ["invoice", "billing"])

    def test_default_model_is_deepseek_v4_flash(self):
        self.assertEqual(DeepSeekEnricher().model, "deepseek-v4-flash")

    def test_default_base_url_is_deepseek(self):
        self.assertEqual(DeepSeekEnricher().base_url, DEEPSEEK_BASE_URL)

    def test_sends_expected_model_maxtokens_and_json_format(self):
        client = _FakeClient(resp=_Resp('["x"]'))
        DeepSeekEnricher(client=client, model="deepseek-v4-flash", max_tokens=300).enrich("t", {})
        call = client.chat.completions.calls[0]
        self.assertEqual(call["model"], "deepseek-v4-flash")
        self.assertEqual(call["max_tokens"], 300)
        self.assertEqual(call["response_format"], {"type": "json_object"})
        # 首条消息是 system prompt
        self.assertEqual(call["messages"][0]["role"], "system")

    def test_no_content_returns_empty(self):
        client = _FakeClient(resp=_Resp(None))  # 内容为 None（如被过滤）
        self.assertEqual(DeepSeekEnricher(client=client).enrich("t", {}), [])

    def test_no_choices_returns_empty(self):
        client = _FakeClient(resp=_Resp(None, choices=[]))
        self.assertEqual(DeepSeekEnricher(client=client).enrich("t", {}), [])

    def test_unparseable_returns_empty(self):
        client = _FakeClient(resp=_Resp("sorry, tags: meeting, notes"))
        self.assertEqual(DeepSeekEnricher(client=client).enrich("t", {}), [])

    def test_transient_api_error_raises_enricher_error(self):
        exc = openai.APIConnectionError(message="boom", request=_req())
        client = _FakeClient(exc=exc)
        with self.assertRaises(EnricherError):
            DeepSeekEnricher(client=client).enrich("t", {})

    def test_rate_limit_raises_enricher_error(self):
        exc = openai.RateLimitError(message="rl", response=httpx.Response(429, request=_req()), body=None)
        client = _FakeClient(exc=exc)
        with self.assertRaises(EnricherError):
            DeepSeekEnricher(client=client).enrich("t", {})

    def test_auth_error_raises_config_error(self):
        exc = openai.AuthenticationError(
            message="bad key", response=httpx.Response(401, request=_req()), body=None
        )
        client = _FakeClient(exc=exc)
        with self.assertRaises(EnricherConfigError):
            DeepSeekEnricher(client=client).enrich("t", {})

    def test_bad_model_raises_config_error(self):
        exc = openai.NotFoundError(
            message="model not found", response=httpx.Response(404, request=_req()), body=None
        )
        client = _FakeClient(exc=exc)
        with self.assertRaises(EnricherConfigError):
            DeepSeekEnricher(client=client).enrich("t", {})

    def test_missing_key_raises_config_error(self):
        with mock.patch("pipeline.enrich.deepseek_enricher.get_deepseek_api_key", return_value=None):
            with self.assertRaises(EnricherConfigError):
                DeepSeekEnricher().enrich("t", {})


@unittest.skipUnless(
    os.environ.get("MEMOSIGHT_LIVE_ENRICH") == "1" and os.environ.get("DEEPSEEK_API_KEY"),
    "需 MEMOSIGHT_LIVE_ENRICH=1 且有 DEEPSEEK_API_KEY 才跑真调用（省 token）",
)
class TestDeepSeekLive(unittest.TestCase):
    def test_real_call_returns_tags(self):
        from pipeline.env import load_env
        load_env()
        enr = DeepSeekEnricher()
        tags = enr.enrich(
            "Q3 Planning\n- Migrate gatekeeper to int8\n- Power vs miss-rate Pareto\n- Freeze by Aug",
            {"timestamp": "2026-07-06T10:00:00+00:00", "trigger_confidence": 0.93},
        )
        self.assertIsInstance(tags, list)
        self.assertTrue(all(isinstance(t, str) for t in tags))
        self.assertFalse(any(t.startswith("mock:") for t in tags))


if __name__ == "__main__":
    unittest.main()
