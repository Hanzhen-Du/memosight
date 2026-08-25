"""Tests for the real CloudEnricher.

- Parsing and error mapping use a fake client and make no real API call, so they cost nothing,
  run offline and work in CI.
- One integration test makes a real call. It needs ANTHROPIC_API_KEY plus the explicit switch
  MEMOSIGHT_LIVE_ENRICH=1, and skips otherwise.
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
        self.assertEqual(out, ["a", "b", "c", "d", "e", "f"])  # deduplicated, lowercased, capped at 6

    def test_parse_invalid_json_returns_empty(self):
        self.assertEqual(_parse_tags("not json at all"), [])

    def test_parse_non_array_returns_empty(self):
        self.assertEqual(_parse_tags('{"tag":"x"}'), [])  # no list value in the object, so []

    def test_parse_dict_wrapped_array(self):
        # json_object mode wraps the array in an object, so the first list value is taken
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
        self.assertFalse(any(t.startswith("mock:") for t in tags))  # real tags carry no mock: prefix

    def test_default_model_is_haiku(self):
        # Locked-in decision: the default model is claude-haiku-4-5, chosen to keep a light
        # task cheap. It stays configurable.
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
        # For example, using a model id that does not exist gives 404 NotFound, which is a
        # configuration error and is not queued
        exc = anthropic.NotFoundError(
            message="model not found", response=httpx.Response(404, request=_req()), body=None
        )
        client = _FakeClient(exc=exc)
        with self.assertRaises(EnricherConfigError):
            CloudEnricher(client=client).enrich("t", {})

    def test_missing_key_raises_config_error(self):
        with mock.patch("pipeline.enrich.cloud_enricher.get_anthropic_api_key", return_value=None):
            with self.assertRaises(EnricherConfigError):
                CloudEnricher().enrich("t", {})  # no injected client and no key


@unittest.skipUnless(
    os.environ.get("MEMOSIGHT_LIVE_ENRICH") == "1" and os.environ.get("ANTHROPIC_API_KEY"),
    "a real call needs MEMOSIGHT_LIVE_ENRICH=1 and ANTHROPIC_API_KEY; skipped to avoid cost",
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
