"""M3: Enricher 接口 + MockCloudEnricher(mock) + 打包逻辑测试。"""

import unittest

from pipeline.enrich import MockCloudEnricher, EnricherError, EnricherInterface
from pipeline.packaging import Payload, build_payload


class TestPackaging(unittest.TestCase):
    def test_build_payload_and_metadata(self):
        p = build_payload("白板文本", "2026-07-06T10:00:00+00:00", 0.88)
        self.assertIsInstance(p, Payload)
        md = p.metadata()
        self.assertEqual(
            set(md), {"timestamp", "trigger_confidence", "raw_image_policy"}
        )
        self.assertNotIn("ocr_text", md)          # ocr_text 不在 metadata 里
        self.assertEqual(md["raw_image_policy"], "delete")

    def test_to_dict_includes_ocr(self):
        p = build_payload("hello", "2026-07-06T10:00:00+00:00", 0.5)
        self.assertEqual(p.to_dict()["ocr_text"], "hello")

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            build_payload("x", "2026-07-06T10:00:00+00:00", 0.5, raw_image_policy="hoard")


class TestMockCloudEnricher(unittest.TestCase):
    def test_is_interface(self):
        self.assertIsInstance(MockCloudEnricher(), EnricherInterface)

    def test_returns_mock_tags(self):
        enr = MockCloudEnricher()
        tags = enr.enrich("Kubernetes 部署架构图", {"trigger_confidence": 0.9})
        self.assertIsInstance(tags, list)
        self.assertTrue(tags)
        # 所有生成标签带 mock: 前缀，诚实标注为假标签
        self.assertTrue(all(t.startswith("mock:") for t in tags))

    def test_deterministic_for_same_input(self):
        enr = MockCloudEnricher()
        md = {"trigger_confidence": 0.9}
        self.assertEqual(enr.enrich("同一段文本", md), enr.enrich("同一段文本", md))

    def test_confidence_reflected(self):
        enr = MockCloudEnricher()
        hi = enr.enrich("t", {"trigger_confidence": 0.95})
        lo = enr.enrich("t", {"trigger_confidence": 0.30})
        self.assertIn("mock:high-conf", hi)
        self.assertIn("mock:low-conf", lo)

    def test_canned_tags(self):
        enr = MockCloudEnricher(canned_tags=["a", "b"])
        self.assertEqual(enr.enrich("whatever", {}), ["a", "b"])

    def test_simulated_failure_raises(self):
        with self.assertRaises(EnricherError):
            MockCloudEnricher(fail=True).enrich("x", {})


if __name__ == "__main__":
    unittest.main()
