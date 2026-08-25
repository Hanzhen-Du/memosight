"""M1: SQLite schema + 数据模型 + 增删查 单元测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline import config
from pipeline.db import CardStore
from pipeline.models import MemoryCard


def sample_card(**overrides) -> MemoryCard:
    base = dict(
        timestamp="2026-07-06T10:00:00+00:00",
        trigger_confidence=0.91,
        ocr_text="设计评审会 白板 Q3 路线图",
    )
    base.update(overrides)
    return MemoryCard(**base)


class TestMemoryCardModel(unittest.TestCase):
    def test_tags_json_roundtrip(self):
        card = sample_card(tags=["meeting", "白板", "roadmap"])
        raw = card.tags_json()
        self.assertEqual(json.loads(raw), ["meeting", "白板", "roadmap"])
        self.assertEqual(MemoryCard.tags_from_json(raw), ["meeting", "白板", "roadmap"])

    def test_tags_none_stays_none(self):
        card = sample_card()
        self.assertIsNone(card.tags)
        self.assertIsNone(card.tags_json())
        self.assertIsNone(MemoryCard.tags_from_json(None))

    def test_defaults(self):
        card = sample_card()
        self.assertEqual(card.raw_image_policy, config.POLICY_DELETE)
        self.assertEqual(card.status, config.STATUS_PENDING)
        self.assertIsNotNone(card.created_at)
        self.assertIsNone(card.enriched_at)

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            sample_card(raw_image_policy="keep_forever")

    def test_invalid_status_rejected(self):
        with self.assertRaises(ValueError):
            sample_card(status="halfway")


class TestCardStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        self.store = CardStore(self.db_path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_insert_and_get(self):
        card = sample_card()
        cid = self.store.insert(card)
        self.assertIsInstance(cid, int)
        self.assertEqual(card.id, cid)

        got = self.store.get(cid)
        self.assertIsNotNone(got)
        self.assertEqual(got.ocr_text, card.ocr_text)
        self.assertAlmostEqual(got.trigger_confidence, 0.91)
        self.assertEqual(got.status, config.STATUS_PENDING)
        self.assertIsNone(got.tags)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get(9999))

    def test_list_all_order_and_limit(self):
        for i in range(3):
            self.store.insert(sample_card(ocr_text=f"card {i}"))
        cards = self.store.list_all()
        self.assertEqual(len(cards), 3)
        limited = self.store.list_all(limit=2)
        self.assertEqual(len(limited), 2)

    def test_search_ocr_and_tags(self):
        c1 = sample_card(ocr_text="Kubernetes 部署架构图")
        c2 = sample_card(ocr_text="午餐菜单")
        self.store.insert(c1)
        cid2 = self.store.insert(c2)
        # 给 c2 打上 tags 再按 tag 搜
        self.store.enrich_card(cid2, ["food", "restaurant", "菜单"])

        by_text = self.store.search("kubernetes")  # 大小写不敏感
        self.assertEqual(len(by_text), 1)
        self.assertIn("Kubernetes", by_text[0].ocr_text)

        by_tag = self.store.search("restaurant")
        self.assertEqual(len(by_tag), 1)
        self.assertEqual(by_tag[0].id, cid2)

        none = self.store.search("量子计算")
        self.assertEqual(none, [])

    def test_enrich_updates_status_and_tags(self):
        cid = self.store.insert(sample_card())
        self.assertEqual(self.store.count(config.STATUS_PENDING), 1)

        updated = self.store.enrich_card(cid, ["meeting", "whiteboard"])
        self.assertEqual(updated.tags, ["meeting", "whiteboard"])
        self.assertEqual(updated.status, config.STATUS_DONE)
        self.assertIsNotNone(updated.enriched_at)

        reread = self.store.get(cid)
        self.assertEqual(reread.status, config.STATUS_DONE)
        self.assertEqual(reread.tags, ["meeting", "whiteboard"])
        self.assertEqual(self.store.count(config.STATUS_PENDING), 0)
        self.assertEqual(self.store.count(config.STATUS_DONE), 1)

    def test_enrich_missing_raises(self):
        with self.assertRaises(KeyError):
            self.store.enrich_card(4242, ["x"])

    def test_list_pending_fifo(self):
        ids = [self.store.insert(sample_card(ocr_text=f"c{i}")) for i in range(3)]
        # 补第二张
        self.store.enrich_card(ids[1], ["done"])
        pending = self.store.list_pending()
        self.assertEqual([c.id for c in pending], [ids[0], ids[2]])

    def test_delete(self):
        cid = self.store.insert(sample_card())
        self.assertTrue(self.store.delete(cid))
        self.assertIsNone(self.store.get(cid))
        self.assertFalse(self.store.delete(cid))  # 二次删除返回 False

    def test_persist_across_reopen(self):
        cid = self.store.insert(sample_card(ocr_text="持久化验证"))
        self.store.close()
        store2 = CardStore(self.db_path)
        try:
            got = store2.get(cid)
            self.assertIsNotNone(got)
            self.assertEqual(got.ocr_text, "持久化验证")
        finally:
            store2.close()


if __name__ == "__main__":
    unittest.main()
