"""M4: 断网队列三条路径 + transport/connectivity 测试。"""

import tempfile
import unittest
from pathlib import Path

from pipeline import config
from pipeline.connectivity import ConnectivityMock
from pipeline.db import CardStore
from pipeline.enrich import MockCloudEnricher
from pipeline.ingest import IngestService
from pipeline.packaging import build_payload
from pipeline.transport import DirectUploadMock


def make_payload(text="设计评审 白板", conf=0.9):
    return build_payload(text, "2026-07-06T10:00:00+00:00", conf)


class TestQueuePaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CardStore(Path(self.tmp.name) / "q.db")
        self.enricher = MockCloudEnricher(canned_tags=["mock:meeting", "mock:whiteboard"])
        self.transport = DirectUploadMock(self.enricher)
        self.conn = ConnectivityMock(online=True)
        self.svc = IngestService(self.store, self.transport, self.conn)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    # 路径 1：联网直存
    def test_online_stores_done_with_tags(self):
        card = self.svc.ingest(make_payload())
        self.assertEqual(card.status, config.STATUS_DONE)
        self.assertEqual(card.tags, ["mock:meeting", "mock:whiteboard"])
        self.assertIsNotNone(card.enriched_at)
        self.assertEqual(self.store.count(config.STATUS_PENDING), 0)
        self.assertEqual(self.store.count(config.STATUS_DONE), 1)

    # 路径 2：断网入队
    def test_offline_queues_pending_without_tags(self):
        self.conn.go_offline()
        card = self.svc.ingest(make_payload())
        self.assertEqual(card.status, config.STATUS_PENDING)
        self.assertIsNone(card.tags)
        self.assertIsNone(card.enriched_at)
        self.assertEqual(self.store.count(config.STATUS_PENDING), 1)

    # 路径 3：恢复联网批量补传
    def test_recovery_backfills_all_pending(self):
        self.conn.go_offline()
        self.svc.ingest(make_payload("卡片A"))
        self.svc.ingest(make_payload("卡片B"))
        self.assertEqual(self.store.count(config.STATUS_PENDING), 2)

        # 断网时补传应为 no-op
        self.assertEqual(self.svc.process_pending(), [])
        self.assertEqual(self.store.count(config.STATUS_PENDING), 2)

        # 恢复联网 → 批量补齐
        self.conn.go_online()
        done = self.svc.process_pending()
        self.assertEqual(len(done), 2)
        for c in done:
            self.assertEqual(c.status, config.STATUS_DONE)
            self.assertEqual(c.tags, ["mock:meeting", "mock:whiteboard"])
            self.assertIsNotNone(c.enriched_at)
        self.assertEqual(self.store.count(config.STATUS_PENDING), 0)
        self.assertEqual(self.store.count(config.STATUS_DONE), 2)

    # 云端失败：联网但 enricher 抛错 → 回退入队，不伪造 tags
    def test_cloud_failure_falls_back_to_queue(self):
        failing = IngestService(
            self.store, DirectUploadMock(MockCloudEnricher(fail=True)), self.conn
        )
        card = failing.ingest(make_payload())
        self.assertEqual(card.status, config.STATUS_PENDING)
        self.assertIsNone(card.tags)
        self.assertEqual(self.store.count(config.STATUS_PENDING), 1)

    # 云端失败的 pending 在补传时也不被误标 done
    def test_recovery_leaves_failing_cloud_pending(self):
        self.conn.go_offline()
        self.svc.ingest(make_payload())
        self.conn.go_online()
        failing = IngestService(
            self.store, DirectUploadMock(MockCloudEnricher(fail=True)), self.conn
        )
        self.assertEqual(failing.process_pending(), [])
        self.assertEqual(self.store.count(config.STATUS_PENDING), 1)


class TestTransport(unittest.TestCase):
    def test_direct_upload_returns_done_card(self):
        tr = DirectUploadMock(MockCloudEnricher(canned_tags=["a"]))
        card = tr.upload(make_payload())
        self.assertEqual(card.status, config.STATUS_DONE)
        self.assertEqual(card.tags, ["a"])
        self.assertIsNone(card.id)  # 尚未入库


if __name__ == "__main__":
    unittest.main()
