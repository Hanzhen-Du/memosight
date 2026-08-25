"""M4: tests for the three offline-queue paths, plus transport and connectivity."""

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


def make_payload(text="design review whiteboard", conf=0.9):
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

    # Path 1: online, stored directly
    def test_online_stores_done_with_tags(self):
        card = self.svc.ingest(make_payload())
        self.assertEqual(card.status, config.STATUS_DONE)
        self.assertEqual(card.tags, ["mock:meeting", "mock:whiteboard"])
        self.assertIsNotNone(card.enriched_at)
        self.assertEqual(self.store.count(config.STATUS_PENDING), 0)
        self.assertEqual(self.store.count(config.STATUS_DONE), 1)

    # Path 2: offline, queued
    def test_offline_queues_pending_without_tags(self):
        self.conn.go_offline()
        card = self.svc.ingest(make_payload())
        self.assertEqual(card.status, config.STATUS_PENDING)
        self.assertIsNone(card.tags)
        self.assertIsNone(card.enriched_at)
        self.assertEqual(self.store.count(config.STATUS_PENDING), 1)

    # Path 3: bulk backfill once connectivity returns
    def test_recovery_backfills_all_pending(self):
        self.conn.go_offline()
        self.svc.ingest(make_payload("card A"))
        self.svc.ingest(make_payload("card B"))
        self.assertEqual(self.store.count(config.STATUS_PENDING), 2)

        # Backfilling while offline should be a no-op
        self.assertEqual(self.svc.process_pending(), [])
        self.assertEqual(self.store.count(config.STATUS_PENDING), 2)

        # Back online, so the queue is backfilled
        self.conn.go_online()
        done = self.svc.process_pending()
        self.assertEqual(len(done), 2)
        for c in done:
            self.assertEqual(c.status, config.STATUS_DONE)
            self.assertEqual(c.tags, ["mock:meeting", "mock:whiteboard"])
            self.assertIsNotNone(c.enriched_at)
        self.assertEqual(self.store.count(config.STATUS_PENDING), 0)
        self.assertEqual(self.store.count(config.STATUS_DONE), 2)

    # Cloud failure: online but the enricher raises, so it falls back to queuing and never
    # fabricates tags
    def test_cloud_failure_falls_back_to_queue(self):
        failing = IngestService(
            self.store, DirectUploadMock(MockCloudEnricher(fail=True)), self.conn
        )
        card = failing.ingest(make_payload())
        self.assertEqual(card.status, config.STATUS_PENDING)
        self.assertIsNone(card.tags)
        self.assertEqual(self.store.count(config.STATUS_PENDING), 1)

    # A card left pending by a cloud failure must not be wrongly marked done during backfill
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
        self.assertIsNone(card.id)  # not stored yet


if __name__ == "__main__":
    unittest.main()
