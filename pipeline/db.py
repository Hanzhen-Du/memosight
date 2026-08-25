"""SQLite storage: schema, CRUD and the pending queue.

Design notes:
- A single-file SQLite database, in keeping with local-first and offline operation.
- One CardStore wraps the connection and all the SQL. Callers see MemoryCard objects and never
  SQL.
- The pending queue is not a separate table, it is simply the rows with status='pending', which
  makes backfilling tags an in-place UPDATE.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from . import config
from .models import MemoryCard, utc_now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_cards (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT    NOT NULL,
    trigger_confidence REAL    NOT NULL,
    ocr_text           TEXT    NOT NULL,
    tags               TEXT,              -- JSON array; NULL means not generated yet
    raw_image_policy   TEXT    NOT NULL DEFAULT 'delete',
    status             TEXT    NOT NULL DEFAULT 'pending',
    created_at         TEXT    NOT NULL,
    enriched_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_cards_status ON memory_cards(status);
CREATE INDEX IF NOT EXISTS idx_cards_created ON memory_cards(created_at);
"""


class CardStore:
    """SQLite repository for memory cards. Use as a context manager, or close() manually."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because the CLI and tests occasionally cross threads. There
        # is a single writer in this phase, so this is sufficient.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---- context manager ----
    def __enter__(self) -> "CardStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # ---- Create ----
    def insert(self, card: MemoryCard) -> int:
        """Insert a card and return the assigned id, also writing it back to card.id."""
        row = card.to_row()
        cur = self._conn.execute(
            """
            INSERT INTO memory_cards
                (timestamp, trigger_confidence, ocr_text, tags,
                 raw_image_policy, status, created_at, enriched_at)
            VALUES
                (:timestamp, :trigger_confidence, :ocr_text, :tags,
                 :raw_image_policy, :status, :created_at, :enriched_at)
            """,
            row,
        )
        self._conn.commit()
        card.id = int(cur.lastrowid)
        return card.id

    # ---- Read ----
    def get(self, card_id: int) -> Optional[MemoryCard]:
        cur = self._conn.execute(
            "SELECT * FROM memory_cards WHERE id = ?", (card_id,)
        )
        row = cur.fetchone()
        return MemoryCard.from_row(row) if row else None

    def list_all(self, limit: Optional[int] = None) -> list[MemoryCard]:
        sql = "SELECT * FROM memory_cards ORDER BY created_at DESC, id DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [MemoryCard.from_row(r) for r in self._conn.execute(sql)]

    def list_pending(self) -> list[MemoryCard]:
        """The pending queue: rows with status='pending', oldest first, so they are
        backfilled in arrival order."""
        cur = self._conn.execute(
            "SELECT * FROM memory_cards WHERE status = ? ORDER BY created_at ASC, id ASC",
            (config.STATUS_PENDING,),
        )
        return [MemoryCard.from_row(r) for r in cur]

    def search(self, keyword: str, limit: Optional[int] = None) -> list[MemoryCard]:
        """Search ocr_text or tags by keyword, case-insensitive substring match.

        tags is stored as JSON text, and a plain LIKE is enough for this phase; no full-text
        index is needed.
        """
        like = f"%{keyword}%"
        sql = (
            "SELECT * FROM memory_cards "
            "WHERE ocr_text LIKE ? COLLATE NOCASE "
            "   OR IFNULL(tags, '') LIKE ? COLLATE NOCASE "
            "ORDER BY created_at DESC, id DESC"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        cur = self._conn.execute(sql, (like, like))
        return [MemoryCard.from_row(r) for r in cur]

    def count(self, status: Optional[str] = None) -> int:
        if status is None:
            cur = self._conn.execute("SELECT COUNT(*) AS n FROM memory_cards")
        else:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM memory_cards WHERE status = ?", (status,)
            )
        return int(cur.fetchone()["n"])

    # ---- Update ----
    def enrich_card(self, card_id: int, tags: Iterable[str]) -> MemoryCard:
        """Backfill tags: write tags, set status to done, fill in enriched_at, and return the
        updated card."""
        card = self.get(card_id)
        if card is None:
            raise KeyError(f"memory card id={card_id} does not exist")
        card.tags = list(tags)
        card.status = config.STATUS_DONE
        card.enriched_at = utc_now_iso()
        self._conn.execute(
            "UPDATE memory_cards SET tags = ?, status = ?, enriched_at = ? WHERE id = ?",
            (card.tags_json(), card.status, card.enriched_at, card_id),
        )
        self._conn.commit()
        return card

    # ---- Delete ----
    def delete(self, card_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM memory_cards WHERE id = ?", (card_id,))
        self._conn.commit()
        return cur.rowcount > 0
