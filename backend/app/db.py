"""SQLite helpers for persisting sentiment classifications."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterable


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sentiment_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL
);
"""

INSERT_SQL = """
INSERT INTO sentiment_history (text, sentiment, confidence, created_at)
VALUES (?, ?, ?, ?)
"""

SELECT_SQL = """
SELECT id, text, sentiment, confidence, created_at
FROM sentiment_history
ORDER BY id DESC
LIMIT ?
"""

DELETE_OLD_SQL = """
DELETE FROM sentiment_history
WHERE id NOT IN (
    SELECT id FROM sentiment_history ORDER BY id DESC LIMIT ?
)
"""


@dataclass(slots=True)
class HistoryEntry:
    id: int
    text: str
    sentiment: str
    confidence: float
    created_at: datetime


class HistoryRepository:
    """Encapsulates SQLite operations."""

    def __init__(self, db_path: Path, retain: int) -> None:
        self.db_path = db_path
        self.retain = retain
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)
            conn.commit()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def add_entry(self, text: str, sentiment: str, confidence: float) -> HistoryEntry:
        timestamp = datetime.now(timezone.utc)
        with self._connect() as conn:
            cursor = conn.execute(
                INSERT_SQL,
                (text, sentiment, confidence, timestamp.isoformat()),
            )
            conn.commit()
            entry = HistoryEntry(
                id=cursor.lastrowid,
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                created_at=timestamp,
            )
        self._trim_old_rows()
        return entry

    def list_entries(self, limit: int) -> list[HistoryEntry]:
        with self._connect() as conn:
            rows = conn.execute(SELECT_SQL, (limit,)).fetchall()
        return [
            HistoryEntry(
                id=row["id"],
                text=row["text"],
                sentiment=row["sentiment"],
                confidence=float(row["confidence"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def _trim_old_rows(self) -> None:
        if self.retain <= 0:
            return
        with self._connect() as conn:
            conn.execute(DELETE_OLD_SQL, (self.retain,))
            conn.commit()

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sentiment_history")
            conn.commit()
