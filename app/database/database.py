"""SQLite storage for confirmed detections/events."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    detected_class  TEXT,
    confidence      REAL NOT NULL,
    timestamp       TEXT NOT NULL,
    image_path      TEXT,
    camera_id       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',   -- pending | reviewed | ignored
    meta            TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
"""


def utcnow_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Database:
    def __init__(self, db_path: Path | str = "data/safety.db") -> None:
        self.db_path = Path(str(db_path))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def insert_event(
        self,
        event_id: str,
        event_type: str,
        detected_class: str,
        confidence: float,
        timestamp: str,
        camera_id: str,
        image_path: str = "",
        meta: str = "",
        status: str = "pending",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (id, event_type, detected_class, confidence, timestamp,"
                " image_path, camera_id, status, meta, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    event_type,
                    detected_class,
                    float(confidence),
                    timestamp,
                    image_path,
                    camera_id,
                    status,
                    meta,
                    utcnow_iso(),
                ),
            )
            self._conn.commit()

    def list_events(self, limit: int = 50, offset: int = 0, status: Optional[str] = None) -> list[dict]:
        query = "SELECT * FROM events"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        return self.fetch_all(query, params)

    def get_event(self, event_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return dict(row) if row else None

    def set_status(self, event_id: str, status: str) -> bool:
        statuses = {"pending", "reviewed", "ignored"}
        if status not in statuses:
            raise ValueError(f"Invalid status: {status!r}. Must be one of {statuses}")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE events SET status = ? WHERE id = ?", (status, event_id)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def set_event_type(self, event_id: str, event_type: str, extra_meta: str = "") -> bool:
        with self._lock:
            if extra_meta:
                self._conn.execute(
                    "UPDATE events SET event_type = ?, meta = COALESCE(meta, '') || ? WHERE id = ?",
                    (event_type, extra_meta, event_id),
                )
            else:
                self._conn.execute(
                    "UPDATE events SET event_type = ? WHERE id = ?", (event_type, event_id)
                )
            self._conn.commit()
        return True

    def count_events(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
        return int(row["n"])

    def fetch_all(self, query: str, params: tuple | list = ()) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()