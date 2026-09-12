"""Tests for SQLite event storage."""
from __future__ import annotations

import pytest

from app.database.database import Database


class TestDatabaseInsert:
    def test_insert_and_retrieve(self, db: Database):
        db.insert_event(
            event_id="evt-1", event_type="weapon", detected_class="gun",
            confidence=0.91, timestamp="2026-01-01T12:00:00", camera_id="cam1",
        )
        row = db.get_event("evt-1")
        assert row is not None
        assert row["event_type"] == "weapon"
        assert row["detected_class"] == "gun"
        assert row["confidence"] == pytest.approx(0.91)
        assert row["camera_id"] == "cam1"

    def test_get_nonexistent_returns_none(self, db: Database):
        assert db.get_event("no-such-id") is None

    def test_count_events(self, db: Database):
        assert db.count_events() == 0
        db.insert_event("e1", "weapon", "knife", 0.88, "2026-01-01T12:00:00", "c1")
        db.insert_event("e2", "weapon", "gun", 0.92, "2026-01-01T12:01:00", "c1")
        assert db.count_events() == 2


class TestDatabaseList:
    def test_list_events_orders_by_timestamp_desc(self, db: Database):
        db.insert_event("e1", "weapon", "gun", 0.90, "2026-01-01T12:00:00", "c1")
        db.insert_event("e2", "weapon", "gun", 0.90, "2026-01-01T13:00:00", "c1")
        rows = db.list_events()
        assert [r["id"] for r in rows] == ["e2", "e1"]

    def test_list_events_limit_offset(self, db: Database):
        for i in range(10):
            db.insert_event(f"e{i}", "weapon", "gun", 0.9, f"2026-01-01T12:{i:02d}:00", "c1")
        rows = db.list_events(limit=3, offset=2)
        assert len(rows) == 3

    def test_list_events_filter_status(self, db: Database):
        db.insert_event("e1", "weapon", "gun", 0.9, "2026-01-01T12:00:00", "c1", status="reviewed")
        db.insert_event("e2", "weapon", "gun", 0.9, "2026-01-01T12:01:00", "c1", status="pending")
        rows = db.list_events(status="pending")
        assert len(rows) == 1
        assert rows[0]["id"] == "e2"


class TestDatabaseStatusUpdate:
    def test_set_status_reviewed(self, db: Database):
        db.insert_event("e1", "weapon", "gun", 0.9, "2026-01-01T12:00:00", "c1")
        assert db.set_status("e1", "reviewed") is True
        row = db.get_event("e1")
        assert row["status"] == "reviewed"

    def test_set_status_ignored(self, db: Database):
        db.insert_event("e1", "weapon", "gun", 0.9, "2026-01-01T12:00:00", "c1")
        assert db.set_status("e1", "ignored") is True
        assert db.get_event("e1")["status"] == "ignored"

    def test_set_status_invalid_raises(self, db: Database):
        with pytest.raises(ValueError, match="Invalid status"):
            db.set_status("e1", "mysteriously_missing")

    def test_set_status_nonexistent_returns_false(self, db: Database):
        assert db.set_status("no-such", "reviewed") is False


class TestDatabaseSetEventType:
    def test_set_event_type(self, db: Database):
        db.insert_event("e1", "weapon", "gun", 0.9, "2026-01-01T12:00:00", "c1")
        assert db.set_event_type("e1", "high_risk", extra_meta=" promoted") is True
        row = db.get_event("e1")
        assert row["event_type"] == "high_risk"
        assert "promoted" in (row["meta"] or "")


class TestDatabaseMeta:
    def test_insert_with_image_path(self, db: Database):
        db.insert_event(
            "e1", "weapon", "knife", 0.93, "2026-01-01T12:00:00", "c1",
            image_path="events/2026/01/01/evt_e1.jpg", meta="test",
        )
        row = db.get_event("e1")
        assert row["image_path"] == "events/2026/01/01/evt_e1.jpg"
        assert row["meta"] == "test"