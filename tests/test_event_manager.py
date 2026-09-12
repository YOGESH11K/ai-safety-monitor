"""Tests for EventManager cooldown and event creation."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.config import ConfirmationConfig, EventConfig, DetectionConfig
from app.detection.base import Detection, FrameResult
from app.events.event_manager import EventManager, TemporalTracker, ConfirmedEvent, now_iso, new_event_id
from app.database.database import Database


def _det(conf: float, cls: str = "gun") -> Detection:
    return Detection(class_name=cls, confidence=conf, bbox=(10, 10, 50, 50), label=f"{cls} {conf:.2f}")

def _frame(*confs: float) -> FrameResult:
    return FrameResult(frame_id=0, detections=[_det(c) for c in confs])


class TestEventCooldown:
    def test_fires_immediately_then_cooldown(self, db: Database):
        """First event fires; second within cooldown is blocked."""
        fired: list[ConfirmedEvent] = []
        mgr = EventManager(
            event_config=EventConfig(cooldown_seconds=5.0, save_evidence=False),
            confirmation_config=ConfirmationConfig(window=3, min_positive=2),
            confidence_threshold=0.70,
            camera_id="c1",
            store=lambda e: fired.append(e),
        )
        import app.events.event_manager as ev
        ev.TEST_MODE = True

        # build 3 positives to trigger
        for _ in range(3):
            assert mgr.update(_frame(0.85)) is None
        assert mgr.update(_frame(0.85)) is not None  # event 1
        assert len(fired) == 1

        # Next frames still positive but within cooldown
        mgr.tracker.reset()  # restart window for new confirmation
        for _ in range(3):
            assert mgr.update(_frame(0.85)) is None
        result2 = mgr.update(_frame(0.85))
        assert result2 is None  # still within cooldown (5s)
        assert len(fired) == 1

    def test_fires_after_cooldown_elapses(self, db: Database):
        fired: list[ConfirmedEvent] = []
        mgr = EventManager(
            event_config=EventConfig(cooldown_seconds=0.0, save_evidence=False),  # instant
            confirmation_config=ConfirmationConfig(window=3, min_positive=2),
            confidence_threshold=0.70,
            camera_id="c1",
            store=lambda e: fired.append(e),
        )
        for _ in range(3):
            mgr.update(_frame(0.85))
        mgr.update(_frame(0.85))
        assert len(fired) == 1

        mgr.tracker.reset()
        for _ in range(3):
            mgr.update(_frame(0.85))
        mgr.update(_frame(0.85))
        assert len(fired) == 2

    def test_event_stored_with_correct_fields(self, db: Database):
        stored = []
        mgr = EventManager(
            event_config=EventConfig(cooldown_seconds=0.0, save_evidence=False),
            confirmation_config=ConfirmationConfig(window=3, min_positive=2),
            confidence_threshold=0.70,
            camera_id="test-cam",
            store=lambda e: stored.append(e),
        )
        for _ in range(3):
            mgr.update(_frame(0.85))
        evt = mgr.update(_frame(0.85))
        assert evt is not None
        assert evt.camera_id == "test-cam"
        assert evt.event_type == "weapon"
        assert evt.detected_class == "gun"
        assert 0.84 <= evt.confidence <= 0.86
        assert evt.timestamp

    def test_event_persisted_to_database(self, db: Database):
        mgr = EventManager(
            event_config=EventConfig(cooldown_seconds=0.0, save_evidence=False),
            confirmation_config=ConfirmationConfig(window=3, min_positive=2),
            confidence_threshold=0.70,
            camera_id="db-cam",
            store=lambda e: db.insert_event(
                event_id=e.event_id, event_type=e.event_type, detected_class=e.detected_class,
                confidence=e.confidence, timestamp=e.timestamp, camera_id=e.camera_id, image_path=e.image_path,
            ),
        )
        for _ in range(3):
            mgr.update(_frame(0.85))
        mgr.update(_frame(0.85))
        assert db.count_events() == 1
        rows = db.list_events()
        assert rows[0]["camera_id"] == "db-cam"

    def test_does_not_fire_when_negatives_break_window(self):
        fired: list[ConfirmedEvent] = []
        mgr = EventManager(
            event_config=EventConfig(cooldown_seconds=0.0, save_evidence=False),
            confirmation_config=ConfirmationConfig(window=3, min_positive=2),
            confidence_threshold=0.70,
            camera_id="c1",
            store=lambda e: fired.append(e),
        )
        mgr.update(_frame(0.85))
        mgr.update(_frame())  # neg
        mgr.update(_frame())  # neg
        mgr.update(_frame(0.85))
        mgr.update(_frame())  # still window has 2 positives but never full window with enough → might not
        assert len(fired) == 0

    def test_reset_cooldown_allows_immediate_fire(self):
        fired: list[ConfirmedEvent] = []
        mgr = EventManager(
            event_config=EventConfig(cooldown_seconds=99.0, save_evidence=False),
            confirmation_config=ConfirmationConfig(window=3, min_positive=2),
            confidence_threshold=0.70,
            camera_id="c1",
            store=lambda e: fired.append(e),
        )
        for _ in range(3):
            mgr.update(_frame(0.85))
        mgr.update(_frame(0.85))
        assert len(fired) == 1
        mgr.reset_cooldown()
        mgr.tracker.reset()
        for _ in range(3):
            mgr.update(_frame(0.85))
        mgr.update(_frame(0.85))
        assert len(fired) == 2