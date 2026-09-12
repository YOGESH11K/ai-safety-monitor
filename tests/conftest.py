"""Shared fixtures and helpers for the test suite."""
from __future__ import annotations

import sys
import os
from pathlib import Path

import pytest

# Ensure project root is on sys.path so tests can import app.* cleanly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig, CameraConfig, DetectionConfig, ConfirmationConfig
from app.database.database import Database

# Put events/events_test.py in a temp-safe location.
from app.events import event_manager as _ev_mod

_fixtures_dir = Path(__file__).resolve().parent


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    p = tmp_path / "test.db"
    database = Database(db_path=p)
    yield database
    database.close()


@pytest.fixture()
def config_minimal() -> AppConfig:
    return AppConfig(
        camera_id="test-cam",
        log_level="DEBUG",
        camera=CameraConfig(source="synthetic", width=320, height=240, fps=10),
        detection=DetectionConfig(
            detector_mode="stub",
            confidence_threshold=0.70,
            weapon_classes=["gun", "knife"],
        ),
        confirmation=ConfirmationConfig(window=3, min_positive=2),
    )


@pytest.fixture()
def detection_config_stub() -> DetectionConfig:
    return DetectionConfig(
        detector_mode="stub",
        confidence_threshold=0.70,
        weapon_classes=["gun", "knife"],
    )


@pytest.fixture()
def confirmation_config() -> ConfirmationConfig:
    return ConfirmationConfig(window=5, min_positive=3)


@pytest.fixture(autouse=True)
def _reset_event_manager_test_mode():
    """Ensure deterministic event IDs in tests."""
    _ev_mod.TEST_MODE = True
    yield
    _ev_mod.TEST_MODE = False


@pytest.fixture()
def evidence_root(tmp_path: Path) -> Path:
    """Return a fresh evidence directory."""
    root = tmp_path / "ev"
    root.mkdir()
    return root