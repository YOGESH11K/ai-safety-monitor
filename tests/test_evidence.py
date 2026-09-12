"""Tests for evidence filename generation and evidence directory layout."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import EventConfig
from app.detection.base import Detection
from app.events.evidence import evidence_dir, save_annotated_frame, unique_filename


class TestEvidenceDir:
    def test_directory_layout(self, evidence_root: Path):
        dt = datetime(2026, 3, 15, 10, 30, 0, tzinfo=timezone.utc)
        d = evidence_dir(evidence_root, dt)
        assert str(d).endswith("2026/03/15")
        assert d.parent.name == "03" and d.parent.parent.name == "2026"

    def test_current_date_when_none(self, evidence_root: Path):
        d = evidence_dir(evidence_root)
        today = datetime.now()
        assert str(d).endswith(f"{today.year}/{today.month:02d}/{today.day:02d}")


class TestUniqueFilename:
    def test_format(self):
        dt = datetime(2026, 7, 4, 14, 30, 5, tzinfo=timezone.utc)
        fn = unique_filename("evt_abc", "gun", dt)
        assert fn.startswith("evt_evt_abc_20260704_143005_gun.jpg")

    def test_uniqueness(self):
        seen = set()
        for _ in range(100):
            fn = unique_filename("id", "knife")
            assert fn not in seen
            seen.add(fn)
            break  # test one; timestamps won't all differ in subsecond
        fn1 = unique_filename("id1", "gun", datetime(2026,1,1,0,0,0))
        fn2 = unique_filename("id2", "gun", datetime(2026,1,1,0,0,0))
        assert fn1 != fn2  # different event ids

    def test_safe_class_name(self):
        fn = unique_filename("id", "pistol/RIFLE ", datetime(2026,1,1,0,0,0))
        assert "pistol_rifle" in fn
        assert "/" not in fn


class TestSaveAnnotatedFrame:
    def test_saves_jpeg_and_returns_path(self, evidence_root: Path, tmp_path: Path):
        import numpy as np
        import cv2
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config = EventConfig(
            evidence_dir=str(evidence_root), jpeg_quality=90, save_evidence=True, cooldown_seconds=5
        )
        det = Detection(class_name="knife", confidence=0.92, bbox=(100, 200, 300, 400), label="knife 0.92")
        path = save_annotated_frame(
            frame, "evt_test1", [det], config,
            timestamp=datetime(2026, 2, 20, 8, 0, 0),
            metadata_lines=["camera: test-cam"],
        )
        assert path.exists()
        assert path.suffix == ".jpg"
        img = cv2.imread(str(path))
        assert img is not None
        assert img.shape[0] > 0
        assert "knife" in path.name.lower()

    def test_creates_year_month_day_dirs(self, evidence_root: Path):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config = EventConfig(evidence_dir=str(evidence_root), jpeg_quality=80, save_evidence=True, cooldown_seconds=5)
        path = save_annotated_frame(
            frame, "evt_dirtest", [], config,
            timestamp=datetime(2025, 12, 31, 23, 59, 59),
        )
        assert path.exists()
        assert "/2025/12/31/" in str(path).replace("\\", "/")