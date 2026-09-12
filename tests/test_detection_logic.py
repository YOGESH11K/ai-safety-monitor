"""Tests for confidence threshold filtering and temporal confirmation logic."""
from __future__ import annotations

import numpy as np

import pytest

from app.config import ConfirmationConfig, DetectionConfig
from app.detection.base import Detection, FrameResult
from app.detection.violence_detector import RiskAssessment, StubViolenceDetector, ViolenceResult
from app.detection.weapon_detector import StubWeaponDetector
from app.events.event_manager import TemporalTracker


# -- Confidence threshold / detection results ---------------------------------

def _make_detection(confidence: float, class_name: str = "gun") -> Detection:
    return Detection(class_name=class_name, confidence=confidence, bbox=(10, 10, 50, 50), label=f"{class_name} {confidence:.2f}")

def _make_frame(*confs: float) -> FrameResult:
    dets = [_make_detection(c) for c in confs] if confs else []
    return FrameResult(frame_id=0, detections=dets)


class TestConfidenceThreshold:
    def test_empty_frame_not_positive(self):
        r = FrameResult(frame_id=0)
        tracker = TemporalTracker(window=3, min_positive=2, confidence_threshold=0.80)
        assert not tracker.is_positive_frame(r)

    def test_high_conf_positive(self):
        tracker = TemporalTracker(window=3, min_positive=2, confidence_threshold=0.80)
        assert tracker.is_positive_frame(_make_frame(0.85))
        assert tracker.is_positive_frame(_make_frame(0.99))

    def test_low_conf_not_positive(self):
        tracker = TemporalTracker(window=3, min_positive=2, confidence_threshold=0.80)
        assert not tracker.is_positive_frame(_make_frame(0.79))
        assert not tracker.is_positive_frame(_make_frame(0.3))
        assert not tracker.is_positive_frame(_make_frame(0.0))

    def test_mixed_conf_positive_if_any_above_threshold(self):
        tracker = TemporalTracker(window=3, min_positive=2, confidence_threshold=0.80)
        assert tracker.is_positive_frame(_make_frame(0.70, 0.82))
        assert not tracker.is_positive_frame(_make_frame(0.70, 0.79))


# -- Temporal confirmation ---------------------------------------------------

class TestTemporalTracker:
    def test_init_requires_positive_window(self):
        with pytest.raises(ValueError, match="positive"):
            TemporalTracker(window=0, min_positive=1, confidence_threshold=0.80)

    def test_init_requires_min_positive_leq_window(self):
        with pytest.raises(ValueError, match="min_positive cannot exceed"):
            TemporalTracker(window=3, min_positive=5, confidence_threshold=0.80)

    def test_init_requires_threshold_in_range(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            TemporalTracker(window=3, min_positive=2, confidence_threshold=1.5)

    def test_not_confirmed_before_window_filled(self):
        t = TemporalTracker(window=5, min_positive=3, confidence_threshold=0.80)
        assert not t.update(_make_frame(0.85))
        assert not t.update(_make_frame(0.85))
        assert not t.update(_make_frame(0.85))

    def test_confirm_when_window_satisfied(self):
        t = TemporalTracker(window=5, min_positive=3, confidence_threshold=0.80)
        for _ in range(3):
            assert not t.update(_make_frame(0.85))
        assert t.update(_make_frame(0.85))
        assert t.update(_make_frame(0.85))

    def test_not_confirmed_when_only_one_negative_in_window(self):
        """3 positives out of 5 frames, min_positive=3 should confirm."""
        t = TemporalTracker(window=5, min_positive=3, confidence_threshold=0.80)
        assert not t.update(_make_frame(0.85))
        assert not t.update(_make_frame(0.85))
        assert not t.update(_make_frame())  # negative
        assert not t.update(_make_frame(0.85))
        assert t.update(_make_frame(0.85))  # positives=3 → confirms

    def test_not_confirmed_when_too_few_positives(self):
        t = TemporalTracker(window=5, min_positive=3, confidence_threshold=0.80)
        assert not t.update(_make_frame(0.85))
        assert not t.update(_make_frame())  # neg
        assert not t.update(_make_frame())  # neg
        assert not t.update(_make_frame(0.85))
        assert not t.update(_make_frame())  # positives=2 < 3 → no confirm

    def test_sliding_window_drops_old_positive(self):
        """After dropping below min_positive due to window sliding."""
        t = TemporalTracker(window=3, min_positive=2, confidence_threshold=0.80)
        assert not t.update(_make_frame(0.85))
        assert not t.update(_make_frame(0.85))
        # Window full [P, P] → but we need positives_in_window >= 2 AND best > 0.0
        # _recent_pseudoconfidence check: there IS a detection with best > 0.0
        assert t.update(_make_frame())  # now window [P, P, N] → positives=2, confirm
        assert not t.update(_make_frame())  # [P, N, N] → positives=1, no confirm

    def test_reset_clears_history(self):
        t = TemporalTracker(window=3, min_positive=2, confidence_threshold=0.80)
        t.update(_make_frame(0.85))
        t.update(_make_frame(0.85))
        t.reset()
        assert not t.update(_make_frame())  # starts fresh, not full yet
        assert not t.update(_make_frame(0.85))
        assert t.update(_make_frame(0.85))  # now filled [N, P, P] but positives=2 AND best>0 → confirm

    def test_positives_in_window_property(self):
        t = TemporalTracker(window=3, min_positive=2, confidence_threshold=0.80)
        assert t.positives_in_window == 0
        t.update(_make_frame(0.85))
        assert t.positives_in_window == 1
        t.update(_make_frame(0.85))
        assert t.positives_in_window == 2


# -- FrameResult helpers ------------------------------------------------------

class TestFrameResult:
    def test_is_positive_when_dets(self):
        r = _make_frame(0.9)
        assert r.is_positive

    def test_not_is_positive_empty(self):
        r = _make_frame()
        assert not r.is_positive

    def test_best_returns_highest_conf(self):
        r = _make_frame(0.6, 0.95, 0.80)
        assert r.best().confidence == 0.95

    def test_best_none_when_empty(self):
        r = _make_frame()
        assert r.best() is None


# -- RiskAssessment -----------------------------------------------------------

class TestRiskAssessment:
    def test_weapon_and_violence_high_risk(self):
        r = RiskAssessment.combine(weapon_present=True, violence_present=True, enable_high_risk=True)
        assert r.risk_level == "high_risk"

    def test_weapon_only_high(self):
        r = RiskAssessment.combine(weapon_present=True, violence_present=False, enable_high_risk=True)
        assert r.risk_level == "high"

    def test_violence_only_medium(self):
        r = RiskAssessment.combine(weapon_present=False, violence_present=True, enable_high_risk=True)
        assert r.risk_level == "medium"

    def test_none(self):
        r = RiskAssessment.combine(False, False, True)
        assert r.risk_level == "none"

    def test_weapon_and_violence_but_high_risk_disabled(self):
        r = RiskAssessment.combine(True, True, enable_high_risk=False)
        assert r.risk_level == "high"


# -- ViolenceDetector stub ---------------------------------------------------

class TestStubViolenceDetector:
    def test_default_not_violent(self):
        d = StubViolenceDetector()
        r = d.push(np.zeros((480, 640, 3), dtype=np.uint8))
        assert not r.violent

    def test_stub_labelled_as_stub(self):
        d = StubViolenceDetector()
        assert d.running_in_stub_mode

    def test_always_label_violent(self):
        d = StubViolenceDetector(always_label_violent=True)
        for _ in range(20):
            r = d.push(np.zeros((480, 640, 3), dtype=np.uint8))
        assert r.violent


# -- StubWeaponDetector -------------------------------------------------------

class TestStubWeaponDetector:
    def test_stub_mode_flag(self, detection_config_stub: DetectionConfig):
        det = StubWeaponDetector(detection_config_stub)
        assert det.running_in_stub_mode
        assert det.name == "weapon-stub"

    def test_detect_returns_empty_on_dark_frame(self, detection_config_stub: DetectionConfig):
        det = StubWeaponDetector(detection_config_stub)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r = det.detect(frame)
        assert r.detections == []

    def test_detect_finds_bright_object(self, detection_config_stub: DetectionConfig):
        det = StubWeaponDetector(detection_config_stub)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2 = __import__("cv2")
        cv2.circle(frame, (320, 240), 30, (0, 165, 255), -1)  # orange ball
        r = det.detect(frame)
        assert len(r.detections) >= 1
        assert r.detections[0].class_name in detection_config_stub.weapon_classes