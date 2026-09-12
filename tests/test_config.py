"""Tests for configuration loading, defaults, and env override."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from app.config import AppConfig, CameraConfig, DetectionConfig, ConfirmationConfig, EventConfig


class TestDefaultConfig:
    def test_load_without_file_returns_defaults(self):
        cfg = AppConfig.load(Path("/nonexistent/config.yaml"))
        assert cfg.camera_id == "camera-1"
        assert cfg.detection.confidence_threshold == 0.80
        assert cfg.confirmation.window == 5
        assert cfg.events.cooldown_seconds == 10.0
        assert cfg.camera.source == "0"

    def test_load_minimal_yaml(self, tmp_path: Path):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(yaml.dump({"camera": {"source": "2"}, "detection": {"confidence_threshold": 0.95}}))
        cfg = AppConfig.load(cfg_path)
        assert cfg.camera.source == "2"
        assert cfg.detection.confidence_threshold == 0.95
        # non-specified fields keep defaults
        assert cfg.confirmation.window == 5
        assert cfg.events.cooldown_seconds == 10.0


class TestEnvOverride:
    def test_camera_source_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(yaml.dump({"camera": {"source": "0"}}))
        monkeypatch.setenv("CAMERA_SOURCE", "rtsp://example.com/stream")
        cfg = AppConfig.load(cfg_path)
        assert cfg.camera.source == "rtsp://example.com/stream"

    def test_coherence_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CONFIRM_WINDOW", "8")
        monkeypatch.setenv("CONFIRM_MIN_POSITIVE", "2")
        monkeypatch.setenv("CONFIRM_CONFIDENCE_THRESHOLD", "0.91")
        cfg = AppConfig.load(Path("/nonexistent"))
        assert cfg.confirmation.window == 8
        assert cfg.confirmation.min_positive == 2
        assert cfg.confirmation.confidence_threshold == 0.91

    def test_detection_model_conf_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DETECT_MODEL_CONF", "0.6")
        monkeypatch.setenv("DETECT_IOU", "0.35")
        cfg = AppConfig.load(Path("/nonexistent"))
        assert cfg.detection.model_conf == 0.6
        assert cfg.detection.iou == 0.35


class TestConfigDataclassDefaults:
    def test_confidence_threshold_default(self):
        cfg = DetectionConfig()
        assert cfg.confidence_threshold == 0.80

    def test_confirmation_window_default(self):
        cfg = ConfirmationConfig()
        assert cfg.window == 5
        assert cfg.min_positive == 3

    def test_cooldown_default(self):
        cfg = EventConfig()
        assert cfg.cooldown_seconds == 10.0

    def test_camera_default(self):
        cfg = CameraConfig()
        assert cfg.source == "0"
        assert cfg.width == 640
        assert cfg.height == 480


class TestToDict:
    def test_serializable(self, config_minimal: AppConfig):
        d = config_minimal.to_dict()
        assert d["camera_id"] == "test-cam"
        assert d["camera"]["source"] == "synthetic"
        assert isinstance(d["confirmation"], dict)
        assert d["confirmation"]["window"] == 3

    def test_auth_token_redacted(self, monkeypatch):
        cfg = AppConfig()
        cfg.api.require_auth_token = True
        cfg.api.auth_token = "secret123"
        d = cfg.to_dict()
        assert d["api"]["auth_token"] == "********"


class TestProjectRoot:
    def test_project_root_is_parent_of_app(self):
        from app.config import PROJECT_ROOT
        assert (PROJECT_ROOT / "app" / "config.py").is_file()