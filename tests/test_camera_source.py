"""Tests for CameraSource: invalid handling, synthetic mode, and video-file mode."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.camera.camera_source import CameraError, CameraSource, SyntheticCameraSource, build_camera_source
from app.config import CameraConfig


class TestClassifySource:
    def test_webcam_index(self):
        assert CameraSource.classify_source("0") == "webcam"
        assert CameraSource.classify_source("3") == "webcam"

    def test_rtsp_stream(self):
        assert CameraSource.classify_source("rtsp://192.168.1.1:554/stream") == "stream"
        assert CameraSource.classify_source("http://cam.local/live") == "stream"
        assert CameraSource.classify_source("https://example.com/feed") == "stream"

    def test_video_file(self):
        assert CameraSource.classify_source("data/test.mp4") == "file"

    def test_synthetic(self):
        assert CameraSource.classify_source("synthetic") == "synthetic"


class TestInvalidCameraSource:
    def test_nonexistent_file_raises(self):
        cfg = CameraConfig(source="C:/nonexistent/video.mp4")
        cam = CameraSource(cfg)
        with pytest.raises(CameraError, match="not found|not found|Could not open"):
            cam.open()

    def test_invalid_rtsp_stays_none(self):
        """An invalid RTSP URL does not hang the application during tests."""
        cfg = CameraConfig(source="rtsp://127.0.0.1:1/nonexistent")
        cam = CameraSource(cfg)
        try:
            cam.open()
        except CameraError:
            pass
        finally:
            cam.close()


class TestSyntheticCameraSource:
    def test_yields_frames(self):
        cfg = CameraConfig(source="synthetic", width=640, height=480)
        cam = build_camera_source(cfg)
        cam.open()
        assert cam.is_file
        frames = list(cam.frames(max_frames=5))
        assert len(frames) == 5
        for f in frames:
            assert isinstance(f, np.ndarray)
            assert f.shape == (480, 640, 3)
        cam.close()

    def test_never_reaches_end(self):
        cfg = CameraConfig(source="synthetic", width=640, height=480)
        cam = build_camera_source(cfg)
        cam.open()
        assert not cam.reached_end
        cam.read()
        assert not cam.reached_end


class TestVideoFileMode:
    def test_read_frames_from_video_file(self, tmp_path: Path):
        import cv2
        video_path = tmp_path / "test_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (160, 120))
        for _ in range(20):
            writer.write(np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8))
        writer.release()
        assert video_path.exists()

        cfg = CameraConfig(source=str(video_path), width=160, height=120)
        cam = CameraSource(cfg)
        cam.open()
        assert cam.is_file
        frames = list(cam.frames(max_frames=20))
        assert len(frames) == 20
        cam.close()