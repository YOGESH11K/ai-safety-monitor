"""Camera source abstraction: webcam index, video file, or RTSP/HTTP stream."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from app.config import CameraConfig


class CameraError(Exception):
    """Raised when a camera/source cannot be opened."""


class CameraSource:
    """Frame iterator over a configurable source.

    ``source`` may be:
      * an integer string like ``"0"`` -> local webcam index
      * a path to a video file (or ``http(s)://...`` stream URL, or ``rtsp://``)
      * ``"synthetic"`` -> programmatic test feed (no camera or file required)
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.source = str(config.source).strip()
        self._video: Optional[cv2.VideoCapture] = None
        self._frame_id = 0
        self._is_file = False
        self._reached_end = False

    @staticmethod
    def classify_source(source: str) -> str:
        s = source.strip()
        if s.lower() == "synthetic":
            return "synthetic"
        if re.match(r"^\d+$", s):
            return "webcam"
        if s.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return "stream"
        return "file"

    def open(self) -> None:
        kind = self.classify_source(self.source)
        if kind == "synthetic":
            self._is_file = True
            return
        if kind == "webcam":
            self._video = cv2.VideoCapture(int(self.source))
        else:
            path = Path(self.source)
            if kind == "file" and not path.is_file():
                raise CameraError(
                    f"Video file not found: {self.source!r}. Check CAMERA_SOURCE or"
                    " use an RTSP URL, a webcam index, or 'synthetic'."
                )
            self._video = cv2.VideoCapture(self.source)
            self._is_file = kind == "file"

        if self._video is None or not self._video.isOpened():
            self._cleanup()
            raise CameraError(
                f"Could not open camera source: {self.source!r}. Check that the"
                " device/URL is reachable."
            )

        self._apply_settings()

    def _apply_settings(self) -> None:
        assert self._video is not None
        if self.config.width > 0:
            self._video.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height > 0:
            self._video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self._is_file:
            self._video.set(cv2.CAP_PROP_BUFFERSIZE, self.config.rtsp_buffer_size)
        else:
            self._video.set(cv2.CAP_PROP_BUFFERSIZE, self.config.rtsp_buffer_size)

    @property
    def is_file(self) -> bool:
        return self._is_file

    @property
    def reached_end(self) -> bool:
        return self._reached_end

    @property
    def fps(self) -> float:
        if self._video is None:
            return float(self.config.fps)
        f = self._video.get(cv2.CAP_PROP_FPS)
        return float(f) if f and f > 0 else float(self.config.fps)

    @property
    def has_grabbed(self) -> bool:
        return self._frame_id > 0

    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame or None on end-of-stream/error."""
        if not self.has_open_handle or self._reached_end:
            return None
        assert self._video is not None
        ok, frame = self._video.read()
        if not ok or frame is None:
            self._reached_end = True
            return None
        self._frame_id += 1
        return frame

    @property
    def has_open_handle(self) -> bool:
        return self._video is not None and self._video.isOpened()

    def frames(self, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
        """Yield frames. For file/synthetic sources, loops repeatedly until max_frames."""
        yielded = 0
        while True:
            frame = self.read()
            if frame is None:
                if self._is_file and max_frames is None:
                    # loop the file: reopen from 0 for continuous monitoring
                    self._rewind()
                    continue
                break
            yield frame
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break

    def _rewind(self) -> None:
        if self._video is None:
            return
        if self._is_file:
            self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._reached_end = False

    def close(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self._video is not None:
            self._video.release()
            self._video = None


class SyntheticCameraSource(CameraSource):
    """Deterministic moving-shape feed for tests and model-free smoke runs.

    Emits frames that the stub detector can recognize (triggering positive
    detections at a known cadence) so the full pipeline can be exercised.
    """

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self._phase = 0

    def open(self) -> None:
        self._is_file = True
        self._reached_end = False

    def read(self) -> Optional[np.ndarray]:
        if self._reached_end:
            return None
        self._phase += 1
        w, h = self.config.width or 640, self.config.height or 480
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (24, 26, 32)
        # central object that moves sinusoidally (meant to look 'object-like')
        cx = int(w / 2 + (w / 4) * np.sin(self._phase / 12.0))
        cy = int(h / 2 + (h / 6) * np.cos(self._phase / 16.0))
        cv2.circle(frame, (cx, cy), 24, (0, 165, 255), -1)
        cv2.rectangle(frame, (cx - 8, cy - 8), (cx + 8, cy + 8), (255, 255, 255), 2)
        # walking dot (resembles motion, useful for violence-stub clarity)
        dx = int((self._phase * 6) % w)
        cv2.circle(frame, (dx, h - 40), 10, (200, 200, 200), -1)
        self._frame_id += 1
        return frame

    @property
    def fps(self) -> float:
        return float(self.config.fps)

    @property
    def is_file(self) -> bool:
        return True

    @property
    def reached_end(self) -> bool:
        return False


def build_camera_source(config: CameraConfig) -> CameraSource:
    if str(config.source).strip().lower() == "synthetic":
        return SyntheticCameraSource(config)
    return CameraSource(config)