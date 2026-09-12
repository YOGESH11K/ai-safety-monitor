"""Weapon detection via Ultralytics YOLO, with a deterministic stub fallback.

The module never pretends a generic COCO model reliably detects weapons. It loads
a user-supplied fine-tuned model from ``MODEL_PATH`` (e.g. models/weapon_model.pt).
If no model is present and ``detector_mode`` is "auto", a clearly-labelled stub is
used so the pipeline can run without a model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import DetectionConfig
from app.detection.base import BaseDetector, Detection, FrameResult

DEFAULT_WEAPON_CLASSES = ("gun", "knife", "pistol", "rifle", "weapon")


class WeaponDetector(BaseDetector):
    name = "weapon-yolov8"

    def __init__(self, config: DetectionConfig, model_path: Optional[Path | str] = None) -> None:
        self.config = config
        self.model_path = Path(str(model_path)) if model_path else None
        self._model = None
        self._names: dict[int, str] = {}
        self._load_yolo()

    def _load_yolo(self) -> None:
        try:
            from ultralytics import YOLO  # heavy import kept lazy
        except ImportError as exc:  # pragma: no cover - exercised only without dependency
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        path = self.model_path or self.config.model_path
        p = Path(str(path))
        if not p.is_absolute():
            # resolve relative to project root
            from app.config import PROJECT_ROOT

            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(
                f"Weapon model not found at {p}. Place a fine-tuned YOLO .pt file"
                " there (see README 'Model placement' / 'Training')."
            )
        self._model = YOLO(str(p))
        self._names = {i: n.lower() for i, n in self._model.names.items()}

    @property
    def weapon_classes(self) -> tuple[str, ...]:
        classes = self.config.weapon_classes
        return tuple(classes) if classes else DEFAULT_WEAPON_CLASSES

    @property
    def model(self):
        return self._model

    def detect(self, frame: np.ndarray) -> FrameResult:
        if self._model is None:
            raise RuntimeError("WeaponDetector used without a loaded model.")
        if frame is None:
            return FrameResult(frame_id=0)

        results = self._model.predict(
            source=frame,
            conf=self.config.model_conf,
            iou=self.config.iou,
            verbose=False,
            device="cpu",
        )
        detections: list[Detection] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    name = self._names.get(cls_id, str(cls_id)).lower()
                    if name not in self.weapon_classes:
                        continue
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    detections.append(
                        Detection(
                            class_name=name,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            label=f"{name} {conf:.2f}",
                        )
                    )
        return FrameResult(frame_id=0, detections=detections)


class StubWeaponDetector(BaseDetector):
    """Deterministic detector used when no trained model is available.

    Uses simple OpenCV heuristics so the pipeline and dashboard can be exercised
    end-to-end. DO NOT use in production: results are not grounded in a real model.
    """

    name = "weapon-stub"

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    @property
    def running_in_stub_mode(self) -> bool:
        return True

    def detect(self, frame: np.ndarray) -> FrameResult:
        if frame is None:
            return FrameResult(frame_id=0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # The synthetic feed draws a bright orange circle; the stub flags it.
        mask = cv2.inRange(frame, (0, 100, 200), (80, 200, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        weapon_classes = self.weapon_classes
        class_name = weapon_classes[0] if weapon_classes else "gun"
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            conf = min(0.98, 0.78 + area / 25_000.0)
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=round(conf, 3),
                    bbox=(x, y, x + w, y + h),
                    label=f"{class_name} stub {conf:.2f}",
                )
            )
        return FrameResult(frame_id=0, detections=detections)

    @property
    def weapon_classes(self) -> tuple[str, ...]:
        classes = self.config.weapon_classes
        return tuple(classes) if classes else DEFAULT_WEAPON_CLASSES


def build_weapon_detector(config: DetectionConfig) -> BaseDetector:
    """Create the best available detector for the config.

    ``detector_mode``:
        * "auto"  -> real YOLO if model file exists, else stub
        * "yolo"  -> real YOLO (raises if model missing)
        * "stub"  -> always stub
    """
    mode = (config.detector_mode or "auto").strip().lower()
    if mode == "stub":
        return StubWeaponDetector(config)
    if mode not in ("auto", "yolo"):
        raise ValueError(f"Unknown detector_mode: {mode!r}. Use 'auto', 'yolo', or 'stub'.")

    path = Path(config.model_path)
    if not path.is_absolute():
        from app.config import PROJECT_ROOT

        candidates = [path, PROJECT_ROOT / path]
    else:
        candidates = [path]

    exists = any(p.exists() for p in candidates)
    if mode == "yolo" and not exists:
        raise FileNotFoundError(
            f"Detector mode is 'yolo' but no model found at {config.model_path}."
            " Provide a trained .pt (see README) or use 'auto'/'stub'."
        )
    if not exists:
        return StubWeaponDetector(config)

    try:
        from ultralytics import YOLO  # noqa: F401  (fail fast if not installed)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "A model file exists but ultralytics is not installed."
            " Run: pip install ultralytics"
        ) from exc
    return WeaponDetector(config)