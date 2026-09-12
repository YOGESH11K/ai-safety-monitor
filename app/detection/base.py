"""Base types and interfaces for detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    label: Optional[str] = None

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class FrameResult:
    frame_id: int
    detections: list[Detection] = field(default_factory=list)
    processed_at: float = 0.0

    @property
    def is_positive(self) -> bool:
        return len(self.detections) > 0

    def best(self) -> Optional[Detection]:
        return max(self.detections, key=lambda d: d.confidence) if self.detections else None


class BaseDetector(ABC):
    """Common interface every detection model must implement."""

    name: str = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> FrameResult:
        """Run inference on a single BGR frame and return filtered detections."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional warmup step; defaults to no-op."""

    def close(self) -> None:
        """Release model resources if any."""

    @property
    def running_in_stub_mode(self) -> bool:
        return False