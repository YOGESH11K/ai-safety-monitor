"""Violence/action-recognition abstraction.

A real violence detector is a *future* plug-in. This module defines the interface
any temporal action model must implement, ships a clearly-marked no-op stub, and
implements the combined risk logic (weapon AND violence -> high-risk).

IMPORTANT: No model can perfectly determine whether violence is occurring from
appearances alone. Results from this module must never be used to autonomously
accuse or act against a person. They are advisory only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.config import ViolenceConfig


@dataclass
class ViolenceResult:
    violent: bool
    confidence: float = 0.0
    label: str = "no_action"
    frames_analyzed: int = 0

    @property
    def is_positive(self) -> bool:
        return self.violent


class ViolenceDetector(ABC):
    """Interface for temporal action-recognition models (fighting, etc.).

    Implementations receive a sequence of frames and return one ViolenceResult.
    Swap the implementation in ``build_violence_detector`` without touching the
    rest of the application.
    """

    name: str = "violence-base"

    @abstractmethod
    def detect(self, frame_sequence: list[np.ndarray]) -> ViolenceResult:
        raise NotImplementedError

    def push(self, frame: np.ndarray) -> ViolenceResult:
        """Streaming convenience: implementations buffer internally and return a
        result each call; the base keeps the most recent ``window`` frames."""
        raise NotImplementedError


class StubViolenceDetector(ViolenceDetector):
    """Deterministic no-op used until a real action model is plugged in.

    Returns ``violent=False`` unless explicitly configured to label the demo feed
    as violent (useful for exercising combined-risk logic in tests).
    """

    name = "violence-stub"
    window = 16

    def __init__(self, config: Optional[ViolenceConfig] = None, always_label_violent: bool = False) -> None:
        self.config = config or ViolenceConfig()
        self.always_label_violent = always_label_violent
        self._buffer: list[np.ndarray] = []
        self._last_result = ViolenceResult(violent=False)

    def detect(self, frame_sequence: list[np.ndarray]) -> ViolenceResult:
        n = len(frame_sequence)
        if not frame_sequence and self.config.enabled:
            return self._last_result
        if self.always_label_violent and n >= self.config.min_positive:
            self._last_result = ViolenceResult(
                violent=True, confidence=self.config.threshold, label="likely_agitation", frames_analyzed=n
            )
            return self._last_result
        self._last_result = ViolenceResult(violent=False, frames_analyzed=n)
        return self._last_result

    def push(self, frame: np.ndarray) -> ViolenceResult:
        self._buffer.append(frame)
        if len(self._buffer) > self.config.window:
            self._buffer.pop(0)
        return self.detect(self._buffer)

    @property
    def running_in_stub_mode(self) -> bool:
        return True


def build_violence_detector(config: ViolenceConfig) -> ViolenceDetector:
    """Factory for the configured violence detector.

    Replace the returned instance with a real model-backed implementation when it
    becomes available (e.g. a 3D-CNN / transformer action recognizer consuming
    ``frame_sequence``).
    """
    return StubViolenceDetector(config)


@dataclass(frozen=True)
class RiskAssessment:
    """Aggregated risk state used to decide event creation."""

    weapon_present: bool
    violence_present: bool
    risk_level: str = "none"  # none | low | medium | high | high_risk
    message: str = ""

    @classmethod
    def combine(cls, weapon_present: bool, violence_present: bool, enable_high_risk: bool) -> "RiskAssessment":
        if weapon_present and violence_present and enable_high_risk:
            return cls(
                weapon_present=True,
                violence_present=True,
                risk_level="high_risk",
                message="Weapon and possible violent behaviour detected concurrently.",
            )
        if weapon_present:
            return cls(
                weapon_present=True,
                violence_present=violence_present,
                risk_level="high",
                message="Possible weapon detected.",
            )
        if violence_present:
            return cls(
                weapon_present=False,
                violence_present=True,
                risk_level="medium",
                message="Possible disruptive behaviour detected.",
            )
        return cls(
            weapon_present=False,
            violence_present=False,
            risk_level="none",
            message="No concerning detections.",
        )