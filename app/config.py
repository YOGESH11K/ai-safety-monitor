"""Configuration loading with layered precedence: config.yaml defaults, then .env/env vars."""
from __future__ import annotations

import copy
import os
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_BOOL_TRUE = {"1", "true", "yes", "on", "y"}


def _env_typed(name: str, default: Any) -> Any:
    """Read an env var and coerce it to the type of ``default``."""
    value = os.environ.get(name)
    if value is None:
        return default
    if isinstance(default, bool):
        return value.strip().lower() in _BOOL_TRUE
    if isinstance(default, int):
        try:
            return int(value)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except ValueError:
            return default
    if isinstance(default, (list, tuple)):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items if isinstance(default, list) else tuple(items)
    return value


def _env_overlay(prefix: str, template: Any) -> dict:
    """Collect env vars named PREFIX<FIELD> into {<field>: value} with type coercion."""
    overlay: dict = {}
    hints = typing.get_type_hints(template)
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):].lower()
        if name not in hints:
            continue
        overlay[name] = _coerce(value, hints[name])
    return overlay


def _coerce(value: str, expected_type: Any) -> Any:
    expected = expected_type
    if isinstance(expected, str):
        # with `from __future__ import annotations`, dataclass field types are
        # strings until resolved by typing.get_type_hints(); keep a fallback.
        expected = _STRINGIFIED_TYPES.get(expected, str)
    if expected is bool:
        return value.strip().lower() in _BOOL_TRUE
    if expected is int:
        try:
            return int(value)
        except ValueError:
            return value
    if expected is float:
        try:
            return float(value)
        except ValueError:
            return value
    if expected in (list,):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


_STRINGIFIED_TYPES = {
    "bool": bool,
    "int": int,
    "float": float,
    "list": list,
    "str": str,
}


def _build(cls: Any, data: dict, env_prefix: str) -> Any:
    """Build dataclass from YAML dict + env overlay (env wins over yaml)."""
    hints = typing.get_type_hints(cls)
    kwargs: dict = {}
    for f in fields(cls):
        if f.name in data:
            val = data[f.name]
            kwargs[f.name] = _coerce(val, hints[f.name]) if isinstance(val, str) else val
    inst = cls(**kwargs)  # untouched fields keep their dataclass defaults
    for key, value in _env_overlay(env_prefix, inst).items():
        setattr(inst, key, value)
    return inst


@dataclass
class CameraConfig:
    source: str = "0"
    width: int = 640
    height: int = 480
    fps: float = 20.0
    rtsp_buffer_size: int = 1


@dataclass
class DetectionConfig:
    detector_mode: str = "auto"  # "auto" -> real model if present, else stub; or "stub"/"yolo"
    model_path: str = "models/weapon_model.pt"
    model_conf: float = 0.25  # raw YOLO NMS confidence (pre-filter)
    confidence_threshold: float = 0.80  # positive-detection threshold for confirmation logic
    weapon_classes: list = field(default_factory=list)
    iou: float = 0.45
    max_det: int = 300


@dataclass
class ConfirmationConfig:
    window: int = 5
    min_positive: int = 3
    min_mean_confidence: float = 0.0
    confidence_threshold: float = 0.0  # reserved; detection.confidence_threshold is the operative gate


@dataclass
class EventConfig:
    cooldown_seconds: float = 10.0
    evidence_dir: str = "events"
    save_evidence: bool = True
    jpeg_quality: int = 92


@dataclass
class ViolenceConfig:
    enabled: bool = False
    window: int = 16
    min_positive: int = 10
    threshold: float = 0.6
    trigger_high_risk: bool = True


@dataclass
class AlertConfig:
    enabled: bool = True
    log: bool = True
    dashboard: bool = True
    webhook_url: str = ""
    email_to: str = ""


@dataclass
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    require_auth_token: bool = False
    auth_token: str = ""


@dataclass
class AppConfig:
    camera_id: str = "camera-1"
    log_level: str = "INFO"
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    confirmation: ConfirmationConfig = field(default_factory=ConfirmationConfig)
    events: EventConfig = field(default_factory=EventConfig)
    violence: ViolenceConfig = field(default_factory=ViolenceConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @classmethod
    def load(cls, config_path: Optional[Path | str] = None) -> "AppConfig":
        data = _load_yaml(config_path)
        cfg = cls._build_sections(data)
        cfg._apply_top_level_env()
        return cfg

    @classmethod
    def _build_sections(cls, data: dict) -> "AppConfig":
        camera = _build(CameraConfig, data.get("camera", {}), "CAMERA_")
        detection = _build(DetectionConfig, data.get("detection", {}), "DETECT_")
        confirmation = _build(ConfirmationConfig, data.get("confirmation", {}), "CONFIRM_")
        events = _build(EventConfig, data.get("events", {}), "EVENTS_")
        violence = _build(ViolenceConfig, data.get("violence", {}), "VIOLENCE_")
        alerts = _build(AlertConfig, data.get("alerts", {}), "ALERTS_")
        api = _build(APIConfig, data.get("api", {}), "API_")
        return cls(
            camera_id=data.get("camera_id", "camera-1"),
            log_level=data.get("log_level", "INFO"),
            camera=camera,
            detection=detection,
            confirmation=confirmation,
            events=events,
            violence=violence,
            alerts=alerts,
            api=api,
        )

    def _apply_top_level_env(self) -> None:
        self.camera_id = os.environ.get("CAMERA_ID", self.camera_id)
        self.log_level = os.environ.get("LOG_LEVEL", self.log_level)
        # Allow CAMERA_SOURCE to exist even though it's under camera section for UX parity.
        if "CAMERA_SOURCE" in os.environ:
            self.camera.source = os.environ["CAMERA_SOURCE"]

    @property
    def model_path(self) -> Path:
        p = Path(self.detection.model_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def evidence_dir(self) -> Path:
        p = Path(self.events.evidence_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict:
        d = {
            "camera_id": self.camera_id,
            "log_level": self.log_level,
            "camera": _defaults_dict(self.camera),
            "detection": _defaults_dict(self.detection),
            "confirmation": _defaults_dict(self.confirmation),
            "events": _defaults_dict(self.events),
            "violence": _defaults_dict(self.violence),
            "alerts": _defaults_dict(self.alerts),
        }
        api_cfg = _defaults_dict(self.api)
        if api_cfg.get("require_auth_token") and api_cfg.get("auth_token"):
            api_cfg["auth_token"] = "********"
        d["api"] = api_cfg
        return d


def _defaults_dict(obj: Any) -> dict:
    out = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, Path):
            value = str(value)
        out[f.name] = copy.deepcopy(value)
    return out


def _load_yaml(config_path: Optional[Path | str]) -> dict:
    path = Path(str(config_path)) if config_path else PROJECT_ROOT / "config.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
        return loaded if isinstance(loaded, dict) else {}