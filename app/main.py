"""Application factory: wires config, camera, detectors, events, alerts, API, frontend.

Runs the capture/detection loop on a background thread driven by FastAPI lifespan.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.alerts.alert_manager import AlertManager, build_alert_manager
from app.api.routes import create_routes
from app.api.ws import WSHub
from app.camera.camera_source import CameraError, build_camera_source
from app.config import AppConfig
from app.database.database import Database
from app.detection.base import BaseDetector, FrameResult
from app.detection.violence_detector import RiskAssessment, ViolenceDetector, build_violence_detector
from app.detection.weapon_detector import build_weapon_detector
from app.events.event_manager import EventManager, ConfirmedEvent

logger = logging.getLogger(__name__)

NORMAL = "NORMAL"
WATCHING = "WATCHING"
DETECTION = "DETECTION"
CONFIRMED_EVENT = "CONFIRMED_EVENT"
VALID_STATES = {NORMAL, WATCHING, DETECTION, CONFIRMED_EVENT}


@dataclass
class MonitorState:
    state: str = NORMAL
    risk_level: str = "none"
    detections: list[dict] = field(default_factory=list)
    violence_present: bool = False
    model_name: str = "n/a"
    stub_mode: bool = True
    frame: int = 0
    since: float = field(default_factory=time.time)
    last_event_time: float = 0.0
    last_event_at: object = None  # ConfirmedEvent | None

    def get(self) -> dict:
        return {
            "state": self.state,
            "risk_level": self.risk_level,
            "detections": self.detections,
            "violence_present": self.violence_present,
            "model_name": self.model_name,
            "stub_mode": self.stub_mode,
            "frame": self.frame,
            "since": self.since,
        }


class MonitorService:
    """Owns the camera, detectors, event pipeline, alerting and shared state."""

    def __init__(
        self,
        config: AppConfig,
        db: Database,
        hub: WSHub,
        alerts: AlertManager,
    ) -> None:
        self.config = config
        self.db = db
        self.hub = hub
        self.alerts = alerts
        self.state = MonitorState()
        self.weapon_detector: BaseDetector = build_weapon_detector(config.detection)
        self.violence_detector: ViolenceDetector = build_violence_detector(config.violence)
        self.risk_assessor = RiskAssessment.combine
        self.camera = build_camera_source(config.camera)

        self._stop_event = threading.Event()
        self._latest_frame: Optional[Any] = None
        self._feed_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        store = self._persist_event
        self.event_manager = EventManager(
            event_config=config.events,
            confirmation_config=config.confirmation,
            confidence_threshold=config.detection.confidence_threshold,
            camera_id=config.camera_id,
            store=store,
        )

        self.state.model_name = self.weapon_detector.name
        self.state.stub_mode = self.weapon_detector.running_in_stub_mode
        if self.state.stub_mode:
            logger.warning(
                "Running with the STUB detector (no trained model at %s)."
                " Results are demo-only; see README for training.",
                config.model_path,
            )

    def _persist_event(self, event: ConfirmedEvent) -> None:
        img_rel = ""
        if event.image_path:
            try:
                img_rel = str(Path(event.image_path))
            except Exception:  # noqa: BLE001
                img_rel = event.image_path
        self.db.insert_event(
            event_id=event.event_id,
            event_type=event.event_type,
            detected_class=event.detected_class,
            confidence=event.confidence,
            timestamp=event.timestamp,
            camera_id=event.camera_id,
            image_path=img_rel,
            meta=str(event.meta) if event.meta else "",
        )

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.camera.open()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="monitor-loop", daemon=True)
        self._thread.start()
        logger.info("Monitor loop started (source=%s)", self.config.camera.source)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.camera.close()
        logger.info("Monitor loop stopped")

    # -- main loop -------------------------------------------------------------

    def _run(self) -> None:
        fps = self.camera.fps or 20.0
        frame_interval = 1.0 / max(fps, 1.0) if fps else 0.05
        next_frame_time = time.monotonic()

        while not self._stop_event.is_set():
            elapsed = next_frame_time - time.monotonic()
            if elapsed > 0:
                time.sleep(elapsed)
            next_frame_time = time.monotonic() + frame_interval

            frame = self.camera.read()
            if frame is None:
                if getattr(self.camera, "reached_end", False):
                    logger.info("Video source ended; waiting and retrying...")
                    time.sleep(0.5)
                    try:
                        self.camera.open()
                    except CameraError as exc:  # noqa: BLE001
                        logger.error("Reconnect failed: %s", exc)
                        time.sleep(2.0)
                    continue
                time.sleep(0.02)
                continue

            self._process_frame(frame)

    def _process_frame(self, frame) -> None:
        result = self.weapon_detector.detect(frame)
        frame_id = self.state.frame + 1
        self.state.frame = frame_id

        violence = self.violence_detector.push(frame) if self.config.violence.enabled else None
        violent = bool(violence and violence.is_positive)

        confirmed = self.event_manager.update(result, frame=frame)

        annotated = self._annotate(frame, result, violent, confirmed)

        with self._feed_lock:
            self._latest_frame = annotated

        risk = self.risk_assessor(
            weapon_present=result.is_positive,
            violence_present=violent,
            enable_high_risk=self.config.violence.trigger_high_risk,
        )

        if confirmed:
            if confirmed.event_type == "weapon" and risk.risk_level == "high_risk":
                # promote stored event type
                self._promote_high_risk(confirmed)
            self.state.last_event_time = time.monotonic()
            self.state.last_event_at = confirmed
            self.state.state = CONFIRMED_EVENT
            self.state.risk_level = "high_risk" if risk.risk_level == "high_risk" else "high"
            self.state.detections = self._detections_to_dict(result.detections)
            self.state.violence_present = violent
            self.alerts.on_event(confirmed)
            self.hub.broadcast({"type": "detections", "data": {"detections": self.state.detections}})
            self.hub.set_snapshot(self._snapshot_payload())
            return

        self._update_state(result, violent, risk)
        self.hub.set_snapshot(self._snapshot_payload())

        # throttled live detection updates
        if frame_id % 3 == 0:
            self.hub.broadcast(
                {
                    "type": "detections",
                    "data": {"detections": self.state.detections, "state": self.state.state},
                }
            )

    def _promote_high_risk(self, event: ConfirmedEvent) -> None:
        # Weapon + violence concurrent: promote the persisted event to high_risk
        # (keeps auto-assigned event id + evidence image unchanged).
        self.db.set_event_type(
            event.event_id,
            "high_risk",
            extra_meta=" high_risk",
        )
        event.event_type = "high_risk"
        self.hub.broadcast(
            {
                "type": "event",
                "data": event.to_dict(),
                "risk_level": "high_risk",
            }
        )

    def _update_state(self, result: FrameResult, violent: bool, risk: RiskAssessment) -> None:
        dets = self._detections_to_dict(result.detections)
        self.state.detections = dets
        self.state.violence_present = violent
        self.state.risk_level = risk.risk_level

        if (time.monotonic() - self.state.last_event_time) < 3.0:
            self.state.state = CONFIRMED_EVENT
        elif self.event_manager.tracker.positives_in_window > 0:
            self.state.state = DETECTION
        elif dets:
            self.state.state = WATCHING
        else:
            self.state.state = NORMAL

    @staticmethod
    def _detections_to_dict(detections) -> list[dict]:
        return [
            {
                "class": d.class_name,
                "confidence": round(float(d.confidence), 4),
                "bbox": list(d.bbox),
            }
            for d in detections
        ]

    def _annotate(self, frame, result: FrameResult, violent: bool, confirmed: Optional[ConfirmedEvent]) -> Any:
        out = frame
        for d in result.detections:
            x1, y1, x2, y2 = d.bbox
            color = (0, 215, 255) if d.confidence >= self.config.detection.confidence_threshold else (0, 215, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d.label or d.class_name}"
            cv2.putText(out, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if violent:
            cv2.putText(out, "movement analysis: possible disruptive behaviour", (8, out.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 255), 1, cv2.LINE_AA)
        status_line = f"{self.state.state} | frame {self.state.frame} | source {self.config.camera.source}"
        if confirmed:
            status_line += f" | event {confirmed.event_id}"
        cv2.putText(out, status_line, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        return out

    def _snapshot_payload(self) -> dict:
        return {
            "type": "snapshot",
            "data": {
                "state": self.state.state,
                "risk_level": self.state.risk_level,
                "detections": self.state.detections,
                "violence_present": self.state.violence_present,
                "model_name": self.state.model_name,
                "stub_mode": self.state.stub_mode,
                "frame": self.state.frame,
                "camera_id": self.config.camera_id,
            },
        }

    # -- feed access -----------------------------------------------------------

    def latest_frame(self):
        with self._feed_lock:
            return self._latest_frame

    @property
    def feed_lock(self) -> threading.Lock:
        return self._feed_lock


def create_app(config: AppConfig, db_path: str = "data/safety.db") -> FastAPI:
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    db = Database(db_path)
    hub = WSHub()
    alerts = build_alert_manager(config.alerts, broadcaster=hub)
    monitor = MonitorService(config, db, hub, alerts)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        monitor.start()
        try:
            yield
        finally:
            monitor.stop()
            db.close()

    app = FastAPI(title="AI Safety Monitor", version="1.0.0", lifespan=lifespan)

    cors_origins = [
        origin.strip()
        for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/ws-health")
    async def ws_health():
        return {"ok": True}

    @app.websocket("/api/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await hub.register(websocket)
        try:
            while True:
                await websocket.receive_text()  # keep-alive / ignore client pings
        except WebSocketDisconnect:
            pass
        finally:
            await hub.unregister(websocket)

    app.include_router(
        create_routes(config, db, monitor.state, monitor.latest_frame)
    )

    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    @app.exception_handler(CameraError)
    async def camera_error_handler(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    # expose for tests/embedding
    app.state.monitor = monitor
    app.state.config = config
    app.state.db = db
    return app