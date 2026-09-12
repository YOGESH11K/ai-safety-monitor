"""REST endpoints: status, events, evidence images, config, plus MJPEG live feed."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.config import AppConfig
from app.database.database import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


def create_routes(config: AppConfig, database: Database, state_holder, frame_holder) -> APIRouter:
    """Build the API router bound to the shared application state.

    ``state_holder``: object with ``current_state`` (or dict) describing the live
    monitoring state.
    ``frame_holder``: callable returning the latest annotated frame (numpy) or None.
    """

    @router.get("/status")
    def get_status():
        state = state_holder.get() if hasattr(state_holder, "get") else state_holder
        return {
            "running": True,
            "state": state.get("state", "unknown"),
            "risk_level": state.get("risk_level", "none"),
            "detections": state.get("detections", []),
            "violence": state.get("violence_present", False),
            "camera_id": config.camera_id,
            "source": config.camera.source,
            "model": state.get("model_name", "stub"),
            "stub_mode": state.get("stub_mode", True),
            "frame": state.get("frame", 0),
            "since": state.get("since", time.time()),
            "events_total": database.count_events(),
        }

    @router.get("/events")
    def list_events(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        status: Optional[str] = Query(None),
    ):
        return {"events": database.list_events(limit=limit, offset=offset, status=status)}

    @router.get("/events/{event_id}")
    def get_event(event_id: str):
        event = database.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        return event

    @router.post("/events/{event_id}/review")
    def review_event(event_id: str, status: str = Query(..., pattern="^(pending|reviewed|ignored)$")):
        updated = database.set_status(event_id, status)
        if not updated:
            raise HTTPException(status_code=404, detail="Event not found")
        return {"id": event_id, "status": status}

    @router.get("/events/{event_id}/image")
    def event_image(event_id: str):
        event = database.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        image_path = event.get("image_path") or ""
        if not image_path or not Path(image_path).is_file():
            raise HTTPException(status_code=404, detail="Evidence image missing")
        return FileResponse(Path(image_path), media_type="image/jpeg")

    @router.get("/config")
    def get_config():
        return config.to_dict()

    @router.get("/feed")
    def live_feed():
        """MJPEG multipart stream of the latest annotated frame."""
        if frame_holder() is None:
            raise HTTPException(status_code=503, detail="Camera feed not ready")

        def generator():
            while True:
                # frame_holder() already guards the shared frame with its own lock.
                frame = frame_holder()
                if frame is None:
                    time.sleep(0.1)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    time.sleep(0.05)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
                time.sleep(0.05)

        return StreamingResponse(
            generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return router