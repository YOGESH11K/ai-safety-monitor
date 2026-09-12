"""WebSocket hub broadcasting real-time detections, state changes, and events."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WSHub:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._snapshot: dict[str, Any] = {}

    async def register(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        logger.info("WebSocket client connected (%d total)", len(self._clients))
        if self._snapshot:
            try:
                await websocket.send_json(self._snapshot)
            except Exception:  # noqa: BLE001
                pass

    async def unregister(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    def set_snapshot(self, data: dict[str, Any]) -> None:
        self._snapshot = data

    def broadcast(self, message: dict) -> None:
        """Synchronous, fire-and-forget: dispatch to the event loop."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send_all(message), loop)
        # If no running loop, there are no connected clients to update.

    async def _send_all(self, message: dict) -> None:
        stale = []
        text = json.dumps(message, default=str)
        for ws in self._clients:
            try:
                await ws.send_text(text)
            except Exception:  # noqa: BLE001
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)
        if stale:
            logger.info("Dropped %d stale WebSocket clients", len(stale))