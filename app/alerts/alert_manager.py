"""Alert abstraction. Notifications are advisory/logging only.

IMPORTANT: This system never triggers physical action against a person. Alerts
are limited to logging and dashboard notifications; email/webhook/other transport
mechanisms are intentionally deferred behind the AlertManager interface.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Protocol

from app.events.event_manager import ConfirmedEvent

logger = logging.getLogger(__name__)


class EventBroadcaster(Protocol):
    """Anything that can broadcast a message to connected dashboard clients."""

    def broadcast(self, message: dict) -> None: ...


class AlertManager(ABC):
    """Base class for notification backends.

    Add new mechanisms by subclassing and registering in ``build_alert_manager``
    (e.g. EmailAlertManager, WebhookAlertManager) without touching the pipeline.
    """

    name: str = "base"

    def on_event(self, event: ConfirmedEvent) -> None:
        """Called whenever a confirmed event is created."""
        self.alert(event)

    @abstractmethod
    def alert(self, event: ConfirmedEvent) -> None: ...

    def on_state_change(self, state: str, context: dict) -> None:
        """Optional: notified when the global monitoring state changes."""

    def close(self) -> None: ...


class ConsoleAlertManager(AlertManager):
    """Logs events and broadcasts to the dashboard over WebSocket."""

    name = "console-dashboard"

    def __init__(self, log: bool = True, broadcaster: EventBroadcaster | None = None) -> None:
        self._log = log
        self._broadcaster = broadcaster

    def alert(self, event: ConfirmedEvent) -> None:
        if self._log:
            logger.warning(
                "CONFIRMED EVENT %s type=%s class=%s conf=%.3f camera=%s time=%s image=%s",
                event.event_id,
                event.event_type,
                event.detected_class,
                event.confidence,
                event.camera_id,
                event.timestamp,
                event.image_path or "(none)",
            )
        if self._broadcaster is not None:
            try:
                self._broadcaster.broadcast({"type": "event", "data": event.to_dict()})
            except Exception:  # noqa: BLE001 - dashboard down must not break pipeline
                logger.exception("Alert broadcast failed")

    def on_state_change(self, state: str, context: dict) -> None:
        if self._broadcaster is not None:
            try:
                self._broadcaster.broadcast({"type": "state", "data": {"state": state, **context}})
            except Exception:  # noqa: BLE001
                logger.exception("State broadcast failed")


class SilentAlertManager(AlertManager):
    """No-op used when alerts are disabled or in tests."""

    name = "silent"

    def alert(self, event: ConfirmedEvent) -> None:
        return


def build_alert_manager(
    alerts_config,
    broadcaster: EventBroadcaster | None = None,
) -> AlertManager:
    if not alerts_config.enabled:
        return SilentAlertManager()
    if alerts_config.webhook_url or alerts_config.email_to:
        # Webhook/email backends are future additions; log a clear notice instead
        # of silently pretending to send them.
        logger.info(
            "Webhook/email alert targets are configured but not yet implemented;"
            " defaulting to console+dashboard alerts."
        )
    return ConsoleAlertManager(log=alerts_config.log, broadcaster=broadcaster)