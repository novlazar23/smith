"""Abstract base class for exchange ingestion adapters.

Defines the common interface with reconnect logic, rate limiting,
heartbeat monitoring, and event validation that all exchange adapters
(Binance Spot, Binance Futures, etc.) inherit from.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from packages.domain.data_quality.validator import (
    MarketDataValidator,
    ValidationResult,
)
from packages.streaming.schemas import (
    SourceMetadata,
)


class ConnectionState(StrEnum):
    """Verbindungszustände des Adapters."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


@dataclass(frozen=True)
class ConnectionConfig:
    """Konfiguration für Exchange-Verbindungen.

    Felder: api_key, api_secret, base_url, ws_url, reconnect_delay,
            max_reconnect_attempts, heartbeat_interval,
            rate_limit_per_second.
    """

    api_key: str
    api_secret: str
    base_url: str = ""
    ws_url: str = ""
    reconnect_delay: float = 1.0
    max_reconnect_attempts: int = 10
    heartbeat_interval: float = 30.0
    rate_limit_per_second: int = 10


class ConnectionError(Exception):  # noqa: A001
    """Wird geworfen wenn die Verbindung fehl schlägt."""


class RateLimitError(Exception):
    """Wird geworfen wenn das Rate-Limit erreicht ist."""


class ExchangeAdapterBase(ABC):
    """Abstrakte Basisklasse für Exchange-Ingestion-Adapter.

    Bietet:
    - Verbindungsaufbau und Reconnect-Logik
    - Rate-Limit-Management
    - Heartbeat-Monitoring
    - Event-Validierung vor Publishing
    """

    def __init__(self, config: ConnectionConfig) -> None:
        self.config = config
        self._state = ConnectionState.DISCONNECTED
        self._lock = threading.Lock()
        self._running = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._rate_limit_lock = asyncio.Lock()
        self._request_timestamps: list[float] = []
        self._validator = MarketDataValidator()
        self._sequence_counter: int = 0
        self._sequence_lock = threading.Lock()

        self.logger = logging.getLogger(self.__module__)

    # -- properties --

    @property
    def is_connected(self) -> bool:
        """True wenn der Adapter verbunden ist."""
        return self._state == ConnectionState.CONNECTED

    @property
    def connection_state(self) -> str:
        """Aktueller Verbindungsstatus.

        Returns:
            'disconnected', 'connecting', 'connected', 'reconnecting'
        """
        return self._state.value

    # -- abstract lifecycle --

    @abstractmethod
    async def connect(self) -> None:
        """Stellt Verbindung zum Exchange her."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Trennt die Verbindung sauber."""
        ...

    @abstractmethod
    async def subscribe(self, streams: list[str]) -> None:
        """Abonniert Exchange-Streams."""
        ...

    # -- reconnect with exponential backoff --

    async def _reconnect(self) -> None:
        """Wiederholter Connect-Versuch mit exponentiellem Backoff.

        Versucht bis zu ``max_reconnect_attempts`` mal, mit exponentiell
        wachsenden Pausen zwischen den Versuchen.
        """
        for attempt in range(1, self.config.max_reconnect_attempts + 1):
            delay = self.config.reconnect_delay * (2 ** (attempt - 1))
            self._state = ConnectionState.RECONNECTING
            self.logger.warning(
                "Reconnect attempt %d/%d in %.1fs ...",
                attempt,
                self.config.max_reconnect_attempts,
                delay,
            )
            try:
                await asyncio.sleep(delay)
                if self._running:
                    await self.connect()
                    self._state = ConnectionState.CONNECTED
                    self.logger.info(
                        "Reconnect erfolgreich nach %d Versuchen.", attempt,
                    )
                    return
            except ConnectionError:
                self.logger.warning(
                    "Reconnect-Versuch %d fehlgeschlagen.", attempt,
                )
            except asyncio.CancelledError:
                self.logger.info("Reconnect abgebrochen.")
                return

        self._state = ConnectionState.DISCONNECTED
        raise ConnectionError(
            f"Connect fehlgeschlagen nach "
            f"{self.config.max_reconnect_attempts} Versuchen"
        )

    # -- rate-limiting (sliding window) --

    async def _rate_limit_wait(self) -> None:
        """Wartet bis das Rate-Limit zur Verfügung steht.

        Implementiert einen Token-Bucket-Ansatz mit Sliding Window:
        Alle Anfragen innerhalb der letzten Sekunde werden gezählt.
        """
        async with self._rate_limit_lock:
            now = time.monotonic()
            window = 1.0
            max_requests = self.config.rate_limit_per_second

            self._request_timestamps = [
                ts for ts in self._request_timestamps if now - ts < window
            ]

            if len(self._request_timestamps) >= max_requests:
                wait_time = self._request_timestamps[0] + window - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    self._request_timestamps.clear()

            self._request_timestamps.append(time.monotonic())

    # -- heartbeat --

    async def _send_heartbeat(self) -> None:
        """Sendet einen Heartbeat/Ping und erwartet einen Pong.

        Muss von Unterklassen implementiert werden, die spezifische
        Protokolle nutzen (z.B. WebSocket Ping/Pong oder API-Endpoint).
        """
        await self._rate_limit_wait()
        self.logger.debug("Heartbeat gesendet.")

    # -- validation & publishing --

    async def _validate_and_publish(self, raw_event: dict[str, Any]) -> bool:
        """Validiert ein Event und publish bei Erfolg.

        Args:
            raw_event: Rohes Event-Dict vom Exchange.

        Returns:
            ``True`` wenn das Event validiert wurde und gepublished wurde,
            ``False`` wenn die Validierung fehlgeschlagen ist.
        """
        try:
            result: ValidationResult = self._validator.validate(raw_event)
        except Exception as exc:
            self.logger.warning("Validierung fehlgeschlagen: %s", exc)
            return False

        if not result.is_valid:
            self.logger.warning(
                "Event ungültig (type=%s, score=%.2f): %s",
                result.event_type,
                result.quality_score,
                [i.message for i in result.issues],
            )
            return False

        self._publish_event(raw_event)
        self.logger.debug("Event published: type=%s", result.event_type)
        return True

    def _publish_event(self, raw_event: dict[str, Any]) -> None:
        """Publish ein validiertes Event an den Stream.

        Muss von Unterklassen implementiert werden, um die Verbindung
        zum Message Broker (z.B. Redpanda/Kafka) herzustellen.
        """
        self.logger.debug(
            "Publish event (subclass hook): %s",
            raw_event.get("type", "unknown"),
        )

    # -- helpers --

    def _next_sequence(self) -> int:
        """Erzeugt eine eindeutige Sequenznummer."""
        with self._sequence_lock:
            self._sequence_counter += 1
            return self._sequence_counter

    def _build_metadata(self, source: str, venue: str) -> SourceMetadata:
        """Erzeugt SourceMetadata für ein Event."""
        return SourceMetadata(
            source=source,
            venue=venue,
            event_time=datetime.now(UTC),
        )

    # -- health check --

    async def health_check(self) -> dict[str, Any]:
        """Ausführlicher Health-Check des Adapters.

        Returns:
            Dict mit Status-Informationen.
        """
        return {
            "connected": self.is_connected,
            "state": self.connection_state,
            "api_key_set": bool(self.config.api_key),
            "base_url": self.config.base_url or "not set",
            "ws_url": self.config.ws_url or "not set",
            "rate_limit": self.config.rate_limit_per_second,
            "heartbeat_interval": self.config.heartbeat_interval,
        }

    # -- heartbeat loop --

    async def start_heartbeat(self) -> None:
        """Startet den periodischen Heartbeat-Loop."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            return

        async def _loop() -> None:
            while self._running and self.is_connected:
                try:
                    await self._send_heartbeat()
                except Exception as exc:
                    self.logger.warning("Heartbeat fehlgeschlagen: %s", exc)
                await asyncio.sleep(self.config.heartbeat_interval)

        self._heartbeat_task = asyncio.create_task(_loop())

    async def stop_heartbeat(self) -> None:
        """Stoppt den periodischen Heartbeat-Loop."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
        self._heartbeat_task = None

    # -- async context manager --

    async def __aenter__(self) -> ExchangeAdapterBase:
        """Async-Kontextmanager: startet Verbindung."""
        await self.connect()
        self._running = True
        await self.start_heartbeat()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc_val: BaseException | None,
                        exc_tb: Any) -> None:  # noqa: ANN401
        """Async-Kontextmanager: stoppt Heartbeat und trennt Verbindung."""
        self._running = False
        await self.stop_heartbeat()
        await self.disconnect()
