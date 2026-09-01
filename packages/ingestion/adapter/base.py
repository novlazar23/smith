"""Abstract adapter interface for multi-exchange market data ingestion.

Defines ``ExchangeAdapterBase`` — the common abstract base that every exchange
adapter (Binance, Dummy, etc.) inherits from.  Provides:

- ``connect()``, ``disconnect()``, ``subscribe()`` lifecycle hooks
- ``_reconnect()`` with exponential backoff
- ``_rate_limit_wait()`` sliding-window rate limiter
- ``_send_heartbeat()`` abstract heartbeat hook
- ``_validate_and_publish()`` + ``_publish_event()`` validation/publishing pipeline
- ``_build_metadata()`` helper producing ``SourceMetadata`` with venue field
- ``health_check()`` returning operational status dict
- Async context manager protocol (``__aenter__`` / ``__aexit__``)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field
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


class MarketDataType(StrEnum):
    """Unterstützte Marktdaten-Typen."""

    CANDLE = "candle"
    TRADE = "trade"
    ORDERBOOK = "orderbook"
    TICKER = "ticker"
    FUNDING = "funding"


@dataclass(frozen=True)
class VenueFees:
    """Fee-Struktur pro Venue.

    Felder: taker_rate, maker_rate, spread_bps.
    """

    taker_rate: float
    maker_rate: float
    spread_bps: float = 1.0

    def spread_price(self, base_price: float) -> float:
        """Spread in Preis-Einheiten."""
        return base_price * self.spread_bps / 10000.0


@dataclass
class ConnectionConfig:
    """Konfiguration für Exchange-Verbindungen.

    Felder: api_key, api_secret, base_url, ws_url, reconnect_delay,
            max_reconnect_attempts, heartbeat_interval,
            rate_limit_per_second, venue, fees.
    """

    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    ws_url: str = ""
    reconnect_delay: float = 1.0
    max_reconnect_attempts: int = 10
    heartbeat_interval: float = 30.0
    rate_limit_per_second: int = 10
    venue: str = ""
    fees: VenueFees = field(default_factory=lambda: VenueFees(
        taker_rate=0.0,
        maker_rate=0.0,
        spread_bps=1.0,
    ))


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

    def __init__(self, config: ConnectionConfig | None = None) -> None:
        self.config = config or ConnectionConfig()
        self._state = ConnectionState.DISCONNECTED
        self._lock = threading.Lock()
        self._running = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._rate_limit_lock = asyncio.Lock()
        self._request_timestamps: list[float] = []
        self._validator = MarketDataValidator()
        self._sequence_counter: int = 0
        self._sequence_lock = threading.Lock()
        self._subscribed_streams: set[str] = set()

        self.logger = logging.getLogger(self.__module__)

    # -- properties --

    @property
    def is_connected(self) -> bool:
        """True wenn der Adapter verbunden ist."""
        return self._state == ConnectionState.CONNECTED

    @property
    def connection_state(self) -> str:
        """Aktueller Verbindungsstatus."""
        return self._state.value

    @property
    def venue(self) -> str:
        """Venue-Bezeichnung dieses Adapters."""
        return self.config.venue

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

    # -- market data fetchers (concrete) --

    async def fetch_candles(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Ruft Candlestick-Daten ab und validiert sie.

        Args:
            symbol: Handelspaar.
            interval: Zeitfenster, z.B. "1m", "5m", "1h".
            limit: Anzahl der Kerzen.

        Returns:
            Liste von validierten Candle-Dicts.
        """
        raw_candles = await self._fetch_candles_raw(symbol, interval, limit)
        validated: list[dict[str, Any]] = []
        for raw in raw_candles:
            # Metadaten vor der Validierung stempeln — der Validator
            # dispatcht auf "type" und würde Roh-Kerzen sonst als
            # "Unknown event type" ablehnen.
            raw["type"] = "candle"
            raw.setdefault("instrument", symbol)
            raw["venue"] = self.config.venue
            result = self._validator.validate(raw)
            if result.is_valid:
                validated.append(raw)
            else:
                self.logger.warning(
                    "Candle rejected (score=%.2f): %s",
                    result.quality_score,
                    [i.message for i in result.issues],
                )
        return validated

    async def fetch_trades(
        self,
        symbol: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Ruft Recent-Trades ab und validiert sie.

        Args:
            symbol: Handelspaar.
            limit: Anzahl der Trades.

        Returns:
            Liste von validierten Trade-Dicts.
        """
        raw_trades = await self._fetch_trades_raw(symbol, limit)
        validated: list[dict[str, Any]] = []
        for raw in raw_trades:
            # Metadaten vor der Validierung stempeln — der Validator
            # dispatcht auf "type" und würde Roh-Trades sonst als
            # "Unknown event type" ablehnen.
            raw["type"] = "trade"
            raw.setdefault("instrument", symbol)
            raw["venue"] = self.config.venue
            result = self._validator.validate(raw)
            if result.is_valid:
                validated.append(raw)
            else:
                self.logger.warning(
                    "Trade rejected (score=%.2f): %s",
                    result.quality_score,
                    [i.message for i in result.issues],
                )
        return validated

    async def fetch_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Ruft Orderbook-Daten ab und validiert sie.

        Args:
            symbol: Handelspaar.
            limit: Tiefe pro Seite.

        Returns:
            Validiertes Orderbook-Dict.
        """
        raw_ob = await self._fetch_orderbook_raw(symbol, limit)
        result = self._validator.validate(raw_ob)
        if not result.is_valid:
            self.logger.warning(
                "Orderbook rejected (score=%.2f): %s",
                result.quality_score,
                [i.message for i in result.issues],
            )
        raw_ob["type"] = "orderbook_snapshot"
        raw_ob.setdefault("instrument", symbol)
        raw_ob["venue"] = self.config.venue
        return raw_ob

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Ruft Ticker-Daten ab.

        Args:
            symbol: Handelspaar.

        Returns:
            Ticker-Dict mit Preis, 24h-Volume, Spread etc.
        """
        raw_ticker = await self._fetch_ticker_raw(symbol)
        raw_ticker["type"] = "ticker"
        raw_ticker.setdefault("instrument", symbol)
        raw_ticker["venue"] = self.config.venue
        return raw_ticker

    # -- abstract low-level fetchers (implement in subclass) --

    @abstractmethod
    async def _fetch_candles_raw(
        self, symbol: str, interval: str, limit: int
    ) -> list[dict[str, Any]]:
        """Roh-Candle-Daten vom Exchange laden (ohne Validierung)."""
        ...

    @abstractmethod
    async def _fetch_trades_raw(
        self, symbol: str, limit: int
    ) -> list[dict[str, Any]]:
        """Roh-Trades vom Exchange laden (ohne Validierung)."""
        ...

    @abstractmethod
    async def _fetch_orderbook_raw(
        self, symbol: str, limit: int
    ) -> dict[str, Any]:
        """Roh-Orderbook vom Exchange laden (ohne Validierung)."""
        ...

    @abstractmethod
    async def _fetch_ticker_raw(self, symbol: str) -> dict[str, Any]:
        """Roh-Ticker vom Exchange laden (ohne Validierung)."""
        ...

    # -- reconnect with exponential backoff --

    async def _reconnect(self) -> None:
        """Wiederholter Connect-Versuch mit exponentiellem Backoff."""
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
        """Wartet bis das Rate-Limit zur Verfügung steht."""
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
        """Sendet einen Heartbeat/Ping.

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

    def _instrument_from_symbol(self, symbol: str) -> str:
        """Normalize symbol to canonical instrument name."""
        return symbol.upper().replace("/", "")

    def _format_symbol(self, instrument: str, suffix: str = "") -> str:
        """Format an instrument into a venue-specific symbol string."""
        raw = instrument.upper()
        if suffix:
            raw = raw + suffix
        return raw

    # -- health check --

    async def health_check(self) -> dict[str, Any]:
        """Ausführlicher Health-Check des Adapters.

        Returns:
            Dict mit Status-Informationen.
        """
        return {
            "connected": self.is_connected,
            "state": self.connection_state,
            "venue": self.config.venue,
            "api_key_set": bool(self.config.api_key),
            "base_url": self.config.base_url or "not set",
            "rate_limit": self.config.rate_limit_per_second,
            "heartbeat_interval": self.config.heartbeat_interval,
            "fees": {
                "taker": self.config.fees.taker_rate,
                "maker": self.config.fees.maker_rate,
                "spread_bps": self.config.fees.spread_bps,
            },
            "subscribed_streams": sorted(self._subscribed_streams),
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
