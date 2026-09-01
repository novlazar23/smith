"""Market-Data-Producer (Binance-Live oder Dummy-Synthese).

Produziert kontinuierlich Candle-Events auf das Redpanda-Topic
``market_data``. Die Datenquelle wird ueber den Parameter ``source``
(respektive Env ``MARKET_DATA_SOURCE``) gewaehlt:

- ``"binance"``: letzte geschlossene 1m-Kerze pro Symbol von der
  Binance-Futures-REST-API via ``BinanceAdapter`` (oeffentliche Endpunkte,
  ohne API-Keys).
- ``"dummy"``: synthetische Kerzen aus dem ``DummyAdapter``.

Im Binance-Modus faellt jeder einzelne Tick bei Fehlern (Netzwerk, HTTP,
keine geschlossene Kerze) transparent auf die Dummy-Kerze zurueck
(Venue ``DUMMY_EXCHANGE``); der Producer versucht jeden Tick erneut,
Live-Daten zu liefern. Die Event-Formate beider Pfade sind identisch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from confluent_kafka import KafkaError, KafkaException, Message, Producer
from confluent_kafka.admin import AdminClient, NewTopic
from packages.ingestion.adapter.binance import BinanceAdapter
from packages.ingestion.adapter.dummy import INSTRUMENT_BASE_PRICES, DummyAdapter

logger = logging.getLogger(__name__)

DEFAULT_TOPIC = "market_data"
DEFAULT_INTERVAL_SECONDS = 60.0
FALLBACK_BASE_PRICE = 100.0
HEARTBEAT_PATH = "/tmp/heartbeat"
VENUE = "DUMMY_EXCHANGE"

SOURCE_BINANCE = "binance"
SOURCE_DUMMY = "dummy"
DEFAULT_SOURCE = SOURCE_DUMMY
SUPPORTED_SOURCES = (SOURCE_BINANCE, SOURCE_DUMMY)
LIVE_CANDLE_FETCH_LIMIT = 3


def to_exchange_symbol(instrument: str) -> str:
    """Mappt ein kanonisches Instrument auf das Exchange-Symbol-Format.

    Args:
        instrument: Kanonisches Instrument, z.B. ``"BTC/USDT"``.

    Returns:
        Exchange-Symbol ohne Separator, in Grossbuchstaben (z.B. ``"BTCUSDT"``).
    """
    return instrument.replace("/", "").upper()


def select_last_closed_candle(
    candles: list[dict[str, Any]], now: datetime | None = None
) -> dict[str, Any] | None:
    """Waehlt die letzte geschlossene Kerze aus einer aufsteigenden Liste.

    Eine Kerze gilt als geschlossen, wenn ihre ``close_time`` nicht in der
    Zukunft liegt — die aktuell gebildete Kerze hat eine ``close_time`` in
    der Zukunft und wird damit ausgeschlossen.

    Args:
        candles: Kerzenliste (aufsteigend nach ``open_time``).
        now: Referenzzeitpunkt (Default: aktueller UTC-Zeitpunkt).

    Returns:
        Letzte geschlossene Kerze oder ``None``, wenn keine geschlossen ist.
    """
    moment = now or datetime.now(UTC)
    closed = [candle for candle in candles if candle["close_time"] <= moment]
    return closed[-1] if closed else None


class DummyMarketDataProducer:
    """Produziert Candle-Events aus Binance (live) oder dem DummyAdapter.

    Fuer jedes Symbol wird ein eigener ``DummyAdapter`` mit dem aus
    ``INSTRUMENT_BASE_PRICES`` bekannten Basispreis erzeugt (Fallback
    ``FALLBACK_BASE_PRICE``). Im Binance-Modus dient derselbe DummyAdapter
    als Fallback, wenn der Live-Fetch eines Ticks fehlschlaegt. Pro Tick
    wird pro Symbol genau eine Kerze gepublished als flaches JSON-Event —
    in beiden Modussen mit identischem Event-Format.
    """

    def __init__(
        self,
        symbols: list[str],
        bootstrap_servers: str,
        topic: str = DEFAULT_TOPIC,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        source: str = DEFAULT_SOURCE,
    ) -> None:
        """Initialisiert confluent-Producer, Adapter und Datenquelle.

        Args:
            symbols: Liste der zu produzierenden Instrumente.
            bootstrap_servers: Redpanda Bootstrap-Server (z.B. "redpanda:9092").
            topic: Ziel-Topic.
            interval_seconds: Abstand zwischen den Ticks in Sekunden.
            source: Datenquelle — ``"binance"`` (live) oder ``"dummy"``
                (synthetisch). Unbekannte Werte fallen mit Warnung auf
                ``"dummy"`` zurueck.
        """
        self._symbols = list(symbols)
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._interval_seconds = interval_seconds
        self._source = SOURCE_DUMMY
        if source == SOURCE_BINANCE:
            self._source = SOURCE_BINANCE
        elif source != SOURCE_DUMMY:
            logger.warning("Unbekanntes source='%s' → Fallback auf '%s'", source, SOURCE_DUMMY)
        self._producer = Producer({"bootstrap.servers": bootstrap_servers})
        self._adapters: dict[str, DummyAdapter] = {
            symbol: DummyAdapter(base_price=INSTRUMENT_BASE_PRICES.get(symbol, FALLBACK_BASE_PRICE))
            for symbol in self._symbols
        }
        self._binance_adapter: BinanceAdapter | None = (
            BinanceAdapter() if self._source == SOURCE_BINANCE else None
        )

    # -- Topic-Setup ----------------------------------------------------

    def _ensure_topic(self) -> None:
        """Stellt idempotent sicher, dass das Ziel-Topic existiert.

        Existiert das Topic bereits, wird nichts getan. Andernfalls wird es
        mit 3 Partitionen und Replikationsfaktor 1 angelegt.
        """
        admin = AdminClient({"bootstrap.servers": self._bootstrap_servers})
        existing_topics = set(admin.list_topics(timeout=10).topics)
        if self._topic in existing_topics:
            logger.info("Topic %s existiert bereits", self._topic)
            return
        admin.create_topics([NewTopic(self._topic, num_partitions=3, replication_factor=1)])
        logger.info("Topic %s angelegt (3 Partitionen, RF=1)", self._topic)

    # -- Tick -----------------------------------------------------------

    async def _tick(self) -> int:
        """Fuehrt einen Produktionstick durch.

        Ruft pro Symbol genau eine Kerze ab und publish ein flaches
        JSON-Event pro erhaltener Kerze. Aktualisiert anschliessend die
        Heartbeat-Datei.

        Returns:
            Anzahl der gepushten Events.
        """
        produced = 0
        for symbol in self._symbols:
            for candle in await self._fetch_tick_candles(symbol):
                event = self._build_event(symbol, candle)
                self._producer.produce(
                    self._topic,
                    key=symbol.encode(),
                    value=json.dumps(event, default=str).encode(),
                    on_delivery=self._on_delivery,
                )
                produced += 1
        self._producer.poll(0)
        self._touch_heartbeat()
        logger.debug("Tick fertig: %d Events gepusht", produced)
        return produced

    async def _fetch_tick_candles(self, symbol: str) -> list[dict[str, Any]]:
        """Liefert die Kerzen eines Ticks fuer ein einzelnes Symbol.

        Im Binance-Modus wird zuerst die letzte geschlossene Live-Kerze
        gesucht. Bei jedem Fehler (Netzwerk, HTTP, Validierung, keine
        geschlossene Kerze) wird mit einer Warnung auf den DummyAdapter
        zurueckgefallen — der Live-Versuch erfolgt jeden Tick erneut
        (kein Moduswechsel).
        """
        if self._binance_adapter is not None:
            try:
                candle = await self._fetch_live_closed_candle(symbol)
            except Exception as exc:
                logger.warning(
                    "Binance-Fetch fehlgeschlagen (%s) → Dummy-Fallback: %s", symbol, exc
                )
            else:
                if candle is not None:
                    return [candle]
                logger.warning(
                    "Binance lieferte keine geschlossene 1m-Kerze (%s) → Dummy-Fallback", symbol
                )
        return await self._adapters[symbol].fetch_candles(symbol, "1m", 1)

    async def _fetch_live_closed_candle(self, symbol: str) -> dict[str, Any] | None:
        """Holt die letzte geschlossene 1m-Live-Kerze fuer ein Symbol.

        Die Adapter-Kerzen werden mit dem Exchange-Symbol abgefragt; das
        kanonische Instrument bleibt dem Event vorbehalten.

        Args:
            symbol: Kanonisches Instrument (z.B. ``"BTC/USDT"``).

        Returns:
            Letzte geschlossene Kerze (inkl. gestempeltem
            ``venue=BINANCE_FUTURES``) oder ``None``.
        """
        adapter = self._binance_adapter
        if adapter is None:
            return None
        if not adapter.is_connected:
            await adapter.connect()
        candles = await adapter.fetch_candles(
            to_exchange_symbol(symbol), "1m", LIVE_CANDLE_FETCH_LIMIT
        )
        return select_last_closed_candle(candles)

    def _build_event(self, symbol: str, candle: dict[str, Any]) -> dict[str, Any]:
        """Buildet das flache Candle-Event aus einer gestempelten Kerze.

        Args:
            symbol: Handelspaar, fuer das die Kerze erzeugt wurde.
            candle: Validiertes Candle-Dict des Adapters.

        Returns:
            Flaches Event-Dict (JSON-serialisierbar).
        """
        open_time = candle["open_time"]
        close_time = candle["close_time"]
        return {
            "symbol": symbol,
            "timestamp": open_time.isoformat(),
            "open_time": open_time.isoformat(),
            "close_time": close_time.isoformat(),
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
            "trade_count": candle["trade_count"],
            "is_closed": candle["is_closed"],
            "type": "candle",
            "instrument": symbol,
            "venue": candle.get("venue", VENUE),
        }

    def _on_delivery(self, err: KafkaError | None, msg: Message | None) -> None:
        """Delivery-Callback des confluent Producer: loggt Fehler."""
        del msg
        if err is not None:
            logger.error("Produktion fehlgeschlagen: %s", err)

    def _touch_heartbeat(self) -> None:
        """Aktualisiert die Heartbeat-Datei fuer den Container-Healthcheck."""
        try:
            with Path(HEARTBEAT_PATH).open("a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now(UTC).isoformat()} {self._topic}\n")
        except OSError as exc:
            logger.warning("Heartbeat-Datei nicht schreibbar: %s", exc)

    # -- Lauffaehigkeit ---------------------------------------------------

    async def run(self) -> None:
        """Startet die kontinuierliche Produktion (laeuft bis zur Unterbrechung).

        Stellt sicher, dass das Topic existiert, verbindet alle Adapter und
        fuehrt dann endlos Ticks im Abstand ``interval_seconds`` aus.
        """
        while True:
            try:
                self._ensure_topic()
                break
            except KafkaException as exc:
                logger.warning("Topic-Setup fehlgeschlagen, neuer Versuch in 5s: %s", exc)
                await asyncio.sleep(5)

        if self._binance_adapter is not None:
            try:
                await self._binance_adapter.connect()
                logger.info("Binance-Adapter verbunden (Live-Modus, venue=%s)", self._binance_adapter.venue)
            except Exception as exc:
                logger.warning(
                    "Binance-Adapter-Verbindung fehlgeschlagen — Ticks faellen auf Dummy "
                    "zurueck, Reconnect-Versuch pro Tick: %s", exc
                )
        for symbol in self._symbols:
            await self._adapters[symbol].connect()
        logger.info(
            "Market-Data-Producer gestartet: source=%s symbols=%s topic=%s interval=%.1fs",
            self._source,
            self._symbols,
            self._topic,
            self._interval_seconds,
        )
        try:
            while True:
                try:
                    count = await self._tick()
                    logger.info("Tick: %d Candle-Events gepusht", count)
                except KafkaException as exc:
                    logger.error("Tick fehlgeschlagen: %s", exc)
                await asyncio.sleep(self._interval_seconds)
        finally:
            if self._binance_adapter is not None:
                with suppress(Exception):
                    await self._binance_adapter.disconnect()
            for adapter in self._adapters.values():
                with suppress(Exception):
                    await adapter.disconnect()
            self._producer.flush(10)
