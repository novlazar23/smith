"""Synthetischer Market-Data-Producer.

Erzeugt kontinuierlich synthetische Candle-Events aus dem ``DummyAdapter``
und publishen sie als flache JSON-Events auf das Redpanda-Topic
``market_data``.
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
from packages.ingestion.adapter.dummy import INSTRUMENT_BASE_PRICES, DummyAdapter

logger = logging.getLogger(__name__)

DEFAULT_TOPIC = "market_data"
DEFAULT_INTERVAL_SECONDS = 60.0
FALLBACK_BASE_PRICE = 100.0
HEARTBEAT_PATH = "/tmp/heartbeat"
VENUE = "DUMMY_EXCHANGE"


class DummyMarketDataProducer:
    """Produziert synthetische Candle-Events aus dem DummyAdapter.

    Fuer jedes Symbol wird ein eigener ``DummyAdapter`` mit dem aus
    ``INSTRUMENT_BASE_PRICES`` bekannten Basispreis erzeugt (Fallback
    ``FALLBACK_BASE_PRICE``). Pro Tick wird pro Symbol genau eine Kerze
    abgerufen und als flaches JSON-Event auf das konfigurierte Topic
    gepublished.
    """

    def __init__(
        self,
        symbols: list[str],
        bootstrap_servers: str,
        topic: str = DEFAULT_TOPIC,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        """Initialisiert confluent-Producer und Adapter.

        Args:
            symbols: Liste der zu produzierenden Instrumente.
            bootstrap_servers: Redpanda Bootstrap-Server (z.B. "redpanda:9092").
            topic: Ziel-Topic.
            interval_seconds: Abstand zwischen den Ticks in Sekunden.
        """
        self._symbols = list(symbols)
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._interval_seconds = interval_seconds
        self._producer = Producer({"bootstrap.servers": bootstrap_servers})
        self._adapters: dict[str, DummyAdapter] = {
            symbol: DummyAdapter(base_price=INSTRUMENT_BASE_PRICES.get(symbol, FALLBACK_BASE_PRICE))
            for symbol in self._symbols
        }

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
            adapter = self._adapters[symbol]
            candles = await adapter._fetch_candles_raw(symbol, interval="1m", limit=1)
            for candle in candles:
                # Der Base-Validator lehnt Roh-Kerzen ohne "type" ab — daher
                # Metadaten selbst stempeln (identisch zu fetch_candles).
                candle.setdefault("type", "candle")
                candle.setdefault("instrument", symbol)
                candle.setdefault("venue", adapter.venue)
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

        for symbol in self._symbols:
            await self._adapters[symbol].connect()
        logger.info(
            "Market-Data-Producer gestartet: symbols=%s topic=%s interval=%.1fs",
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
            for adapter in self._adapters.values():
                with suppress(Exception):
                    await adapter.disconnect()
            self._producer.flush(10)
