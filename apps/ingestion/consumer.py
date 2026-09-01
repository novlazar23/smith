from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

try:
    from confluent_kafka import Consumer as _KafkaConsumer

    REDPANDA_AVAILABLE = True
except ImportError:
    _KafkaConsumer = None
    REDPANDA_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketDataProcessor:
    """Processes raw market events into structured format."""

    def process_candle(self, raw_event: dict[str, Any]) -> dict[str, Any]:
        """Converts a raw candle event into standard format.

        Returns:
            A dict with keys: symbol, open, high, low, close, volume,
            timestamp, type — plus any source metadata present in the raw
            event (venue, instrument, open_time, close_time, trade_count,
            is_closed), which the persistence layer relies on.
        """
        required = ("symbol", "open", "high", "low", "close", "volume", "timestamp")
        missing = [f for f in required if f not in raw_event]
        if missing:
            raise ValueError(f"Missing required candle fields: {missing}")

        symbol = str(raw_event["symbol"])
        open_price = float(raw_event["open"])
        high_price = float(raw_event["high"])
        low_price = float(raw_event["low"])
        close_price = float(raw_event["close"])
        volume = float(raw_event["volume"])
        ts_raw = raw_event["timestamp"]

        if isinstance(ts_raw, (int, float)):
            timestamp = datetime.fromtimestamp(ts_raw, tz=UTC)
        elif isinstance(ts_raw, str):
            try:
                timestamp = datetime.fromisoformat(ts_raw).replace(tzinfo=UTC)
            except (ValueError, TypeError):
                timestamp = datetime.now(tz=UTC)
        elif isinstance(ts_raw, datetime):
            if ts_raw.tzinfo is None:
                ts_raw = ts_raw.replace(tzinfo=UTC)
            timestamp = ts_raw
        else:
            raise ValueError(f"Unsupported timestamp type: {type(ts_raw)}")

        if volume < 0:
            raise ValueError(f"Volume must be >= 0, got {volume}")

        # Swap high/low if they are inverted
        if high_price < low_price:
            high_price, low_price = low_price, high_price

        event = {
            "symbol": symbol,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timestamp": timestamp,
            "type": "candle",
        }
        # Quell-Metadaten durchreichen (die Persistenz braucht z.B. die
        # Venue-Spalte für die ClickHouse-Zuordnung).
        for field in ("venue", "instrument", "open_time", "close_time", "trade_count", "is_closed"):
            if field in raw_event:
                event[field] = raw_event[field]
        return event

    def process_tick(self, raw_event: dict[str, Any]) -> dict[str, Any]:
        """Converts a raw tick event into standard format.

        Returns:
            A dict with keys: symbol, price, volume, timestamp, type.
        """
        required = ("symbol", "price", "volume", "timestamp")
        missing = [f for f in required if f not in raw_event]
        if missing:
            raise ValueError(f"Missing required tick fields: {missing}")

        symbol = str(raw_event["symbol"])
        price = float(raw_event["price"])
        volume = float(raw_event["volume"])
        ts_raw = raw_event["timestamp"]

        if isinstance(ts_raw, (int, float)):
            timestamp = datetime.fromtimestamp(ts_raw, tz=UTC)
        elif isinstance(ts_raw, str):
            try:
                timestamp = datetime.fromisoformat(ts_raw).replace(tzinfo=UTC)
            except (ValueError, TypeError):
                timestamp = datetime.now(tz=UTC)
        elif isinstance(ts_raw, datetime):
            if ts_raw.tzinfo is None:
                ts_raw = ts_raw.replace(tzinfo=UTC)
            timestamp = ts_raw
        else:
            raise ValueError(f"Unsupported timestamp type: {type(ts_raw)}")

        if volume < 0:
            raise ValueError(f"Volume must be >= 0, got {volume}")

        return {
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": timestamp,
            "type": "tick",
        }

    def validate_event(self, event: dict[str, Any]) -> bool:
        """Validates that an event has the required top-level fields."""
        required = ("symbol", "timestamp", "type")
        return all(f in event for f in required)


class DataIngestionService:
    """Consumes market data from Redpanda topics."""

    def __init__(
        self,
        topic: str = "market_data",
        bootstrap_servers: list[str] | None = None,
        processor: MarketDataProcessor | None = None,
    ) -> None:
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers or ["localhost:9092"]
        self.processor = processor or MarketDataProcessor()
        self._processed_events: list[dict[str, Any]] = []

    def consume(
        self,
        handler: Callable[[dict[str, Any]], None],
        max_messages: int | None = None,
    ) -> None:
        """Verbraucht Messages vom Topic und ruft handler für jedes Event auf.

        Jede gültige Message wird als Candle verarbeitet (Fallback: Tick);
        unparsierbare oder nicht unterstützte Messages werden geloggt und
        übersprungen. Mit ``max_messages`` beendet sich der Loop nach N
        erfolgreich verarbeiteten Events.
        """
        if not REDPANDA_AVAILABLE or _KafkaConsumer is None:
            logger.warning(
                "Kafka-Client nicht verfügbar — kein Live-Consume möglich. "
                "Nutze process_batch für die Offline-Verarbeitung."
            )
            return

        servers = self.bootstrap_servers or ["redpanda:9092"]
        bootstrap = servers if isinstance(servers, str) else ",".join(servers)
        consumer = _KafkaConsumer(
            {
                "bootstrap.servers": bootstrap,
                "group.id": "ingestion-consumer",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
            }
        )
        consumer.subscribe([self.topic])

        processed = 0
        try:
            while max_messages is None or processed < max_messages:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error() is not None:
                    logger.warning("Kafka-Consumer-Fehler: %s", msg.error())
                    continue

                raw_value = msg.value()
                if raw_value is None:
                    logger.warning("Message ohne Payload übersprungen.")
                    continue
                try:
                    raw = json.loads(raw_value)
                except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
                    logger.warning("Unparsierbare Message übersprungen: %s", exc)
                    continue
                if not isinstance(raw, dict):
                    logger.warning("Nicht-Dict-Payload übersprungen.")
                    continue

                try:
                    event = self.processor.process_candle(raw)
                except ValueError:
                    try:
                        event = self.processor.process_tick(raw)
                    except ValueError:
                        logger.warning(
                            "Nicht unterstütztes Event übersprungen: %s",
                            raw.get("symbol", "unknown"),
                        )
                        continue
                handler(event)
                self._processed_events.append(event)
                processed += 1
        finally:
            consumer.close()

    def process_batch(self, raw_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Processes a batch of raw events.

        Returns only successfully processed events; invalid events are
        logged as warnings and skipped.
        """
        results: list[dict[str, Any]] = []
        for raw in raw_events:
            try:
                event_type = raw.get("type", "candle")
                if event_type == "tick":
                    processed = self.processor.process_tick(raw)
                else:
                    processed = self.processor.process_candle(raw)
                results.append(processed)
                self._processed_events.append(processed)
            except ValueError as exc:
                logger.warning("Skipping invalid event: %s", exc)
        return results

    def get_processed_stream(self) -> list[dict[str, Any]]:
        """Returns the accumulated list of processed events."""
        return list(self._processed_events)
