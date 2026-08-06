from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

try:
    from redpanda import RedpandaClient

    REDPANDA_AVAILABLE = True
except ImportError:
    RedpandaClient = None
    REDPANDA_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketDataProcessor:
    """Processes raw market events into structured format."""

    def process_candle(self, raw_event: dict[str, Any]) -> dict[str, Any]:
        """Converts a raw candle event into standard format.

        Returns:
            A dict with keys: symbol, open, high, low, close, volume,
            timestamp, type.
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

        return {
            "symbol": symbol,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timestamp": timestamp,
            "type": "candle",
        }

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
        """Consumes messages from Redpanda and calls handler for each.

        If REDPANDA_AVAILABLE the real client is used; otherwise a warning
        is logged and no messages are processed (mock/test path).
        """
        if not REDPANDA_AVAILABLE:
            logger.warning(
                "Redpanda client not available — skipping live consume. "
                "Use process_batch for offline processing."
            )
            return

        client = RedpandaClient(
            bootstrap_servers=self.bootstrap_servers,
            auto_offset_reset="earliest",
        )

        for idx, message in enumerate(client.consume(self.topic)):
            raw: dict[str, Any] = json.loads(message.value) if isinstance(message.value, bytes) else message.value
            try:
                processed = self.processor.process_candle(raw)
                handler(processed)
            except ValueError:
                try:
                    processed = self.processor.process_tick(raw)
                    handler(processed)
                except ValueError:
                    logger.warning("Skipping unparseable event: %s", raw.get("symbol", "unknown"))
            else:
                self._processed_events.append(processed)

            if max_messages is not None and idx + 1 >= max_messages:
                break

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
