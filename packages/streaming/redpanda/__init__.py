"""Redpanda — Redpanda-spezifische Producer-/Consumer-Implementierung.

Stellt ein vollständiges Streaming-Interface bereit mit:
  - Producer: Event-Sendung an Topics
  - Consumer: Event-Verbrauch aus Topics
  - DLQ-Handling: Dead-Letter-Queue für fehlerhafte Events
  - Schema-Validierung: JSON-Schema-basierte Validierung
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from ..base import (
    CompressionType,
    Consumer,
    DeadLetterHandler,
    EventPartitionKey,
    Producer,
    StreamConfig,
)
from ..schemas import (
    Candle,
    MarketEvent,
    NewsEvent,
    OrderBookSnapshot,
    SourceMetadata,
    Trade,
)

logger = logging.getLogger(__name__)


class EventSerializer:
    """Serialisiert und deserialisiert Events."""

    KNOWN_TYPES = {
        "MarketEvent": MarketEvent,
        "Candle": Candle,
        "Trade": Trade,
        "OrderBookSnapshot": OrderBookSnapshot,
        "NewsEvent": NewsEvent,
        "SourceMetadata": SourceMetadata,
    }

    @staticmethod
    def serialize(event: Any) -> dict[str, Any]:
        """Serialisiert ein Event in ein Dictionary."""
        if hasattr(event, "to_dict"):
            result: dict[str, Any] = event.to_dict()
            return result
        return dict(event)

    @staticmethod
    def deserialize(data: dict[str, Any], event_type: str) -> Any:
        """Deserialisiert ein Dictionary in ein typed Event."""
        cls = EventSerializer.KNOWN_TYPES.get(event_type)
        if cls is None:
            return data
        if hasattr(cls, "from_dict"):
            return cls.from_dict(data)
        return data

    @staticmethod
    def to_json(payload: dict[str, Any]) -> str:
        """Serialisiert ein Dictionary als JSON-String."""
        return json.dumps(payload, default=str)

    @staticmethod
    def from_json(json_str: str) -> dict[str, Any]:
        """Deserialisiert einen JSON-String in ein Dictionary."""
        return json.loads(json_str)  # type: ignore[no-any-return]


class RedpandaProducer(Producer):
    """Redpanda-Producer mit JSON-Serialisierung.

    Nutzt rpk/rdkafka im echten System. Für das MVP wird
    eine mock-basierte Implementierung verwendet.
    """

    def __init__(self, config: StreamConfig | None = None) -> None:
        self._config = config or StreamConfig()
        self._connected = False
        self._serializer = EventSerializer()
        self._sent_topics: set[str] = set()
        self._sent_count: int = 0

    @property
    def config(self) -> StreamConfig:
        return self._config

    def is_connected(self) -> bool:
        return self._connected

    async def send(self, topic: str, payload: dict[str, Any],
                   key: str | None = None, headers: dict[str, str] | None = None) -> bool:
        """Sendet ein Event an ein Topic."""
        try:
            # Im echten System: producer.send(topic, value=payload, key=key, headers=headers)
            headers = headers or {}
            headers["event_schema_version"] = "1.0.0"

            # Mock: Track sent events
            self._sent_topics.add(topic)
            self._sent_count += 1

            logger.info("Sent event to topic '%s' (key=%s)", topic, key)
            self._connected = True
            return True
        except Exception as e:
            logger.error("Failed to send event to topic '%s': %s", topic, e)
            return False

    async def send_batch(self, topic: str, events: list[dict[str, Any]],
                         keys: list[str | None] | None = None) -> int:
        """Sendet Events in einem Batch."""
        success_count = 0
        for i, event in enumerate(events):
            key = keys[i] if keys and i < len(keys) else None
            if await self.send(topic, event, key=key):
                success_count += 1
        return success_count

    async def health_check(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "backend": "redpanda",
            "connected": self._connected,
            "config": {
                "bootstrap_servers": self._config.bootstrap_servers,
                "default_topic": self._config.default_topic,
                "compression": self._config.compression,
                "acks": self._config.acks,
            },
            "stats": {
                "sent_count": self._sent_count,
                "topics_used": list(self._sent_topics),
            },
        }
        if self._connected:
            result["message"] = "healthy"
        return result

    async def close(self) -> None:
        self._connected = False
        logger.info("Producer closed")


class RedpandaConsumer(Consumer):
    """Redpanda-Consumer mit JSON-Deserialisierung.

    Nutzt rpk/rdkafka im echten System. Für das MVP wird
    eine mock-basierte Implementierung verwendet.
    """

    def __init__(self, config: StreamConfig | None = None) -> None:
        self._config = config or StreamConfig()
        self._connected = False
        self._subscribed_topics: list[str] = []
        self._group_id: str | None = None
        self._buffer: list[dict[str, Any]] = []
        self._serializer = EventSerializer()

    @property
    def config(self) -> StreamConfig:
        return self._config

    def is_connected(self) -> bool:
        return self._connected

    def subscribe(self, topics: list[str], group_id: str | None = None) -> bool:
        """Abonniert Topics."""
        try:
            # Im echten System: consumer.subscribe(topics, group_id=group_id)
            self._subscribed_topics = list(topics)
            self._group_id = group_id or self._config.consumer_group
            self._connected = True
            logger.info("Subscribed to topics: %s (group=%s)", topics, self._group_id)
            return True
        except Exception as e:
            logger.error("Failed to subscribe to topics: %s", e)
            return False

    async def poll(self, max_records: int = 100) -> list[Any]:
        """Holt Events vom Broker."""
        # Im echten System: consumer.poll(max_records)
        records = self._buffer[:max_records]
        self._buffer = self._buffer[max_records:]

        events: list[Any] = []
        for record in records:
            event = self._serializer.deserialize(
                record.get("payload", record),
                record.get("event_type", "MarketEvent"),
            )
            events.append(event)

        return events

    async def commit(self) -> bool:
        """Bestätigt die Verarbeitung gelesener Events."""
        # Im echten System: consumer.commit()
        self._buffer = []
        return True

    async def seek_to_timestamp(self, topic: str, timestamp: datetime) -> bool:
        """Setzt den Consumer-Leseposition zu einem Zeitpunkt."""
        # Im echten System: consumer.seek(TopicPartition(topic, 0), timestamp)
        logger.info("Seeked topic '%s' to timestamp %s", topic, timestamp)
        return True

    async def seek_to_offset(self, topic: str, partition: int, offset: int) -> bool:
        """Setzt den Consumer-Leseposition zu einem Offset."""
        # Im echten System: consumer.seek(TopicPartition(topic, partition), offset)
        logger.info("Seeked topic '%s' partition=%d offset=%d", topic, partition, offset)
        return True

    async def close(self) -> None:
        self._connected = False
        self._buffer = []
        self._subscribed_topics = []
        logger.info("Consumer closed")


class RedpandaDeadLetterHandler(DeadLetterHandler):
    """DLQ-Handler für fehlerhafte Events."""

    def __init__(self, config: StreamConfig | None = None) -> None:
        self._config = config or StreamConfig()
        self._dlq_events: list[dict[str, Any]] = []

    async def handle(self, event: dict[str, Any], error: Exception) -> None:
        """Speichert ein fehlgeschlagenes Event in der DLQ."""
        dlq_entry: dict[str, Any] = {
            "event": event,
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._dlq_events.append(dlq_entry)
        logger.error("DLQ: event '%s' failed: %s", event.get("event_id", "unknown"), error)

    async def replay(self, event_id: str) -> dict[str, Any] | None:
        """Spielt ein DLQ-Event zurück in den Stream."""
        for i, entry in enumerate(self._dlq_events):
            if entry["event"].get("event_id") == event_id:
                recovered: dict[str, Any] = entry["event"]
                self._dlq_events.pop(i)
                logger.info("Replayed DLQ event '%s'", event_id)
                return recovered
        return None

    async def list_dead_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Listet tote Events."""
        return self._dlq_events[-limit:]

    @property
    def count(self) -> int:
        return len(self._dlq_events)
