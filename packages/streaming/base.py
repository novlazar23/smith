"""Base — Abstrakte Producer/Consumer-Protocols für Event Streaming."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol


class EventPartitionKey(StrEnum):
    """Standard-Partitionierungsschlüssel für Events."""

    INSTRUMENT = "instrument"
    VENUE = "venue"
    RUN_ID = "run_id"
    NEWS_ID = "news_id"
    NONE = "none"


class CompressionType(StrEnum):
    """Unterstützte Komprimierungstypen."""

    NONE = "none"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    GZIP = "gzip"


@dataclass(frozen=True)
class StreamConfig:
    """Konfiguration für Event Streaming Topics."""

    bootstrap_servers: str = "localhost:9092"
    default_topic: str = "trading-events"
    partition_key: EventPartitionKey = EventPartitionKey.INSTRUMENT
    compression: CompressionType = CompressionType.NONE
    max_in_flight: int = 100
    acks: str = "all"
    retries: int = 3
    delivery_timeout: int = 30000  # ms
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 3000
    auto_offset_reset: str = "earliest"
    enable_idempotence: bool = True
    enable_transaction: bool = False
    consumer_group: str = "trading-orchestra"
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000
    dlq_topic: str = "trading-events-dlq"
    metadata_topic: str = "trading-source-metadata"

    @property
    def broker_url(self) -> str:
        return self.bootstrap_servers


@dataclass(frozen=True)
class Envelope:
    """Event-Envelope mit Metadaten."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    source: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    partition_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.schema_version:
            object.__setattr__(self, "schema_version", "1.0.0")


class Producer(ABC):
    """Abstrakter Producer für Event-Streams."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Prüft, ob die Verbindung zum Broker aktiv ist."""

    @abstractmethod
    async def send(self, topic: str, payload: dict[str, Any],
                   key: str | None = None, headers: dict[str, str] | None = None) -> bool:
        """Sendet ein Event an ein Topic."""

    @abstractmethod
    async def send_batch(self, topic: str, events: list[dict[str, Any]],
                         keys: list[str | None] | None = None) -> int:
        """Sendet Events in einem Batch."""

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Health-Check des Brokers."""

    @abstractmethod
    async def close(self) -> None:
        """Schließt die Producer-Verbindung."""


class Consumer(ABC):
    """Abstrakter Consumer für Event-Streams."""

    @abstractmethod
    def subscribe(self, topics: list[str], group_id: str | None = None) -> bool:
        """Abonniert Topics."""

    @abstractmethod
    async def poll(self, max_records: int = 100) -> list[Envelope]:
        """Holt Events vom Broker."""

    @abstractmethod
    async def commit(self) -> bool:
        """Bestätigt die Verarbeitung gelesener Events."""

    @abstractmethod
    async def seek_to_timestamp(self, topic: str, timestamp: datetime) -> bool:
        """Setzt den Consumer-Leseposition zu einem Zeitpunkt."""

    @abstractmethod
    async def seek_to_offset(self, topic: str, partition: int, offset: int) -> bool:
        """Setzt den Consumer-Leseposition zu einem Offset."""

    @abstractmethod
    async def close(self) -> None:
        """Schließt die Consumer-Verbindung."""


class DeadLetterHandler(Protocol):
    """Protocol für Dead-Letter-Queue-Handler."""

    async def handle(self, event: dict[str, Any], error: Exception) -> None:
        """Speichert ein fehlgeschlagenes Event in der DLQ."""

    async def replay(self, event_id: str) -> dict[str, Any] | None:
        """Spielt ein DLQ-Event zurück in den Stream."""

    async def list_dead_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Listet tote Events."""
