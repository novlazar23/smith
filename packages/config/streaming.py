"""Streaming Configuration — Redpanda / Kafka-Konfiguration.

Enthält Broker-Verbindungen, Topic-Definitionen und Consumer-Gruppen.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RedpandaConfig(BaseModel):
    """Redpanda/Kafka-Verbindungskonfiguration."""

    bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka/Redpanda Bootstrap-Servers.",
    )
    client_id: str = Field(
        default="trading-orchestra",
        description="Client-ID für Broker-Logging.",
    )
    compression_type: str = Field(
        default="none",
        description="Kompression: none, gzip, lz4, snappy, zstd.",
    )
    acks: str = Field(
        default="all",
        description="Acknowledge-Level: 0, 1, all.",
    )
    retry_max: int = Field(
        default=3, ge=0, description="Max. Retry-Versuche."
    )
    request_timeout_ms: int = Field(
        default=30000, ge=1000, description="Timeout in ms."
    )
    enable_idempotence: bool = Field(
        default=True, description="Idempotente Producer aktivieren."
    )


class ConsumerConfig(BaseModel):
    """Kafka Consumer-Konfiguration."""

    group_id: str = Field(
        default="trading-consumer-group",
        description="Consumer-Group-ID."
    )
    auto_offset_reset: str = Field(
        default="earliest",
        description="Offset-Reset: earliest, latest, none.",
    )
    enable_auto_commit: bool = Field(
        default=True, description="Auto-Commit aktivieren."
    )
    session_timeout_ms: int = Field(
        default=30000, ge=1000, description="Session-Timeout in ms."
    )
    max_poll_records: int = Field(
        default=500, ge=1, description="Max. Records pro Poll."
    )


class StreamingConfig(BaseModel):
    """Gesamte Streaming-Konfiguration."""

    redpanda: RedpandaConfig = Field(default_factory=RedpandaConfig)
    consumer: ConsumerConfig = Field(default_factory=ConsumerConfig)

    # Topic-Namen
    market_data_topic: str = Field(
        default="market.data",
        description="Topic für Marktdaten (Candles, Trades, Orderbook)."
    )
    analysis_request_topic: str = Field(
        default="analysis.requests",
        description="Topic für Analyse-Anfragen."
    )
    agent_report_topic: str = Field(
        default="agents.reports",
        description="Topic für Agenten-Berichte."
    )
    final_decision_topic: str = Field(
        default="trading.decisions",
        description="Topic für Endentscheidungen."
    )
    order_topic: str = Field(
        default="trading.orders",
        description="Topic für Orders."
    )
    dead_letter_topic: str = Field(
        default="dead.letter",
        description="Dead Letter Queue Topic."
    )
