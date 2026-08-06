"""Topics — Topic-Definitionen und -Verwaltung für Event Streaming.

Definiert alle Topics und deren Konfiguration:
  - trading-events: Main event stream
  - trading-source-metadata: Source metadata
  - trading-decisions: Final decisions
  - trading-dead-letter: Dead letter queue
  - Feature-, Strategy- und Risk-Typen
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TopicConfig:
    """Konfiguration für ein einzelnes Topic."""

    name: str
    partitions: int = 6
    replication_factor: int = 3
    retention_ms: int = 604800000  # 7 days
    min_insync_replicas: int = 2
    compression: str = "producer"
    max_message_bytes: int = 1048576  # 1MB
    tags: list[str] = field(default_factory=list)

    @property
    def dlq(self) -> bool:
        return "dlq" in self.tags


class TopicRegistry:
    """Zentrale Registry für alle Event-Typen und deren Topics."""

    # Main event streams
    EVENTS = TopicConfig(
        name="trading-events",
        partitions=6,
        tags=["streaming"],
    )
    SOURCE_METADATA = TopicConfig(
        name="trading-source-metadata",
        partitions=3,
        tags=["streaming", "metadata"],
    )
    DECISIONS = TopicConfig(
        name="trading-decisions",
        partitions=3,
        retention_ms=2592000000,  # 30 days
        tags=["streaming", "decisions"],
    )

    # Feature, strategy, risk events
    FEATURES = TopicConfig(
        name="trading-features",
        partitions=6,
        retention_ms=604800000,
        tags=["streaming", "features"],
    )
    STRATEGY = TopicConfig(
        name="trading-strategy",
        partitions=3,
        retention_ms=2592000000,
        tags=["streaming", "decisions"],
    )
    RISK = TopicConfig(
        name="trading-risk",
        partitions=3,
        retention_ms=2592000000,
        tags=["streaming", "decisions"],
    )

    # Dead letter queue
    DLQ = TopicConfig(
        name="trading-dead-letter",
        partitions=1,
        retention_ms=2592000000,
        tags=["dlq"],
    )

    # Topic lookup by event type
    EVENT_TYPE_TO_TOPIC: dict[str, TopicConfig] = {
        "MarketEvent": EVENTS,
        "Candle": FEATURES,
        "Trade": EVENTS,
        "OrderBookSnapshot": FEATURES,
        "NewsEvent": EVENTS,
        "SourceMetadata": SOURCE_METADATA,
        "FinalDecision": DECISIONS,
        "StrategyProposal": STRATEGY,
        "RiskDecision": RISK,
    }

    ALL_TOPICS: list[TopicConfig] = [
        EVENTS,
        SOURCE_METADATA,
        DECISIONS,
        FEATURES,
        STRATEGY,
        RISK,
        DLQ,
    ]

    @classmethod
    def get_topic_for_event(cls, event_type: str) -> TopicConfig:
        """Bestimmt das Topic für einen Event-Typ."""
        return cls.EVENT_TYPE_TO_TOPIC.get(event_type, cls.EVENTS)

    @classmethod
    def get_topic_name_for_event(cls, event_type: str) -> str:
        """Bestimmt den Topic-Namen für einen Event-Typ."""
        return cls.get_topic_for_event(event_type).name

    @classmethod
    def create_all_topics(cls) -> list[TopicConfig]:
        """Erstellt alle Topics (für Setup).

        Im echten System würde hier die rpk/admin-CLI verwendet werden.
        """
        return list(cls.ALL_TOPICS)

    @classmethod
    def list_topics(cls) -> list[TopicConfig]:
        return list(cls.ALL_TOPICS)
