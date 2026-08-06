"""Schemas — Event-Schemas für Serialization/Deserialization.

Definiert die strukturierten Event-Typen aus dem Ereignismodell:
  - SourceMetadata: Metadaten zu Datenquellen und Events
  - MarketEvent: Generischer Markt-Event-Wrapper
  - Candle: OHLCV-Kerzen-Daten
  - Trade: Einzel-Trades
  - OrderBookSnapshot: Orderbook-Snapshot
  - NewsEvent: Nachrichten-Events
"""

from __future__ import annotations

import uuid as _uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timezone
from enum import StrEnum
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(UTC)


# ── SourceMetadata ──────────────────────────────────────────────────


@dataclass(frozen=True)
class SourceMetadata:
    """Metadaten zu Datenquellen und Events (Spec §8).

    Felder: source, venue, event_time, ingestion_time,
            availability_time, sequence, revision, quality (0.0-1.0).
    """

    source: str
    venue: str
    event_time: datetime
    ingestion_time: datetime = field(default_factory=_utc_now)
    availability_time: datetime = field(default_factory=_utc_now)
    sequence: int = 0
    revision: int = 1
    quality: float = 1.0

    def __post_init__(self) -> None:
        if self.quality < 0.0 or self.quality > 1.0:
            raise ValueError(f"quality must be in [0.0, 1.0], got {self.quality}")
        if self.sequence < 0:
            raise ValueError(f"sequence must be >= 0, got {self.sequence}")
        if self.revision < 1:
            raise ValueError(f"revision must be >= 1, got {self.revision}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_time"] = self.event_time.isoformat()
        d["ingestion_time"] = self.ingestion_time.isoformat()
        d["availability_time"] = self.availability_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceMetadata:
        dt_keys = ("event_time", "ingestion_time", "availability_time")
        for k in dt_keys:
            if isinstance(data.get(k), str):
                data[k] = datetime.fromisoformat(data[k])
        return cls(**data)


# ── MarketEvent ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarketEvent:
    """Generischer Markt-Event-Wrapper (Spec §8).

    Felder: event_id, event_type, instrument, metadata, payload.
    """

    event_id: str
    event_type: str  # "candle", "trade", "orderbook_snapshot"
    instrument: str
    metadata: SourceMetadata
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metadata"] = self.metadata.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarketEvent:
        d = dict(data)
        meta = d.get("metadata")
        if isinstance(meta, dict):
            d["metadata"] = SourceMetadata.from_dict(meta)
        elif isinstance(meta, SourceMetadata):
            d["metadata"] = meta
        elif "metadata" not in d:
            d["metadata"] = SourceMetadata(
                source=d.get("instrument", "unknown"),
                venue=d.get("instrument", "unknown"),
                event_time=datetime.now(UTC),
            )
        d.setdefault("event_id", str(_uuid.uuid4()))
        d.setdefault("event_type", d.get("payload", {}).get("event_type", "generic"))
        d.setdefault("instrument", "unknown")
        return cls(**d)


# ── Candle ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Candle:
    """OHLCV-Kerze.

    Felder: instrument, venue, timeframe, open_time, close_time,
            open, high, low, close, volume, trade_count, is_closed.
    """

    instrument: str
    venue: str
    timeframe: str  # "1m", "5m", "15m", "1h", "4h", "1d"
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = 0
    is_closed: bool = True

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low}) for candle")
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError("OHLC values must be > 0 for candle")
        if self.volume < 0:
            raise ValueError("volume must be >= 0 for candle")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["open_time"] = self.open_time.isoformat()
        d["close_time"] = self.close_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candle:
        d = dict(data)
        d["open_time"] = datetime.fromisoformat(d["open_time"])
        d["close_time"] = datetime.fromisoformat(d["close_time"])
        return cls(**d)


# ── Trade ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Trade:
    """Einzel-Trade.

    Felder: trade_id, instrument, venue, price, quantity, side, event_time.
    """

    trade_id: str
    instrument: str
    venue: str
    price: float
    quantity: float
    side: str  # "buy", "sell"
    event_time: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"trade price must be > 0, got {self.price}")
        if self.quantity <= 0:
            raise ValueError(f"trade quantity must be > 0, got {self.quantity}")
        if self.side not in ("buy", "sell"):
            raise ValueError(f"trade side must be 'buy' or 'sell', got '{self.side}'")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_time"] = self.event_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trade:
        d = dict(data)
        d["event_time"] = datetime.fromisoformat(d["event_time"])
        return cls(**d)


# ── OrderBookSnapshot ───────────────────────────────────────────────


@dataclass(frozen=True)
class OrderBookSnapshot:
    """Orderbook-Snapshot.

    Felder: instrument, venue, sequence, bids, asks, metadata.
    bids/asks: list of [price, quantity].
    """

    instrument: str
    venue: str
    sequence: int
    bids: list[list[float]] = field(default_factory=list)
    asks: list[list[float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError(f"sequence must be >= 0, got {self.sequence}")
        for side_name, side in [("bid", self.bids), ("ask", self.asks)]:
            for entry in side:
                if len(entry) != 2 or entry[0] <= 0 or entry[1] < 0:
                    raise ValueError(
                        f"Invalid {side_name} entry {entry}: price > 0, quantity >= 0 required"
                    )
        if self.bids and self.asks and self.bids[0][0] >= self.asks[0][0]:
            raise ValueError(f"Best bid ({self.bids[0][0]}) >= best ask ({self.asks[0][0]})")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrderBookSnapshot:
        return cls(**data)


# ── NewsEvent ───────────────────────────────────────────────────────


class NewsStatus(StrEnum):
    """News-Event-Status (Spec §13 News-Agent)."""

    RUMOR = "RUMOR"
    INITIAL = "INITIAL"
    CONFIRMATION = "CONFIRMATION"
    UPDATE = "UPDATE"
    CORRECTION = "CORRECTION"
    RETRACTION = "RETRACTION"


@dataclass(frozen=True)
class NewsEvent:
    """Nachrichten-Event (Spec §8).

    Felder: news_id, event_identity, title, body, source_name,
            source_type, url_hash, published_at, received_at,
            entities, instruments, language, revision.
    """

    news_id: str
    event_identity: str
    title: str
    body: str
    source_name: str
    source_type: str
    url_hash: str
    published_at: datetime
    received_at: datetime = field(default_factory=_utc_now)
    entities: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    language: str = "en"
    revision: int = 1
    status: NewsStatus = NewsStatus.INITIAL

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["published_at"] = self.published_at.isoformat()
        d["received_at"] = self.received_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NewsEvent:
        d = dict(data)
        d["published_at"] = datetime.fromisoformat(d["published_at"])
        d["received_at"] = datetime.fromisoformat(d["received_at"])
        return cls(**d)
