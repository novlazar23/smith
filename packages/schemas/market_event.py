"""MarketEvent — Typisierte Marktdaten-Ereignisse.

Enthält Ohlcv-Kerzen, Einzel Trades, Orderbook-Snapshots und News-Ereignisse.
Jedes Ereignis trägt `SourceMetadata` für Point-in-Time-Validität.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventType(StrEnum):
    """Kategorien von Marktdaten-Ereignissen."""

    CANDLE = "candle"
    TRADE = "trade"
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    ORDERBOOK_DELTA = "orderbook_delta"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    LIQUIDATION = "liquidation"
    NEWS = "news"


class PriceLevel(BaseModel):
    """Ein einzelnes Price/Quantity-Paar."""

    model_config = ConfigDict(frozen=True)

    price: float = Field(gt=0)
    quantity: float = Field(gt=0)


class Candle(BaseModel):
    """Ein OHLCV-Kandelaber."""

    model_config = ConfigDict(frozen=True)

    instrument: str
    venue: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    trade_count: int | None = None
    is_closed: bool = True
    metadata: dict[str, Any] | None = None

    @model_validator(mode='after')
    def validate_candle_integrity(self) -> Candle:
        """high >= low und low > 0 muss erfüllt sein."""
        if self.low <= 0 or self.high < self.low:
            raise ValueError("Invalid candle: low > 0 and high >= low required")
        return self


class Trade(BaseModel):
    """Ein einzelner Trade."""

    model_config = ConfigDict(frozen=True)

    trade_id: str
    instrument: str
    venue: str
    price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    side: str | None = None  # "buy" or "sell" — exchange-specific
    metadata: dict[str, Any] | None = None


class OrderBookSnapshot(BaseModel):
    """Ein vollständiges Orderbook zu einem Zeitpunkt."""

    model_config = ConfigDict(frozen=True)

    instrument: str
    venue: str
    sequence: int
    bids: list[PriceLevel]
    asks: list[PriceLevel]
    metadata: dict[str, Any] | None = None


class NewsEvent(BaseModel):
    """Ein Nachrichten-Ereignis."""

    model_config = ConfigDict(frozen=True)

    news_id: str
    event_identity: str
    title: str
    body: str | None = None
    source_name: str
    source_type: str
    url_hash: str | None = None
    published_at: datetime | None = None
    received_at: datetime | None = None
    entities: list[str] | None = None
    instruments: list[str] | None = None
    language: str | None = None
    revision: int = 1


# Union über alle Markt-Ereignistypen
MarketEvent = Candle | Trade | OrderBookSnapshot | NewsEvent
