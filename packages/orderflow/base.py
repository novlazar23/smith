"""Basis-Klassen und Typen für Order Flow Analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Side(StrEnum):
    """Order-Flow Richtung."""

    BID = "bid"
    ASK = "ask"


class OrderFlowSignal(StrEnum):
    """Mögliche Order-Flow-Signale."""

    AGGRESSIVE_BUY = "aggressive_buy"
    AGGRESSIVE_SELL = "aggressive_sell"
    ABSORPTION = "absorption"
    ICEBERG = "iceberg"
    IMBALANCE = "imbalance"
    NONE = "none"


@dataclass
class OrderBookSnapshot:
    """Momentaufnahme des Orderbuchs zu einem Zeitpunkt."""

    timestamp: int
    bids: list[tuple[float, int]] = field(default_factory=list)
    asks: list[tuple[float, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bids = sorted(self.bids, key=lambda x: x[0], reverse=True)
        self.asks = sorted(self.asks, key=lambda x: x[0])


@dataclass
class OrderFlowResult:
    """Ergebnis einer Order-Flow-Analyse."""

    signals: list[OrderFlowSignal] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    cumulative_delta: float = 0.0
