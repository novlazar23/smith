"""Derivatives Domain Models.

Enthält Datenstrukturen für:
- Funding Rate: Periodische Zahlungen zwischen Long/Short Positionen
- Open Interest: Gesamte Offene Positionen
- Liquidation: Geschlossene Positionen durch Margin-Calls
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class LiquidationSide(StrEnum):
    """Seite der Liquidation."""

    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class FundingRate:
    """Funding Rate Ereignis für Perpetual Futures.

    Funding Rates werden periodisch (typ. alle 8h) zwischen
    Long- und Short-Positionen ausgetauscht. Positive Rate:
    Longs zahlen Shorts. Negative Rate: Shorts zahlen Longs.
    """

    instrument: str
    venue: str
    funding_rate: float  # Dezimal (z.B. 0.0001 = 0.01%)
    mark_price: float
    next_funding_time: datetime
    event_time: datetime
    settlement_interval: str = "8h"  # "1h", "4h", "8h"
    predicted_rate: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mark_price <= 0:
            raise ValueError("mark_price must be > 0")
        if self.funding_rate < -1.0 or self.funding_rate > 1.0:
            raise ValueError("funding_rate must be in [-1.0, 1.0]")

    @property
    def funding_rate_pct(self) -> float:
        """Funding Rate in Prozent."""
        return self.funding_rate * 100.0


@dataclass(frozen=True)
class OpenInterest:
    """Open Interest Snapshot.

    Open Interest = Gesamte Anzahl offener Kontrakte.
    Steigendes OI + steigender Preis = starker Aufwärtstrend.
    Steigendes OI + fallender Preis = starker Abwärtstrend.
    """

    instrument: str
    venue: str
    open_interest: float  # Anzahl offener Kontrakte
    event_time: datetime
    open_interest_value: float | None = None  # USD-Wert (optional)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.open_interest < 0:
            raise ValueError("open_interest must be >= 0")


@dataclass(frozen=True)
class Liquidation:
    """Liquidation Ereignis.

    Tritt auf wenn eine Position zwangsgeschlossen wird,
    weil der Margin nicht mehr ausreicht.
    """

    instrument: str
    venue: str
    side: LiquidationSide
    quantity: float
    price: float
    value: float  # USD-Wert der liquidierten Position
    event_time: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be > 0")
        if self.price <= 0:
            raise ValueError("price must be > 0")
        if self.value <= 0:
            raise ValueError("value must be > 0")


@dataclass(frozen=True)
class DerivativeSnapshot:
    """Kompletter Derivatives-Snapshot zu einem Zeitpunkt.

    Kombiniert Funding Rate, Open Interest und Liquidationen
    zu einem einzigen Snapshot.
    """

    instrument: str
    venue: str
    event_time: datetime
    funding_rate: FundingRate | None = None
    open_interest: OpenInterest | None = None
    recent_liquidations: list[Liquidation] = field(default_factory=list)

    @property
    def total_liquidation_value(self) -> float:
        """Gesamter Liquidationswert in der letzten Periode."""
        return sum(liq.value for liq in self.recent_liquidations)

    @property
    def net_liquidation_side(self) -> str | None:
        """Netto-Liquidationsseite (long/short/none).

        > 0 Long-Liquidationen (Mehrheit Short-Long = bullish Signal)
        < 0 Short-Liquidationen (Mehrheit Long-Short = bearish Signal)
        """
        if not self.recent_liquidations:
            return None

        long_val = sum(liq.value for liq in self.recent_liquidations if liq.side == LiquidationSide.LONG)
        short_val = sum(liq.value for liq in self.recent_liquidations if liq.side == LiquidationSide.SHORT)

        diff = long_val - short_val
        if diff > 0:
            return "long"
        elif diff < 0:
            return "short"
        return "balanced"
