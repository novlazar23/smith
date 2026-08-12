"""Trade Domain Models mit Volume-Profile-Berechnung.

Erweitert Einzel-Trades um Aggregation und Volume-Profile:
- TradeAggregation: Gruppierung nach Zeitfenstern
- VolumeProfile: Preis-Volumen-Verteilung mit POC, VA, VaR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TradeAggregation:
    """Aggregierte Trade-Gruppe über ein Zeitfenster.

    Enthält alle Trades eines Zeitfensters plus Aggregationsmetadaten.
    """

    instrument: str
    venue: str
    start_time: datetime
    end_time: datetime
    trade_count: int
    total_volume: float
    total_value: float
    avg_price: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    max_price: float = 0.0
    min_price: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trade_count < 0:
            raise ValueError("trade_count must be >= 0")
        if self.total_volume < 0:
            raise ValueError("total_volume must be >= 0")
        if self.max_price < self.min_price:
            raise ValueError(f"max_price ({self.max_price}) < min_price ({self.min_price})")
        if self.total_volume > 0 and self.avg_price <= 0:
            raise ValueError("avg_price must be > 0 when volume > 0")


@dataclass(frozen=True)
class VolumeProfile:
    """Preis-Volumen-Verteilung (Volume Profile).

    Berechnet POC (Point of Control), VA (Value Area), VaP (Value Area Percentage).
    """

    instrument: str
    venue: str
    start_time: datetime
    end_time: datetime
    price_levels: list[PriceLevelVolume] = field(default_factory=list)
    total_volume: float = 0.0
    poc_price: float = 0.0
    poc_volume: float = 0.0
    value_area_high: float = 0.0
    value_area_low: float = 0.0
    value_area_volume: float = 0.0
    value_area_percentage: float = 0.0

    @classmethod
    def from_trades(
        cls,
        trades: list[dict],
        instrument: str,
        venue: str,
        bucket_size: float = 0.001,
        value_area_pct: float = 0.68,
    ) -> VolumeProfile:
        """Berechnet Volume Profile aus einer Liste von Trades.

        Args:
            trades: Liste von Trade-Dicts (price, quantity)
            instrument: Instrument
            venue: Venue
            bucket_size: Preis-Bucket-Größe (relativ zum Preis)
            value_area_pct: Ziel-Value-Area-Prozent (default 68%)

        Returns:
            VolumeProfile mit berechneten Kennzahlen
        """
        if not trades:
            raise ValueError("Cannot compute VolumeProfile from empty trade list")

        prices = np.array([t["price"] for t in trades], dtype=np.float64)
        quantities = np.array([t["quantity"] for t in trades], dtype=np.float64)

        min_price = prices.min()
        max_price = prices.max()

        if bucket_size <= 0:
            bucket_size = (max_price - min_price) / 100.0

        # Preis-Buckets erstellen
        price_range = max_price - min_price
        if price_range <= 0:
            bucket_values = np.array([min_price])
            volumes = np.array([float(np.sum(quantities))])
        else:
            n_buckets = max(1, int(price_range / bucket_size) + 1)
            bucket_values = np.linspace(min_price, max_price, n_buckets)
            # Use integer bin count (not edge array) so volumes and bucket_values align
            volumes, _ = np.histogram(prices, bins=n_buckets - 1, weights=quantities)
            # Trim bucket_values to match actual bin count
            bucket_values = bucket_values[:len(volumes)]

        price_levels: list[PriceLevelVolume] = []
        for vol, price in zip(volumes, bucket_values, strict=True):
            if vol > 0:
                price_levels.append(PriceLevelVolume(price=float(price), volume=float(vol)))

        # Sortieren nach Preis
        price_levels.sort(key=lambda x: x.price)

        # POC (Point of Control) = höchstes Volumen
        max_vol_idx = int(np.argmax(volumes))
        poc_price = float(bucket_values[max_vol_idx])
        poc_volume = float(volumes[max_vol_idx])

        # Value Area: vom POC aus expandieren bis value_area_pct erreicht
        total_vol = float(np.sum(volumes))
        target_vol = total_vol * value_area_pct

        va_low, va_high = poc_price, poc_price
        current_va_vol = poc_volume

        left, right = max_vol_idx, max_vol_idx
        while current_va_vol < target_vol and (left > 0 or right < len(volumes) - 1):
            left_vol = float(volumes[left - 1]) if left > 0 else 0
            right_vol = float(volumes[right + 1]) if right < len(volumes) - 1 else 0

            if left_vol >= right_vol and left > 0:
                left -= 1
                current_va_vol += float(volumes[left])
                va_low = float(bucket_values[left])
            elif right < len(volumes) - 1:
                right += 1
                current_va_vol += float(volumes[right])
                va_high = float(bucket_values[right])
            else:
                break

        return cls(
            instrument=instrument,
            venue=venue,
            start_time=min(t.get("event_time", datetime.now()) for t in trades),
            end_time=max(t.get("event_time", datetime.now()) for t in trades),
            price_levels=price_levels,
            total_volume=total_vol,
            poc_price=poc_price,
            poc_volume=poc_volume,
            value_area_low=va_low,
            value_area_high=va_high,
            value_area_volume=current_va_vol,
            value_area_percentage=current_va_vol / total_vol if total_vol > 0 else 0.0,
        )


@dataclass(frozen=True)
class PriceLevelVolume:
    """Einzelner Preis-Level mit zugehörigem Volumen."""

    price: float
    volume: float


class TradeAggregator:
    """Gruppiert Trades nach Zeitfenstern.

    Nimmt eine sortierte Liste von Trades und erzeugt
    Zeitfenster-basierte Aggregationen.
    """

    def __init__(self, window: timedelta = timedelta(minutes=1)) -> None:
        self._window = window

    def aggregate(
        self,
        trades: list[dict],
        instrument: str,
        venue: str,
    ) -> list[TradeAggregation]:
        """Gruppiert Trades in Zeitfenster.

        Args:
            trades: Sortierte Liste von Trade-Dicts (aufsteigend nach event_time)
            instrument: Instrument
            venue: Venue

        Returns:
            Liste von TradeAggregation für jedes Fenster
        """
        if not trades:
            return []

        groups: list[list[dict]] = []
        current_group: list[dict] = []
        current_start: datetime | None = None

        for trade in trades:
            event_time = trade.get("event_time", datetime.now())
            if current_start is None:
                current_start = event_time
                current_group = [trade]
            elif event_time < current_start + self._window:
                current_group.append(trade)
            else:
                groups.append(current_group)
                current_group = [trade]
                current_start = event_time

        if current_group:
            groups.append(current_group)

        results: list[TradeAggregation] = []
        for group in groups:
            prices = np.array([t["price"] for t in group], dtype=np.float64)
            quantities = np.array([t["quantity"] for t in group], dtype=np.float64)
            bid_vol = sum(
                float(t["quantity"]) for t in group if t.get("side") == "buy"
            )
            ask_vol = sum(
                float(t["quantity"]) for t in group if t.get("side") == "sell"
            )

            results.append(
                TradeAggregation(
                    instrument=instrument,
                    venue=venue,
                    start_time=group[0].get("event_time", datetime.now()),
                    end_time=group[-1].get("event_time", datetime.now()),
                    trade_count=len(group),
                    total_volume=float(np.sum(quantities)),
                    total_value=float(np.sum(prices * quantities)),
                    avg_price=float(np.sum(prices * quantities) / np.sum(quantities))
                    if np.sum(quantities) > 0
                    else 0.0,
                    bid_volume=bid_vol,
                    ask_volume=ask_vol,
                    max_price=float(np.max(prices)),
                    min_price=float(np.min(prices)),
                )
            )

        return results
