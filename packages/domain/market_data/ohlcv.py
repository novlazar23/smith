"""OHLCV Domain Models mit Multi-Timeframe-Aggregation.

Erweitert die basale Candle-Darstellung um Aggregationslogik:
- CandleAggregation: gruppiert 1m-Kerzen in höhere Timeframes
- MultiTimeframeAggregator: berechnete Indikatoren über mehrere Timeframes
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class CandleAggregation:
    """Aggregierte Kerze über mehrere Unter-Perioden.

    Enthält die aggregierten OHLCV-Werte plus Metadaten
    zur Aggregation (Anzahl Unter-Perioden, Zeitfenster).
    """

    instrument: str
    venue: str
    timeframe: str  # "5m", "15m", "1h", "4h", "1d"
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    aggregation_source: str = "1m"  # Ursprünglicher timeframe
    sub_candle_count: int = 1

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError("OHLC values must be > 0")
        if self.volume < 0:
            raise ValueError("volume must be >= 0")
        if self.trade_count < 0:
            raise ValueError("trade_count must be >= 0")

    @classmethod
    def from_raw_candles(
        cls,
        candles: list[dict],
        target_timeframe: str,
        instrument: str,
        venue: str,
    ) -> CandleAggregation:
        """Aggregiert eine Liste von Rohkerzen zu einer aggregierten Kerze.

        Args:
            candles: Liste von Rohkerzen-Dicts (open, high, low, close, volume, etc.)
            target_timeframe: Zieltimeframe
            instrument: Instrumenten-Symbol
            venue: Handelsplatz

        Returns:
            Aggregierte Candle
        """
        if not candles:
            raise ValueError("Cannot aggregate empty candle list")

        opens = [c["open"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]
        trade_counts = [c.get("trade_count", 0) for c in candles]

        return cls(
            instrument=instrument,
            venue=venue,
            timeframe=target_timeframe,
            open_time=candles[0]["open_time"],
            close_time=candles[-1]["close_time"],
            open=opens[0],
            high=max(highs),
            low=min(lows),
            close=closes[-1],
            volume=sum(volumes),
            trade_count=sum(trade_counts),
            aggregation_source="1m",
            sub_candle_count=len(candles),
        )


# Mapping von Timeframe-Symbolen zu Datums-Offsets
TIMEFRAME_OFFSETS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}

# Zeitframe-Hierarchie für die Aggregation
TIMEFRAME_HIERARCHY: list[str] = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]


def _timeframe_delta(tf: str) -> timedelta:
    """Konvertiert Timeframe-Symbol in timedelta."""
    if tf not in TIMEFRAME_OFFSETS:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return TIMEFRAME_OFFSETS[tf]


def _group_candles_into_periods(
    candles: list[dict],
    target_tf: str,
) -> list[list[dict]]:
    """Gruppieren von Kerzen nach Target-Timeframe-Fenstern.

    Args:
        candles: Sortierte Liste von Rohkerzen-Dicts (aufsteigend nach open_time)
        target_tf: Target Timeframe

    Returns:
        Liste von Candle-Batches, jede repräsentiert eine Target-Periode
    """
    delta = _timeframe_delta(target_tf)
    groups: list[list[dict]] = []
    current_group: list[dict] = []
    current_start: datetime | None = None

    for candle in candles:
        start = candle["open_time"]
        if current_start is None:
            current_start = start
            current_group = [candle]
        elif start < current_start + delta:
            current_group.append(candle)
        else:
            groups.append(current_group)
            current_group = [candle]
            current_start = start

    if current_group:
        groups.append(current_group)

    return groups


class MultiTimeframeAggregator:
    """Berechnet aggregierte Kerzen über mehrere Timeframes.

    Nimmt 1m-Basisdaten und berechnet konsistent aggregierte Kerzen
    für 5m, 15m, 1h, 4h, 1d.
    """

    def __init__(self, base_timeframe: str = "1m") -> None:
        self._base_tf = base_timeframe
        self._cache: dict[str, CandleAggregation] = {}

    def aggregate(
        self,
        candles: list[dict],
        target_timeframe: str,
        instrument: str,
        venue: str,
    ) -> CandleAggregation:
        """Berechnet aggregierte Kerzen für einen Timeframe.

        Args:
            candles: Rohkerzen (muss >= 1 sein)
            target_timeframe: Zielttimeframe
            instrument: Instrument
            venue: Venue

        Returns:
            Aggregierte Candle

        Raises:
            ValueError: bei ungültigen Parametern
        """
        if target_timeframe == self._base_tf:
            raise ValueError("Cannot aggregate base timeframe into itself")
        if target_timeframe not in TIMEFRAME_OFFSETS:
            raise ValueError(f"Unsupported target timeframe: {target_timeframe}")

        # Cache-Key: (instrument, venue, timeframe, window_start)
        _cache_key = f"{instrument}:{venue}:{target_timeframe}"
        del _cache_key  # TODO: implement caching

        groups = _group_candles_into_periods(candles, target_timeframe)
        if not groups:
            raise ValueError("No candles to aggregate")

        # Wenn mehrere Gruppen existieren, aggregiere die letzte Gruppe
        if len(groups) == 1:
            return CandleAggregation.from_raw_candles(
                groups[0], target_timeframe, instrument, venue
            )
        else:
            # Bei mehreren Gruppen: nur die letzte (vollständigste) Gruppe zurückgeben
            last_group = groups[-1]
            return CandleAggregation.from_raw_candles(
                last_group, target_timeframe, instrument, venue
            )

    def aggregate_all(
        self,
        candles: list[dict],
        instrument: str,
        venue: str,
        target_timeframes: list[str] | None = None,
    ) -> dict[str, CandleAggregation]:
        """Aggregiert alle Ziel-Timeframes auf einmal (effizienter als einzelne Aufrufe).

        Args:
            candles: Rohkerzen (muss nach open_time sortiert sein)
            instrument: Instrument
            venue: Venue
            target_timeframes: Liste der Ziel-Timeframes (default: alle außer base)

        Returns:
            Dict mapping timeframe → CandleAggregation
        """
        if target_timeframes is None:
            target_timeframes = [
                tf for tf in TIMEFRAME_HIERARCHY
                if tf != self._base_tf and tf in TIMEFRAME_OFFSETS
            ]

        results: dict[str, CandleAggregation] = {}
        for tf in target_timeframes:
            try:
                results[tf] = self.aggregate(candles, tf, instrument, venue)
            except ValueError:
                # Ungültiges timeframe → überspringen
                continue

        return results

    def compute_simple_moving_average(
        self,
        candles: list[dict],
        price_key: str = "close",
        window: int = 20,
    ) -> np.ndarray:
        """Berechnet den Simple Moving Average über eine Preis-Reihe.

        Args:
            candles: Rohkerzen (muss nach open_time sortiert sein)
            price_key: Welcher Preis (close, open, high, low)
            window: SMA-Perioden

        Returns:
            numpy Array mit SMA-Werten (len(candles) - window + 1)
        """
        if len(candles) < window:
            raise ValueError(f"Need at least {window} candles for SMA({window})")

        prices = np.array([c[price_key] for c in candles], dtype=np.float64)
        cumsum = np.cumsum(prices)
        sma = np.empty_like(prices, dtype=np.float64)
        sma[:window - 1] = np.nan
        sma[window - 1 :] = (cumsum[window - 1 :] - np.append(0, cumsum[:-window])) / window
        return sma

    def compute_exponential_moving_average(
        self,
        candles: list[dict],
        price_key: str = "close",
        span: int = 20,
    ) -> np.ndarray:
        """Berechnet den Exponential Moving Average.

        Args:
            candles: Rohkerzen
            price_key: Welcher Preis
            span: EMA-Perioden

        Returns:
            numpy Array mit EMA-Werten
        """
        if len(candles) < span:
            raise ValueError(f"Need at least {span} candles for EMA({span})")

        prices = np.array([c[price_key] for c in candles], dtype=np.float64)
        alpha = 2.0 / (span + 1)
        ema = np.zeros_like(prices, dtype=np.float64)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema
