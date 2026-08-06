"""Trend Indicators — SMA, EMA, Bollinger Bands, ADX.

Berechnung basiert auf numpy für Performanz.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import Indicator, IndicatorResult


class SMA(Indicator):
    """Simple Moving Average.

    Gleitender Durchschnitt über N Perioden.
    """

    name = "SMA"

    def __init__(self, period: int = 20) -> None:
        self.period = period
        self.min_periods = period

    def compute(self, data: dict[str, NDArray[np.float64]]) -> IndicatorResult:
        self._validate_data(data, ["close"])
        close = data["close"].astype(np.float64)

        if len(close) < self.period:
            raise ValueError(f"Need at least {self.period} data points, got {len(close)}")

        sma = np.full(len(close), np.nan)
        window_sum = np.sum(close[:self.period])
        sma[self.period - 1] = window_sum / self.period

        for i in range(self.period, len(close)):
            window_sum += close[i] - close[i - self.period]
            sma[i] = window_sum / self.period

        return IndicatorResult(
            name=self.name,
            values=sma,
            metadata={"period": self.period},
        )


class EMA(Indicator):
    """Exponential Moving Average.

    Gewichtet aktuelle Preise stärker.
    """

    name = "EMA"

    def __init__(self, period: int = 20) -> None:
        self.period = period
        self.min_periods = period

    def compute(self, data: dict[str, NDArray[np.float64]]) -> IndicatorResult:
        self._validate_data(data, ["close"])
        close = data["close"].astype(np.float64)

        if len(close) < self.period:
            raise ValueError(f"Need at least {self.period} data points, got {len(close)}")

        multiplier = 2.0 / (self.period + 1)
        ema = np.full(len(close), np.nan)
        ema[self.period - 1] = np.mean(close[:self.period])

        for i in range(self.period, len(close)):
            ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return IndicatorResult(
            name=self.name,
            values=ema,
            metadata={"period": self.period},
        )


class BollingerBands(Indicator):
    """Bollinger Bands.

    Mittlere Linie (SMA) +/- 2 Standardabweichungen.
    Standard-Periode: 20, std_dev: 2.0.
    """

    name = "BollingerBands"
    min_periods = 20

    def __init__(self, period: int = 20, std_dev: float = 2.0) -> None:
        self.period = period
        self.std_dev = std_dev
        self.min_periods = period

    def compute(self, data: dict[str, NDArray[np.float64]]) -> IndicatorResult:
        self._validate_data(data, ["close"])
        close = data["close"].astype(np.float64)

        if len(close) < self.period:
            raise ValueError(f"Need at least {self.period} data points, got {len(close)}")

        sma = np.full(len(close), np.nan)
        upper = np.full(len(close), np.nan)
        lower = np.full(len(close), np.nan)

        for i in range(self.period - 1, len(close)):
            window = close[i - self.period + 1: i + 1]
            sma[i] = np.mean(window)
            std = np.std(window, ddof=0)
            upper[i] = sma[i] + self.std_dev * std
            lower[i] = sma[i] - self.std_dev * std

        # Combine into single array: [upper, middle, lower]
        result = np.stack([upper, sma, lower], axis=1)

        return IndicatorResult(
            name=self.name,
            values=result,
            metadata={
                "period": self.period,
                "std_dev": self.std_dev,
            },
        )


class ADX(Indicator):
    """Average Directional Index (ADX).

    Misst Trendstärke (nicht Richtung).
    Standard: periode=14.
    """

    name = "ADX"
    min_periods = 14 + 1

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.min_periods = period + 1

    def compute(self, data: dict[str, NDArray[np.float64]]) -> IndicatorResult:
        self._validate_data(data, ["high", "low", "close"])
        high = data["high"].astype(np.float64)
        low = data["low"].astype(np.float64)
        close = data["close"].astype(np.float64)

        if len(close) < self.min_periods:
            raise ValueError(f"Need at least {self.min_periods} data points, got {len(close)}")

        # True Range
        tr1 = np.diff(high)
        tr2 = np.diff(low)
        tr3 = np.abs(np.diff(close))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # Directional Movement
        up_move = np.diff(high)
        down_move = -np.diff(low)

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smoothed averages
        atr = np.zeros_like(high)
        dmi = np.zeros_like(high)

        # Wilder smoothing
        atr[self.period] = np.sum(tr[:self.period])
        dmi[self.period] = self._calculate_adx(
            atr[self.period],
            self._smooth(plus_dm[:self.period], self.period),
            self._smooth(minus_dm[:self.period], self.period),
        )

        for i in range(self.period + 1, len(high)):
            atr[i] = (atr[i - 1] * (self.period - 1) + tr[i - 1]) / self.period
            dmi[i] = self._calculate_adx(
                atr[i],
                self._smooth(plus_dm[i - self.period + 1: i + 1], self.period),
                self._smooth(minus_dm[i - self.period + 1: i + 1], self.period),
            )

        return IndicatorResult(
            name=self.name,
            values=dmi,
            metadata={"period": self.period},
        )

    @staticmethod
    def _smooth(data: NDArray[np.float64], period: int) -> float:
        """Wilder Summe."""
        return np.sum(data)

    @staticmethod
    def _calculate_adx(atr: float, plus_di: float, minus_di: float) -> float:
        """Berechnet ADX-Wert aus einem einzelnen Schritt."""
        if atr == 0:
            return 0.0
        dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0.0
        return dx
