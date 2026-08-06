"""Momentum Indicators — RSI, MACD, Stochastic Oscillator.

Berechnung basiert auf numpy für Performanz.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import Indicator, IndicatorResult


class RSI(Indicator):
    """Relative Strength Index (RSI).

    Misst Überkauft/Überverkauft-Bedingungen.
    Standard-Periode: 14.
    """

    name = "RSI"
    min_periods = 14

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.min_periods = period

    def compute(self, data: dict[str, NDArray[np.float64]]) -> IndicatorResult:
        self._validate_data(data, ["close"])
        close = data["close"].astype(np.float64)

        if len(close) < self.period:
            raise ValueError(f"Need at least {self.period} data points, got {len(close)}")

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Wilder smoothing
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])

        rsi_values = np.full(len(close), np.nan)
        rsi_values[self.period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss)) if avg_loss != 0 else 100.0

        for i in range(self.period + 1, len(close)):
            avg_gain = (avg_gain * (self.period - 1) + gains[i - 1]) / self.period
            avg_loss = (avg_loss * (self.period - 1) + losses[i - 1]) / self.period
            if avg_loss == 0:
                rsi_values[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

        return IndicatorResult(
            name=self.name,
            values=rsi_values,
            metadata={"period": self.period, "overbought": 70, "oversold": 30},
        )


class MACD(Indicator):
    """Moving Average Convergence Divergence (MACD).

    Linien: MACD line, Signal line, Histogram.
    Standard: fast=12, slow=26, signal=9.
    """

    name = "MACD"
    min_periods = 26 + 9

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.min_periods = slow_period + signal_period

    def compute(self, data: dict[str, NDArray[np.float64]]) -> IndicatorResult:
        self._validate_data(data, ["close"])
        close = data["close"].astype(np.float64)

        if len(close) < self.min_periods:
            raise ValueError(f"Need at least {self.min_periods} data points, got {len(close)}")

        # EMA calculations
        macd_line = self._ema(close, self.fast_period) - self._ema(close, self.slow_period)
        # Pad macd_line with NaNs for alignment
        full_macd = np.full(len(close), np.nan)
        full_macd[-len(macd_line):] = macd_line
        signal_line = self._ema(full_macd, self.signal_period)
        histogram = full_macd - signal_line

        # Strip leading NaNs for cleaner output
        valid_mask = ~np.isnan(histogram)
        if not np.any(valid_mask):
            raise ValueError("No valid MACD values computed")

        return IndicatorResult(
            name=self.name,
            values=histogram,
            metadata={
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "signal_period": self.signal_period,
            },
        )

    @staticmethod
    def _ema(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
        """Exponential Moving Average."""
        multiplier = 2.0 / (period + 1)
        ema = np.full_like(data, np.nan)
        ema[period - 1] = np.nanmean(data[:period])

        for i in range(period, len(data)):
            if np.isnan(data[i]):
                ema[i] = ema[i - 1] if not np.isnan(ema[i - 1]) else np.nan
            else:
                prev = ema[i - 1]
                if not np.isnan(prev):
                    ema[i] = (data[i] - prev) * multiplier + prev
                else:
                    # Seed EMA when previous value is NaN (e.g. leading gaps)
                    ema[i] = data[i]

        return ema


class StochasticOscillator(Indicator):
    """Stochastic Oscillator (%K und %D).

    Misst aktuelle Preise im Verhältnis zum Hoch-Tief-Bereich.
    Standard: %K periode=14, %D periode=3.
    """

    name = "Stochastic"
    min_periods = 14 + 3

    def __init__(self, k_period: int = 14, d_period: int = 3) -> None:
        self.k_period = k_period
        self.d_period = d_period
        self.min_periods = k_period + d_period

    def compute(self, data: dict[str, NDArray[np.float64]]) -> IndicatorResult:
        self._validate_data(data, ["high", "low", "close"])
        high = data["high"].astype(np.float64)
        low = data["low"].astype(np.float64)
        close = data["close"].astype(np.float64)

        if len(close) < self.min_periods:
            raise ValueError(f"Need at least {self.min_periods} data points, got {len(close)}")

        k_values = np.full(len(close), np.nan)

        for i in range(self.k_period - 1, len(close)):
            nlo = np.min(low[i - self.k_period + 1: i + 1])
            nhi = np.max(high[i - self.k_period + 1: i + 1])
            if nhi != nlo:
                k_values[i] = 100.0 * (close[i] - nlo) / (nhi - nlo)
            else:
                k_values[i] = 50.0

        d_values = np.full(len(close), np.nan)
        for i in range(self.k_period - 1 + self.d_period - 1, len(close)):
            d_values[i] = np.mean(k_values[i - self.d_period + 1: i + 1])

        return IndicatorResult(
            name=self.name,
            values=k_values,
            metadata={
                "k_period": self.k_period,
                "d_period": self.d_period,
                "overbought": 80,
                "oversold": 20,
            },
        )
