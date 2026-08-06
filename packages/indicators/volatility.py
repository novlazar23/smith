"""Volatility Indicators — ATR (Average True Range).

Misst Marktvolatilität.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import Indicator, IndicatorResult


class ATR(Indicator):
    """Average True Range (ATR).

    Misst Volatilität als gleitender Durchschnitt des True Range.
    Standard-Periode: 14.
    """

    name = "ATR"
    min_periods = 14

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.min_periods = period

    def compute(self, data: dict[str, NDArray[np.float64]]) -> IndicatorResult:
        self._validate_data(data, ["high", "low", "close"])
        high = data["high"].astype(np.float64)
        low = data["low"].astype(np.float64)
        close = data["close"].astype(np.float64)

        if len(close) < self.period:
            raise ValueError(f"Need at least {self.period} data points, got {len(close)}")

        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # Wilder smoothing of ATR
        atr = np.full(len(high), np.nan)
        atr[self.period - 1] = np.mean(tr[:self.period - 1]) if len(tr) >= self.period else np.mean(tr)

        for i in range(self.period, len(high)):
            atr[i] = (atr[i - 1] * (self.period - 1) + tr[i - 1]) / self.period

        return IndicatorResult(
            name=self.name,
            values=atr,
            metadata={"period": self.period},
        )
