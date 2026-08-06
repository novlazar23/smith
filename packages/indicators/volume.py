"""Volume Indicators — OBV, VWAP.

On-Balance Volume und Volume-Weighted Average Price.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import Indicator, IndicatorResult


class OBV(Indicator):
    """On-Balance Volume (OBV).

    Kumulative Volumen-Index basierend auf Preisrichtung.
    """

    name = "OBV"

    def compute(self, data: dict[str, NDArray[np.float64]]) -> IndicatorResult:
        self._validate_data(data, ["close", "volume"])
        close = data["close"].astype(np.float64)
        volume = data["volume"].astype(np.float64)

        obv = np.zeros(len(close))
        obv[0] = volume[0] if volume[0] > 0 else 0.0

        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        return IndicatorResult(
            name=self.name,
            values=obv,
            metadata={},
        )


class VWAP(Indicator):
    """Volume-Weighted Average Price (VWAP).

    Durchschnittspreis gewichtet mit Volumen.
    """

    name = "VWAP"

    def compute(self, data: dict[str, NDArray[np.float64]]) -> IndicatorResult:
        self._validate_data(data, ["high", "low", "close", "volume"])
        high = data["high"].astype(np.float64)
        low = data["low"].astype(np.float64)
        close = data["close"].astype(np.float64)
        volume = data["volume"].astype(np.float64)

        typical_price = (high + low + close) / 3.0
        cumulative_tp_vol = np.cumsum(typical_price * volume)
        cumulative_vol = np.cumsum(volume)

        vwap = np.full(len(close), np.nan)
        valid = cumulative_vol > 0
        vwap[valid] = cumulative_tp_vol[valid] / cumulative_vol[valid]

        return IndicatorResult(
            name=self.name,
            values=vwap,
            metadata={},
        )
