"""Swing-Pivot-Erkennung — lokaler Extremwert-Filter."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import SwingPivot


class SwingDetector:
    """Erkennt Swing-Highs und Swing-Lows in Preisdaten.

    Ein Swing-Pivot ist ein lokaler Extremwert innerhalb eines Lookback-Fensters.
    """

    def __init__(self, lookback: int = 5) -> None:
        """Initialisiert den Swing-Detektor.

        Args:
            lookback: Anzahl der Bars links und rechts eines Pivots zur Bestätigung.
        """
        if lookback < 1:
            raise ValueError(f"lookback muss >= 1 sein, erhalten: {lookback}")
        self.lookback = lookback

    def detect_swings(
        self,
        data: dict[str, NDArray[np.float64]],
    ) -> list[SwingPivot]:
        """Erkennt Swing-Pivots aus Marktdaten.

        Ein Bar i ist ein Swing-High, wenn high[i] >= max(high[i-lookback:i+lookback+1]).
        Ein Bar i ist ein Swing-Low, wenn low[i] <= min(low[i-lookback:i+lookback+1]).

        Args:
            data: Dict mit 'high' (required), 'low' (required), 'close' (optional).

        Returns:
            Liste von SwingPivot-Objekten.

        Raises:
            ValueError: Wenn keine oder unzureichende Daten vorhanden sind.
        """
        required_keys = ("high", "low")
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"Missing required data keys: {missing}")

        highs = data["high"]
        lows = data["low"]
        close = data.get("close", highs.copy())
        n = len(close)
        min_len = 2 * self.lookback + 1
        if n < min_len:
            raise ValueError(
                f"Ungenügende Daten: {n} Bars erforderlich für lookback={self.lookback}, "
                f"mindestens {min_len} erforderlich"
            )

        pivots: list[SwingPivot] = []

        for i in range(self.lookback, n - self.lookback):
            window_high = highs[i - self.lookback : i + self.lookback + 1]
            window_low = lows[i - self.lookback : i + self.lookback + 1]

            # Swing High
            if highs[i] >= window_high.max():
                quality = self._quality_score_high(i, highs, lows, window_high, window_low)
                pivots.append(
                    SwingPivot(
                        price=float(highs[i]),
                        time=i,
                        direction="high",
                        quality_score=quality,
                    )
                )

            # Swing Low
            if lows[i] <= window_low.min():
                quality = self._quality_score_low(i, highs, lows, window_high, window_low)
                pivots.append(
                    SwingPivot(
                        price=float(lows[i]),
                        time=i,
                        direction="low",
                        quality_score=quality,
                    )
                )

        return pivots

    @staticmethod
    def _quality_score_high(
        i: int,
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        window_high: NDArray[np.float64],
        window_low: NDArray[np.float64],
    ) -> float:
        """Berechnet die Qualität eines Swing-High.

        quality = (swing_value - max(neighboring_bars)) / swing_value
        """
        lookback = len(window_high) // 2
        lb_start = max(0, i - lookback)

        # Max of adjacent bars before and after the pivot
        neighbor_high = np.concatenate(
            [highs[lb_start:i], highs[i + 1 : min(i + lookback + 1, len(highs))]]
        )
        if len(neighbor_high) == 0:
            return 0.0

        nearest_barrier = float(neighbor_high.max())
        swing_value = float(highs[i])

        if swing_value == 0:
            return 0.0
        return float(abs(swing_value - nearest_barrier) / swing_value)

    @staticmethod
    def _quality_score_low(
        i: int,
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        window_high: NDArray[np.float64],
        window_low: NDArray[np.float64],
    ) -> float:
        """Berechnet die Qualität eines Swing-Low.

        quality = (min(neighboring_bars) - swing_value) / swing_value
        """
        lookback = len(window_low) // 2
        lb_start = max(0, i - lookback)

        neighbor_low = np.concatenate(
            [lows[lb_start:i], lows[i + 1 : min(i + lookback + 1, len(lows))]]
        )
        if len(neighbor_low) == 0:
            return 0.0

        nearest_barrier = float(neighbor_low.min())
        swing_value = float(lows[i])

        if swing_value == 0:
            return 0.0
        return float(abs(nearest_barrier - swing_value) / swing_value)
