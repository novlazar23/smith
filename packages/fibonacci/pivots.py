"""Pivot-Erkennung in OHLC-Daten."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import FibonacciPivot


class PivotDetector:
    """Erkennt Swing-High und Swing-Low Pivots in Zeitreihendaten."""

    def __init__(
        self,
        high_window: int = 10,
        low_window: int = 10,
    ) -> None:
        """Initialisiert den Pivot-Detektor.

        Args:
            high_window: Anzahl der Balken links/rechts für Swing-High-Erkennung.
            low_window: Anzahl der Balken links/rechts für Swing-Low-Erkennung.
        """
        self.high_window = high_window
        self.low_window = low_window

    def detect_pivots(
        self,
        data: dict[str, NDArray[np.float64]],
    ) -> list[FibonacciPivot]:
        """Erkennt Swing-Pivots aus Marktdaten.

        Ein Swing-High ist ein High, das höher ist als die
        high_window Balken links und rechts davon.
        Ein Swing-Low ist ein Low, das tiefer ist als die
        low_window Balken links und rechts davon.

        Args:
            data: Dict mit 'high' (required), 'low' (required),
                  und optional 'close' (default: arange).

        Returns:
            Sortierte Liste von FibonacciPivot-Objekten nach Zeit aufsteigend.

        Raises:
            ValueError: Wenn nicht genug Balken für die Fenstergrösse vorhanden sind.
        """
        high: NDArray[np.float64] = data["high"]
        low: NDArray[np.float64] = data["low"]
        close = data.get("close", np.arange(len(high), dtype=np.float64))

        max_window = max(self.high_window, self.low_window)
        if len(close) < 2 * max_window + 1:
            raise ValueError(
                f"Need at least {2 * max_window + 1} bars, got {len(close)}"
            )

        pivots: list[FibonacciPivot] = []

        # Detect swing highs
        for i in range(self.high_window, len(close) - self.high_window):
            window = high[i - self.high_window : i + self.high_window + 1]
            if high[i] == np.max(window):
                pivots.append(
                    FibonacciPivot(
                        price=float(high[i]),
                        time=i,
                        type="swing_high",
                    )
                )

        # Detect swing lows
        for i in range(self.low_window, len(close) - self.low_window):
            window = low[i - self.low_window : i + self.low_window + 1]
            if low[i] == np.min(window):
                pivots.append(
                    FibonacciPivot(
                        price=float(low[i]),
                        time=i,
                        type="swing_low",
                    )
                )

        pivots.sort(key=lambda p: p.time)
        return pivots
