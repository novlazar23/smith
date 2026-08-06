"""Iceberg Detection — Erkennung versteckter Akkumulation/Distribution."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import OrderFlowResult, OrderFlowSignal


class IcebergDetector:
    """Erkennt versteckte Iceberg-Ordnern durch kleine Balken mit Richtungs- Bias.

    Args:
        min_subtle_threshold: Mindestanzahl kleiner Balken in einem Fenster,
            um ein potenzielles Iceberg zu signalisieren.
        lookback: Fenstergröße für die Analyse der Füllmuster.
    """

    def __init__(
        self,
        min_subtle_threshold: int = 5,
        lookback: int = 20,
    ) -> None:
        self.min_subtle_threshold = min_subtle_threshold
        self.lookback = lookback

    def detect_iceberg(self, data: dict[str, NDArray[np.float64]]) -> OrderFlowResult:
        """Erkennt Iceberg-Muster aus OHLCV-Daten.

        Algorithmus:
            - Im lookback-Fenster wird geprüft, ob viele kleine Balken existieren
              (Range < 0.5% des Preises).
            - Zusätzlich muss ein kumulatives Volumen mit Richtungsbias vorliegen.
            - Richtungs-Bias: Summe von (close - open) über kleine Balken
              geteilt durch Gesamtrange.

        Args:
            data: Dict mit 'open', 'high', 'low', 'close', 'volume'.

        Returns:
            OrderFlowResult mit Iceberg-Signalen und Metadaten.

        Raises:
            ValueError: Wenn Daten unzureichend für lookback-Fenster sind.
        """
        required = ("open", "high", "low", "close", "volume")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required data keys: {missing}")

        n = len(data["close"])
        if n < self.lookback:
            raise ValueError(
                f"Insufficient data: need at least {self.lookback} bars, "
                f"got {n}"
            )

        opens = data["open"]
        highs = data["high"]
        lows = data["low"]
        closes = data["close"]
        volumes = data["volume"]

        signals: list[OrderFlowSignal] = []
        iceberg_windows: list[dict[str, Any]] = []

        # Über alle möglichen lookback-Fenster laufen
        for start in range(n - self.lookback + 1):
            end = start + self.lookback
            window = {
                "open": opens[start:end],
                "high": highs[start:end],
                "low": lows[start:end],
                "close": closes[start:end],
                "volume": volumes[start:end],
            }
            result = self._analyze_window(window)
            if result["is_iceberg"]:
                direction = "buying" if result["direction_bias"] > 0 else "selling"
                signals.append(OrderFlowSignal.ICEBERG)
                iceberg_windows.append(
                    {
                        "start": int(start),
                        "end": int(end),
                        "direction": direction,
                        "small_bar_count": int(result["small_bar_count"]),
                        "direction_bias": float(result["direction_bias"]),
                    }
                )

        direction_summary: str | None = None
        if signals:
            buy_count = sum(
                1 for w in iceberg_windows if w["direction"] == "buying"
            )
            sell_count = len(iceberg_windows) - buy_count
            direction_summary = "buying" if buy_count >= sell_count else "selling"

        return OrderFlowResult(
            signals=signals,
            scores={
                "iceberg_count": float(len(signals)),
            },
            metadata={
                "iceberg_windows": iceberg_windows,
                "direction": direction_summary,
            },
        )

    def _analyze_window(
        self, window: dict[str, NDArray[np.float64]]
    ) -> dict[str, Any]:
        """Analysiert ein einzelnes Fenster auf Iceberg-Muster.

        Returns:
            Dict mit 'is_iceberg', 'direction_bias', 'small_bar_count'.
        """
        opens = window["open"]
        highs = window["high"]
        lows = window["low"]
        closes = window["close"]

        ranges = highs - lows
        mid_prices = (highs + lows) / 2.0

        # Kleine Balken: Range < 0.5% des Durchschnittspreises
        avg_price = float(np.mean(np.abs(mid_prices)))
        small_threshold = 0.005 * max(avg_price, 1e-10)
        is_small = ranges < small_threshold
        small_bar_count = int(np.sum(is_small))

        if small_bar_count < self.min_subtle_threshold:
            return {
                "is_iceberg": False,
                "direction_bias": 0.0,
                "small_bar_count": small_bar_count,
            }

        # Richtungs-Bias über kleine Balken berechnen
        body = closes[is_small] - opens[is_small]
        small_ranges = ranges[is_small]
        with np.errstate(divide="ignore", invalid="ignore"):
            direction_bias = float(np.sum(body) / max(float(np.sum(small_ranges)), 1e-10))

        # Richtung muss signifikant sein: |bias| > 0.3
        is_iceberg = abs(direction_bias) > 0.3

        return {
            "is_iceberg": is_iceberg,
            "direction_bias": direction_bias,
            "small_bar_count": small_bar_count,
        }
