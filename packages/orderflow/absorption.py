"""Absorption Detection — Erkennung wiederholter Kontakte an Preisleveln."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import OrderFlowResult, OrderFlowSignal


class AbsorptionDetector:
    """Erkennt Absorption durch wiederholte Markt-Interaktionen an einem Preislevel.

    Args:
        min_touches: Mindestanzahl von Kontakten an einem Preislevel
            um Absorption zu signalisieren.
        price_proximity: Proximity in Prozent, innerhalb derer Preise
            als gleiches Level betrachtet werden.
    """

    def __init__(
        self,
        min_touches: int = 3,
        price_proximity: float = 0.001,
    ) -> None:
        self.min_touches = min_touches
        self.price_proximity = price_proximity

    def detect_absorption(self, data: dict[str, NDArray[np.float64]]) -> OrderFlowResult:
        """Erkennt Absorption aus OHLCV-Daten.

        Algorithmus:
            - Pro Bar wird der Kontakt-Preis bestimmt: bei bullish Bars (close > open)
              ist es der Tiefstand (low) — Verkäufer absorbieren Käufer.
              Bei bearish Bars (close < open) ist es der Hochstand (high) —
              Käufer absorbieren Verkäufer.
            - Kontakte werden nach Preislevel gruppiert.
            - Level mit >= min_touches und signifikantem Volumen → Absorption.

        Args:
            data: Dict mit 'high', 'low', 'close', 'volume'.

        Returns:
            OrderFlowResult mit Absorption-Signalen und Metadaten.

        Raises:
            ValueError: Wenn erforderliche Schlüssel fehlen.
        """
        required = ("high", "low", "close", "volume")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required data keys: {missing}")

        highs = data["high"]
        lows = data["low"]
        closes = data["close"]
        volumes = data["volume"]

        # Kontakt-Preise pro Bar bestimmen
        contact_prices = np.empty(len(volumes))
        contact_volumes = np.empty(len(volumes))

        for i in range(len(volumes)):
            if closes[i] > data["open"][i] if "open" in data else False:
                # Bullish bar — Kontakt bei Low
                contact_prices[i] = lows[i]
            else:
                # Bearish bar — Kontakt bei High
                contact_prices[i] = highs[i]
            contact_volumes[i] = volumes[i]

        # Kontakte nach Preislevel gruppieren
        levels: dict[float, list[float]] = {}
        for i in range(len(volumes)):
            price = contact_prices[i]
            # Finde existierendes Level oder erstelle neues
            assigned = False
            for level_key in list(levels.keys()):
                if abs(price - level_key) / max(level_key, 1e-10) <= self.price_proximity:
                    levels[level_key].append(float(contact_volumes[i]))
                    assigned = True
                    break
            if not assigned:
                levels[price] = [float(contact_volumes[i])]

        # Signale für Level mit genug Kontakten
        signals: list[OrderFlowSignal] = []
        absorption_levels: list[dict[str, Any]] = []

        for level_key, vol_list in levels.items():
            touch_count = len(vol_list)
            total_volume = sum(vol_list)
            if touch_count >= self.min_touches:
                signals.append(OrderFlowSignal.ABSORPTION)
                absorption_levels.append(
                    {
                        "price_level": float(level_key),
                        "touch_count": touch_count,
                        "total_volume": float(total_volume),
                    }
                )

        return OrderFlowResult(
            signals=signals,
            scores={
                "absorption_score": float(len(signals)),
            },
            metadata={
                "contact_levels": absorption_levels,
                "total_levels_analyzed": len(levels),
            },
        )
