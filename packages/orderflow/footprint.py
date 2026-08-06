"""Footprint-Analyse — Delta-Integration und Imbalance-Erkennung."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import OrderFlowResult, OrderFlowSignal


class FootprintAnalyzer:
    """Analysiert Footprint-Daten für Delta und Imbalances.

    Args:
        imbalance_threshold: Verhältnis von Bid/Ask-Volumen an einem
            Preislevel, um eine Imbalance zu markieren.
        tick_size: Mindestes Preis-Inkrement.
    """

    def __init__(
        self,
        imbalance_threshold: float = 3.0,
        tick_size: float = 0.01,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.tick_size = tick_size

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> OrderFlowResult:
        """Analysiert Footprint-Daten über alle Balken hinweg.

        Args:
            data: Dict mit den Schlüsseln 'open', 'high', 'low', 'close',
                'volume' — alle als NDArray[np.float64].

        Returns:
            OrderFlowResult mit cumulative_delta und erkannten Signalen.

        Raises:
            ValueError: Wenn erforderliche Schlüssel fehlen oder Arrays
                unterschiedliche Längen haben.
        """
        required = ("open", "high", "low", "close", "volume")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required data keys: {missing}")

        lengths = [len(v) for v in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent array lengths: {dict(zip(required, lengths, strict=True))}")

        opens = data["open"]
        highs = data["high"]
        lows = data["low"]
        closes = data["close"]
        volumes = data["volume"]

        # Delta pro Bar: (close - open) / (high - low) * volume
        ranges = highs - lows
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_per_bar = np.where(
                ranges != 0,
                (closes - opens) / ranges * volumes,
                0.0,
            )

        cumulative_delta = float(np.sum(delta_per_bar))

        # Imbalances erkennen
        imbalances = self.compute_imbalance(data)
        has_imbalance = len(imbalances) > 0

        # Signale basierend auf Delta und Imbalances
        signals: list[OrderFlowSignal] = []
        if cumulative_delta > 0:
            signals.append(OrderFlowSignal.AGGRESSIVE_BUY)
        elif cumulative_delta < 0:
            signals.append(OrderFlowSignal.AGGRESSIVE_SELL)

        if has_imbalance:
            signals.append(OrderFlowSignal.IMBALANCE)

        return OrderFlowResult(
            signals=signals,
            scores={
                "cumulative_delta": cumulative_delta,
                "imbalance_count": float(len(imbalances)),
            },
            metadata={
                "delta_per_bar": delta_per_bar.tolist(),
                "imbalances": imbalances,
            },
            cumulative_delta=cumulative_delta,
        )

    def compute_imbalance(self, data: dict[str, NDArray[np.float64]]) -> list[dict[str, Any]]:
        """Prüft für jeden Balken, ob das implizierte Bid/Ask-Verhältnis
        den Schwellenwert überschreitet.

        Args:
            data: Dict mit 'open', 'high', 'low', 'close', 'volume'.

        Returns:
            Liste von Dicts mit 'price_level', 'bid_volume', 'ask_volume', 'ratio'.
        """
        opens = data["open"]
        highs = data["high"]
        lows = data["low"]
        closes = data["close"]
        volumes = data["volume"]

        ranges = highs - lows
        bodies = closes - opens
        with np.errstate(divide="ignore", invalid="ignore"):
            body_ratio = np.where(ranges != 0, bodies / ranges, 0.0)

        results: list[dict[str, Any]] = []
        for i in range(len(volumes)):
            br = float(body_ratio[i])
            vol = float(volumes[i])

            # Bid/Ask-Volume aus Body-to-Range-Verhältnis ableiten
            bid_vol = (1.0 + br) / 2.0 * vol
            ask_vol = (1.0 - br) / 2.0 * vol

            # Verhältnis: größeres durch kleineres
            # Bei vol=0 beide Volumes = 0 → kein Imbalance
            if bid_vol < 1e-10 and ask_vol < 1e-10:
                continue
            max_vol = max(bid_vol, ask_vol)
            min_vol = min(bid_vol, ask_vol, 1e-10)
            effective_ratio = max_vol / min_vol

            if effective_ratio >= self.imbalance_threshold:
                mid_price = (opens[i] + closes[i]) / 2.0
                results.append(
                    {
                        "price_level": float(mid_price),
                        "bid_volume": float(bid_vol),
                        "ask_volume": float(ask_vol),
                        "ratio": float(effective_ratio),
                    }
                )

        return results
