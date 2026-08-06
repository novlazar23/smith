"""Support-/Resistance-Level-Erkennung — Preis-Clustering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import SupportResistanceLevel


class SupportResistanceDetector:
    """Erkennt Support- und Resistance-Level durch Clustering von Preisdaten."""

    def __init__(
        self,
        price_proximity: float = 0.002,
        min_touches: int = 2,
    ) -> None:
        """Initialisiert den Level-Detektor.

        Args:
            price_proximity: Proximität-Schwelle als Prozentsatz (0.002 = 0.2%).
            min_touches: Minimale Anzahl von Berührungen für ein Level.
        """
        if price_proximity <= 0:
            raise ValueError(f"price_proximity muss > 0 sein, erhalten: {price_proximity}")
        if min_touches < 1:
            raise ValueError(f"min_touches muss >= 1 sein, erhalten: {min_touches}")
        self.price_proximity = price_proximity
        self.min_touches = min_touches

    def detect_levels(
        self,
        data: dict[str, NDArray[np.float64]],
    ) -> list[SupportResistanceLevel]:
        """Erkennt Support- und Resistance-Level aus Preisdaten.

        Cluster nearby prices using price_proximity as percentage threshold,
        then determine level type and strength.

        Args:
            data: Dict mit 'close' (required).

        Returns:
            Liste von SupportResistanceLevel-Objekten.
        """
        if "close" not in data:
            raise ValueError("Missing required data keys: ['close']")

        close = data["close"]
        if len(close) == 0:
            return []

        # Cluster nearby prices using price_proximity as percentage threshold
        clusters = self._cluster_prices(close)

        levels: list[SupportResistanceLevel] = []
        median_price = float(np.median(close))

        for cluster in clusters:
            touch_count = len(cluster)
            if touch_count < self.min_touches:
                continue

            center = float(np.mean(cluster))
            strength = float(min(touch_count / (touch_count + 1), 1.0))
            level_type = "resistance" if center > median_price else "support"

            levels.append(
                SupportResistanceLevel(
                    price=center,
                    level_type=level_type,
                    strength=strength,
                    touch_count=touch_count,
                )
            )

        return levels

    @staticmethod
    def _cluster_prices(close: NDArray[np.float64]) -> list[list[float]]:
        """Cluster nearby prices.

        Sort unique prices, group nearby prices into clusters where
        consecutive prices differ by less than the proximity threshold.

        Args:
            close: Array of close prices.

        Returns:
            Liste von Clustern, jedes eine Liste von Preiswerten.
        """
        if len(close) == 0:
            return []

        sorted_prices = np.sort(np.unique(close))

        if len(sorted_prices) == 0:
            return [[float(close[0])]]

        clusters: list[list[float]] = []
        current_cluster: list[float] = [float(sorted_prices[0])]

        for i in range(1, len(sorted_prices)):
            prev_price = sorted_prices[i - 1]
            curr_price = sorted_prices[i]

            threshold = float(abs(prev_price) * 0.002)
            if threshold == 0:
                threshold = 0.01

            if abs(float(curr_price - prev_price)) <= threshold:
                current_cluster.append(float(curr_price))
            else:
                clusters.append(current_cluster)
                current_cluster = [float(curr_price)]

        clusters.append(current_cluster)
        return clusters
