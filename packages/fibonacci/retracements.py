"""Fibonacci-Retracement- und Extension-Berechnung aus Pivots."""

from __future__ import annotations

from .base import (
    FIBONACCI_EXTENSIONS,
    FIBONACCI_RETRACEMENTS,
    FibonacciArea,
    FibonacciPivot,
)


class FibonacciRetracement:
    """Berechnet Fibonacci-Retracement- und Extension-Zonen aus Pivot-Paaren."""

    def __init__(self, zone_band: float = 0.005) -> None:
        """Initialisiert den Retracement-Berechner.

        Args:
            zone_band: Halber Zonen-Band als Prozentsatz um jedes Level.
        """
        if zone_band <= 0:
            raise ValueError(f"zone_band muss > 0 sein, erhalten: {zone_band}")
        self.zone_band = zone_band

    def calculate_retracements(
        self,
        pivots: list[FibonacciPivot],
    ) -> list[FibonacciArea]:
        """Berechnet Retracement- und Extension-Zonen aus Pivot-Paaren.

        Paart aufeinanderfolgende Pivots und berechnet Fibonacci-Level
        für jeden Leg.

        Args:
            pivots: Liste von FibonacciPivot-Objekten (nicht sortiert erforderlich).

        Returns:
            Liste von FibonacciArea-Objekten, sortiert nach Preis aufsteigend.
        """
        if len(pivots) < 2:
            return []

        sorted_pivots = sorted(pivots, key=lambda p: p.time)

        areas: list[FibonacciArea] = []

        # Pair consecutive pivots: swing_high→swing_low (downtrend) or
        # swing_low→swing_high (uptrend)
        i = 0
        while i < len(sorted_pivots) - 1:
            pivot_a = sorted_pivots[i]
            pivot_b = sorted_pivots[i + 1]

            if pivot_a.type == "swing_high" and pivot_b.type == "swing_low":
                # Downtrend leg: retracement from high down, extension below low
                areas.extend(
                    self._compute_downtrend(pivot_a.price, pivot_b.price)
                )
            elif pivot_a.type == "swing_low" and pivot_b.type == "swing_high":
                # Uptrend leg: retracement from low up, extension above high
                areas.extend(
                    self._compute_uptrend(pivot_a.price, pivot_b.price)
                )

            i += 1

        areas.sort(key=lambda a: a.lower)
        return areas

    def _make_zone(
        self, price: float, level_types: list[str]
    ) -> FibonacciArea:
        """Erstellt eine Fibonacci-Zone um einen Preis."""
        lower = price * (1 - self.zone_band)
        upper = price * (1 + self.zone_band)
        return FibonacciArea(lower=lower, upper=upper, level_types=level_types)

    def _compute_downtrend(
        self, high_price: float, low_price: float
    ) -> list[FibonacciArea]:
        """Berechnet Level für ein Downtrend-Leg (high→low)."""
        range_ = high_price - low_price
        areas: list[FibonacciArea] = []

        # Retracement levels: price from high downward
        for factor in FIBONACCI_RETRACEMENTS:
            price = high_price - range_ * factor
            areas.append(self._make_zone(price, [str(factor), "retracement"]))

        # Extension levels: below the low
        for factor in FIBONACCI_EXTENSIONS:
            price = low_price - range_ * factor
            areas.append(self._make_zone(price, [str(factor), "extension"]))

        return areas

    def _compute_uptrend(
        self, low_price: float, high_price: float
    ) -> list[FibonacciArea]:
        """Berechnet Level für ein Uptrend-Leg (low→high)."""
        range_ = high_price - low_price
        areas: list[FibonacciArea] = []

        # Retracement levels: price from low upward
        for factor in FIBONACCI_RETRACEMENTS:
            price = low_price + range_ * factor
            areas.append(self._make_zone(price, [str(factor), "retracement"]))

        # Extension levels: above the high
        for factor in FIBONACCI_EXTENSIONS:
            price = high_price + range_ * factor
            areas.append(self._make_zone(price, [str(factor), "extension"]))

        return areas
