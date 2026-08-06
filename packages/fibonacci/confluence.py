"""Konfluenz-Scanner — Fibonacci-Zonen mit Support/Resistance-Level abgleichen."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ConfluenceResult, FibonacciArea

if TYPE_CHECKING:
    from packages.chart_structure.base import SupportResistanceLevel


class ConfluenceScanner:
    """Sucht Konfluenz zwischen Fibonacci-Zonen und SR-Leveln."""

    def __init__(self, price_proximity: float = 0.002) -> None:
        """Initialisiert den Konfluenz-Scanner.

        Args:
            price_proximity: Proximität-Schwelle als Prozentsatz (0.002 = 0.2%).
        """
        if price_proximity <= 0:
            raise ValueError(
                f"price_proximity muss > 0 sein, erhalten: {price_proximity}"
            )
        self.price_proximity = price_proximity

    def find_confluence(
        self,
        fib_areas: list[FibonacciArea],
        sr_levels: list[SupportResistanceLevel] | None = None,
    ) -> list[ConfluenceResult]:
        """Findet Konfluenz zwischen Fibonacci-Zonen und SR-Leveln.

        Args:
            fib_areas: Liste von FibonacciArea-Objekten.
            sr_levels: Liste von SupportResistanceLevel-Objekten oder None.

        Returns:
            Sortierte Liste von ConfluenceResult nach score absteigend.
        """
        if sr_levels is None:
            return []

        results: list[ConfluenceResult] = []

        for fib_area in fib_areas:
            matching_prices: list[float] = []

            for sr in sr_levels:
                if fib_area.lower <= sr.price <= fib_area.upper:
                    matching_prices.append(sr.price)

            if matching_prices:
                score = min(0.5 + 0.1 * len(matching_prices), 1.0)
                results.append(
                    ConfluenceResult(
                        level=str(fib_area.lower),
                        score=score,
                        matching_prices=matching_prices,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results
