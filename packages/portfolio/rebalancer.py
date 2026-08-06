"""Rebalancer — Drift-basiertes Portfolio-Rebalancing.

Berechnet Drift von Zielgewichten, prüft Rebalance-Bedarf
und generiert Rebalancing-Orders.
"""

from __future__ import annotations

from typing import Any

from .base import Position


class Rebalancer:
    """Verwaltet drif-basiertes Rebalancing eines Portfolios."""

    def __init__(
        self,
        target_weights: dict[str, float] | None = None,
        drift_threshold: float = 0.05,
        min_trade_size: float = 0.01,
        max_rebalance_pct: float = 0.50,
    ) -> None:
        self.target_weights = target_weights or {}
        self.drift_threshold = drift_threshold
        self.min_trade_size = min_trade_size
        self.max_rebalance_pct = max_rebalance_pct

    def calculate_drift(
        self,
        positions: list[Position],
        total_equity: float,
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        """Berechnet die Abweichung (Drift) von Zielgewichten.

        Args:
            positions: Aktuelle Positionen.
            total_equity: Gesamtwert des Portfolios.
            target_weights: {symbol: target_weight} Ziekgewichte.

        Returns:
            Dict {symbol: drift} mit drift = actual_weight - target_weight.
        """
        # Berechne tatsächliche Gewichte aus Positionen
        symbol_weights: dict[str, float] = {}
        for pos in positions:
            if total_equity > 0:
                symbol_weights[pos.symbol] = (
                    symbol_weights.get(pos.symbol, 0.0) + pos.weight
                )

        drift: dict[str, float] = {}
        for symbol, target in target_weights.items():
            if symbol == "CASH":
                cash = total_equity - sum(
                    pos.market_value for pos in positions
                )
                actual_weight = cash / total_equity if total_equity > 0 else 0.0
            else:
                actual_weight = symbol_weights.get(symbol, 0.0)
            drift[symbol] = actual_weight - target

        return drift

    def needs_rebalance(
        self,
        positions: list[Position],
        total_equity: float,
        target_weights: dict[str, float],
    ) -> bool:
        """Prüft ob eine Rebalancierung erforderlich ist.

        Args:
            positions: Aktuelle Positionen.
            total_equity: Gesamtwert des Portfolios.
            target_weights: {symbol: target_weight} Ziekgewichte.

        Returns:
            True wenn Drift über Schwellenwert.
        """
        # Wenn keine Cash-Target und keine Positionen → Rebalancing nötig
        if "CASH" not in target_weights and not positions:
            return True

        drift = self.calculate_drift(positions, total_equity, target_weights)
        return any(abs(d) > self.drift_threshold for _symbol, d in drift.items())

    def calculate_rebalance_orders(
        self,
        positions: list[Position],
        total_equity: float,
        target_weights: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Berechnet Rebalancing-Orders basierend auf Drift.

        Args:
            positions: Aktuelle Positionen.
            total_equity: Gesamtwert des Portfolios.
            target_weights: {symbol: target_weight} Ziekgewichte.

        Returns:
            Liste von Order-Dicts mit symbol, direction, amount, reason.
        """
        if total_equity <= 0:
            raise ValueError("total_equity must be positive")

        # Berechne tatsächliche Gewichte
        symbol_values: dict[str, float] = {}
        for pos in positions:
            symbol_values[pos.symbol] = (
                symbol_values.get(pos.symbol, 0.0) + pos.market_value
            )

        # Berechne Delta für jedes Ziel-Symbol
        orders: list[dict[str, Any]] = []
        total_rebalance_amount: float = 0.0
        max_rebalance_value = self.max_rebalance_pct * total_equity

        for symbol, target_weight in target_weights.items():
            target_value = target_weight * total_equity
            if symbol == "CASH":
                actual_value = total_equity - sum(
                    pos.market_value for pos in positions
                )
            else:
                actual_value = symbol_values.get(symbol, 0.0)

            delta = target_value - actual_value

            # Prüfe minimum trade size
            trade_ratio = abs(delta) / total_equity
            if trade_ratio < self.min_trade_size:
                continue

            # Prüfe max rebalance cap
            if total_rebalance_amount + abs(delta) > max_rebalance_value:
                continue

            direction = "BUY" if delta > 0 else "SELL"
            orders.append(
                {
                    "symbol": symbol,
                    "direction": direction,
                    "amount": abs(delta),
                    "reason": "rebalance",
                }
            )
            total_rebalance_amount += abs(delta)

        return orders

    @staticmethod
    def compute_portfolio_pnl(
        positions: list[Position],
    ) -> tuple[float, float]:
        """Berechnet Gesamt-PnL und gewichtetes PnL-Prozent.

        Args:
            positions: Alle Positionen.

        Returns:
            Tuple (total_pnl, total_pnl_pct).
        """
        total_pnl = sum(pos.pnl for pos in positions)

        weighted_pnl_sum = sum(
            pos.pnl_pct * pos.weight for pos in positions if pos.weight > 0
        )
        weighted_sum = sum(
            pos.weight for pos in positions if pos.weight > 0
        )

        total_pnl_pct = (
            weighted_pnl_sum / weighted_sum if weighted_sum > 0 else 0.0
        )

        return total_pnl, total_pnl_pct
