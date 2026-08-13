"""Tests für packages.portfolio.rebalancer — Drift-basiertes Rebalancing."""

from __future__ import annotations

import pytest
from packages.portfolio import Position, Rebalancer


@pytest.fixture
def positions() -> list[Position]:
    return [
        Position(symbol="AAPL", quantity=100.0, avg_price=150.0, current_price=160.0, weight=0.20, pnl=1000.0, pnl_pct=6.67),
        Position(symbol="GOOG", quantity=50.0, avg_price=2800.0, current_price=2900.0, weight=0.29, pnl=5000.0, pnl_pct=3.57),
        Position(symbol="MSFT", quantity=30.0, avg_price=300.0, current_price=310.0, weight=0.093, pnl=300.0, pnl_pct=3.33),
    ]


class TestRebalancerInit:
    """Testet Rebalancer-Konstruktion."""

    def test_default_params(self) -> None:
        reb = Rebalancer()
        assert reb.target_weights == {}
        assert reb.drift_threshold == 0.05
        assert reb.min_trade_size == 0.01
        assert reb.max_rebalance_pct == 0.50

    def test_custom_params(self) -> None:
        reb = Rebalancer(
            target_weights={"AAPL": 0.30, "CASH": 0.40},
            drift_threshold=0.10,
            min_trade_size=0.02,
            max_rebalance_pct=0.30,
        )
        assert reb.target_weights == {"AAPL": 0.30, "CASH": 0.40}
        assert reb.drift_threshold == 0.10
        assert reb.min_trade_size == 0.02
        assert reb.max_rebalance_pct == 0.30


class TestRebalancerCalculateDrift:
    """Testet Drift-Berechnung."""

    def test_drift_calculation(self, positions: list[Position]) -> None:
        """Berechnet korrekten Drift aus Positionen."""
        reb = Rebalancer()
        targets = {"AAPL": 0.25, "GOOG": 0.25, "MSFT": 0.10, "CASH": 0.40}
        drift = reb.calculate_drift(positions, 10000.0, targets)
        assert drift["AAPL"] == pytest.approx(0.20 - 0.25, abs=0.001)
        assert drift["GOOG"] == pytest.approx(0.29 - 0.25, abs=0.001)

    def test_drift_cash_position(self, positions: list[Position]) -> None:
        """CASH-Drift wird korrekt berechnet."""
        reb = Rebalancer()
        # AAPL MV=16000, GOOG MV=145000, MSFT MV=9300 → total MV=170300
        drift = reb.calculate_drift(positions, 200000.0, {"AAPL": 0.20, "CASH": 0.50})
        expected_cash = (200000.0 - 170300.0) / 200000.0
        assert drift["CASH"] == pytest.approx(expected_cash - 0.50, abs=0.001)

    def test_drift_empty_positions(self) -> None:
        """Keine Positionen -> Drift fur alle Target = -target."""
        reb = Rebalancer()
        targets = {"AAPL": 0.30, "CASH": 0.40}
        drift = reb.calculate_drift([], 10000.0, targets)
        assert drift["AAPL"] == pytest.approx(-0.30, abs=0.001)
        assert drift["CASH"] == pytest.approx(1.0 - 0.40, abs=0.001)

    def test_drift_unweighted_positions(self, positions: list[Position]) -> None:
        """Positionen ohne Gewicht (weight=0) liefern Drift=-target."""
        reb = Rebalancer()
        zero_weight_pos = [
            Position(symbol="AAPL", quantity=100.0, avg_price=150.0, current_price=160.0, weight=0.0),
        ]
        drift = reb.calculate_drift(zero_weight_pos, 10000.0, {"AAPL": 0.25, "CASH": 0.50})
        assert drift["AAPL"] == pytest.approx(-0.25, abs=0.001)


class TestRebalancerNeedsRebalance:
    """Testet Rebalance-Bedarf."""

    def test_needs_rebalance_above_threshold(self) -> None:
        """Drift uber Schwellenwert -> Rebalance notig."""
        reb = Rebalancer(drift_threshold=0.05)
        positions = [
            Position(symbol="AAPL", quantity=1000.0, avg_price=150.0, current_price=160.0, weight=0.80),
        ]
        targets = {"AAPL": 0.25, "CASH": 0.50}
        assert reb.needs_rebalance(positions, 10000.0, targets) is True

    def test_no_rebalance_needed(self) -> None:
        """Drift unter Schwellenwert -> kein Rebalance."""
        reb = Rebalancer(drift_threshold=0.05)
        positions = [
            Position(symbol="AAPL", quantity=15.0, avg_price=150.0, current_price=160.0, weight=0.24),
        ]
        targets = {"AAPL": 0.24, "CASH": 0.76}
        assert reb.needs_rebalance(positions, 10000.0, targets) is False

    def test_no_positions_no_cash_target(self) -> None:
        """Keine Positionen, kein Cash-Target -> Rebalance notig."""
        reb = Rebalancer(drift_threshold=0.05)
        targets = {"AAPL": 0.50}
        assert reb.needs_rebalance([], 10000.0, targets) is True

    def test_drift_threshold_affects(self) -> None:
        """Verschiedene Schwellenwerte ergeben verschiedene Ergebnisse."""
        reb_strict = Rebalancer(drift_threshold=0.01)
        reb_loose = Rebalancer(drift_threshold=0.20)
        positions = [
            Position(symbol="AAPL", quantity=30.0, avg_price=150.0, current_price=160.0, weight=0.48),
        ]
        targets = {"AAPL": 0.55, "CASH": 0.40}
        assert reb_strict.needs_rebalance(positions, 10000.0, targets) is True
        assert reb_loose.needs_rebalance(positions, 10000.0, targets) is False

    def test_needs_rebalance_empty_both(self) -> None:
        """Keine Positionen, kein Target -> False."""
        reb = Rebalancer()
        assert reb.needs_rebalance([], 10000.0, {}) is False


class TestRebalancerCalculateOrders:
    """Testet Rebalancing-Orders."""

    def test_orders_generated(self) -> None:
        """Rebalancing-Orders werden generiert."""
        reb = Rebalancer(min_trade_size=0.01)
        positions = [
            Position(symbol="AAPL", quantity=50.0, avg_price=150.0, current_price=160.0, weight=0.80),
        ]
        targets = {"AAPL": 0.50, "CASH": 0.40}
        orders = reb.calculate_rebalance_orders(positions, 10000.0, targets)
        assert isinstance(orders, list)
        assert len(orders) >= 1

    def test_sell_order_direction(self) -> None:
        """Ubergewicht -> SELL-Order."""
        reb = Rebalancer(min_trade_size=0.01)
        positions = [
            Position(symbol="AAPL", quantity=50.0, avg_price=150.0, current_price=160.0, weight=0.80),
        ]
        targets = {"AAPL": 0.30, "CASH": 0.50}
        orders = reb.calculate_rebalance_orders(positions, 10000.0, targets)
        assert any(o["symbol"] == "AAPL" and o["direction"] == "SELL" for o in orders)

    def test_buy_order_direction(self) -> None:
        """Untergewicht -> BUY-Order."""
        reb = Rebalancer(min_trade_size=0.01)
        positions: list[Position] = []
        targets = {"AAPL": 0.30, "CASH": 0.50}
        orders = reb.calculate_rebalance_orders(positions, 10000.0, targets)
        assert any(o["symbol"] == "AAPL" and o["direction"] == "BUY" for o in orders)

    def test_min_trade_size_filters(self) -> None:
        """Kleine Trades werden gefiltert."""
        reb = Rebalancer(min_trade_size=0.10)
        positions = [
            Position(symbol="AAPL", quantity=10.0, avg_price=150.0, current_price=160.0, weight=0.01),
        ]
        targets = {"AAPL": 0.15, "CASH": 0.70}
        orders = reb.calculate_rebalance_orders(positions, 10000.0, targets)
        assert all(o["amount"] >= 1000.0 for o in orders)

    def test_max_rebalance_pct_caps(self) -> None:
        """Max Rebalance Cap begrenzt Orders."""
        reb = Rebalancer(max_rebalance_pct=0.10, min_trade_size=0.001)
        positions = [
            Position(symbol="AAPL", quantity=10.0, avg_price=150.0, current_price=160.0, weight=0.01),
            Position(symbol="GOOG", quantity=50.0, avg_price=2800.0, current_price=2900.0, weight=0.145),
        ]
        targets = {"AAPL": 0.30, "GOOG": 0.30, "CASH": 0.40}
        orders = reb.calculate_rebalance_orders(positions, 10000.0, targets)
        total = sum(o["amount"] for o in orders)
        assert total <= 0.10 * 10000.0 + 1e-6

    def test_order_structure(self) -> None:
        """Order-Dict enthalt symbol, direction, amount, reason."""
        reb = Rebalancer(min_trade_size=0.01)
        positions = [
            Position(symbol="AAPL", quantity=50.0, avg_price=150.0, current_price=160.0, weight=0.80),
        ]
        targets = {"AAPL": 0.30, "CASH": 0.50}
        orders = reb.calculate_rebalance_orders(positions, 10000.0, targets)
        for o in orders:
            assert "symbol" in o
            assert "direction" in o
            assert "amount" in o
            assert "reason" in o
            assert o["amount"] > 0.0

    def test_zero_equity_raises(self) -> None:
        """Zero total_equity lost ValueError."""
        reb = Rebalancer()
        with pytest.raises(ValueError, match="total_equity must be positive"):
            reb.calculate_rebalance_orders([], total_equity=0.0, target_weights={"AAPL": 0.25})

    def test_negative_equity_raises(self) -> None:
        """Negative total_equity lost ValueError."""
        reb = Rebalancer()
        with pytest.raises(ValueError, match="total_equity must be positive"):
            reb.calculate_rebalance_orders([], total_equity=-100.0, target_weights={"AAPL": 0.25})

    def test_order_amounts_positive(self) -> None:
        """Alle Order-Betraege sind positiv."""
        reb = Rebalancer(min_trade_size=0.01)
        positions = [
            Position(symbol="AAPL", quantity=50.0, avg_price=150.0, current_price=160.0, weight=0.80),
        ]
        targets = {"AAPL": 0.30, "CASH": 0.50}
        orders = reb.calculate_rebalance_orders(positions, 10000.0, targets)
        assert all(o["amount"] > 0.0 for o in orders)


class TestRebalancerComputePortfolioPnl:
    """Testet compute_portfolio_pnl."""

    def test_compute_pnl_long_only(self) -> None:
        """PnL fur Long-Positionen korrekt."""
        positions = [
            Position(symbol="AAPL", quantity=100.0, avg_price=150.0, current_price=160.0, weight=0.25, pnl=1000.0, pnl_pct=6.67),
            Position(symbol="GOOG", quantity=50.0, avg_price=2800.0, current_price=2900.0, weight=0.30, pnl=5000.0, pnl_pct=3.57),
        ]
        total_pnl, total_pct = Rebalancer.compute_portfolio_pnl(positions)
        assert total_pnl == 6000.0
        assert total_pct > 0.0

    def test_compute_pnl_mixed(self) -> None:
        """PnL fur gemischte Long/Short-Positionen."""
        positions = [
            Position(symbol="AAPL", quantity=100.0, avg_price=150.0, current_price=160.0, weight=0.40, pnl=1000.0, pnl_pct=6.67),
            Position(symbol="TSLA", quantity=-50.0, avg_price=200.0, current_price=180.0, weight=0.20, pnl=1000.0, pnl_pct=10.0),
        ]
        total_pnl, total_pct = Rebalancer.compute_portfolio_pnl(positions)
        assert total_pnl == 2000.0
        assert total_pct > 0.0

    def test_compute_pnl_empty(self) -> None:
        """Leere Positionen ergeben 0."""
        total_pnl, total_pct = Rebalancer.compute_portfolio_pnl([])
        assert total_pnl == 0.0
        assert total_pct == 0.0

    def test_compute_pnl_zero_weight(self) -> None:
        """Positionen mit weight=0 werden ignoriert."""
        positions = [
            Position(symbol="AAPL", quantity=100.0, avg_price=150.0, current_price=160.0, weight=0.0, pnl=1000.0, pnl_pct=6.67),
        ]
        total_pnl, total_pct = Rebalancer.compute_portfolio_pnl(positions)
        assert total_pnl == 1000.0
        assert total_pct == 0.0
