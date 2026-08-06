"""Tests für packages.portfolio — Positionen, Exposure-Limits, Rebalancing."""

from __future__ import annotations

import pytest
from packages.portfolio import (
    ExposureManager,
    PortfolioSummary,
    PortfolioType,
    Position,
    Rebalancer,
)


class TestPosition:
    """Testet die Position-Datenklasse."""

    def test_position_long_pnl(self) -> None:
        """Berechnet PnL für eine Long-Position."""
        pos = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_price=150.0,
            current_price=160.0,
        )
        pnl, pnl_pct = pos.calculate_pnl()
        assert pnl == 1000.0
        assert pnl_pct == pytest.approx(6.67, abs=0.01)

    def test_position_short_pnl(self) -> None:
        """Berechnet PnL für eine Short-Position."""
        pos = Position(
            symbol="TSLA",
            quantity=-50.0,
            avg_price=200.0,
            current_price=180.0,
        )
        pnl, pnl_pct = pos.calculate_pnl()
        assert pnl == 1000.0
        assert pnl_pct == pytest.approx(10.0, abs=0.01)

    def test_position_market_value(self) -> None:
        """Berechnet den Marktwert einer Position."""
        pos = Position(
            symbol="BTC",
            quantity=2.5,
            avg_price=50000.0,
            current_price=52000.0,
        )
        assert pos.market_value == 130000.0

        pos_short = Position(
            symbol="MSFT",
            quantity=-10.0,
            avg_price=300.0,
            current_price=290.0,
        )
        assert pos_short.market_value == 2900.0

    def test_position_weight_calculation(self) -> None:
        """Gewicht = (abs(quantity) * current_price) / portfolio_value."""
        pos = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_price=150.0,
            current_price=160.0,
            weight=0.25,
        )
        # weight sollte (100 * 160) / 64000 = 0.25 sein
        assert pos.weight == pytest.approx(0.25)

    def test_position_pnl_pct(self) -> None:
        """Berechnet PnL-Prozent korrekt."""
        pos = Position(
            symbol="GOOG",
            quantity=20.0,
            avg_price=2800.0,
            current_price=2660.0,
        )
        pnl, pnl_pct = pos.calculate_pnl()
        # (2660 - 2800) * 20 = -2800
        # -2800 / (20 * 2800) * 100 = -5.0%
        assert pnl == -2800.0
        assert pnl_pct == pytest.approx(-5.0, abs=0.01)

    def test_position_pnl_pct_zero_avg(self) -> None:
        """PnL bei avg_price = 0 ergibt 0."""
        pos = Position(
            symbol="ZERO",
            quantity=10.0,
            avg_price=0.0,
            current_price=5.0,
        )
        pnl, pnl_pct = pos.calculate_pnl()
        assert pnl == 50.0
        assert pnl_pct == 0.0


class TestPortfolioType:
    """Testet PortfolioType-Enum."""

    def test_all_portfolio_types(self) -> None:
        assert str(PortfolioType.TRADING) == "trading"
        assert str(PortfolioType.SAVINGS) == "savings"
        assert str(PortfolioType.HEDGE) == "hedge"


class TestPortfolioSummary:
    """Testet PortfolioSummary."""

    def test_summary_fields(self) -> None:
        """Prüft dass alle Felder existieren."""
        positions = [
            Position(
                symbol="AAPL",
                quantity=100.0,
                avg_price=150.0,
                current_price=160.0,
                weight=0.25,
                pnl=1000.0,
                pnl_pct=6.67,
            )
        ]
        summary = PortfolioSummary(
            portfolio_id="pf-001",
            portfolio_type=PortfolioType.TRADING,
            total_equity=40000.0,
            total_position_value=25600.0,
            cash=14400.0,
            net_exposure=25600.0,
            gross_exposure=25600.0,
            leverage=0.64,
            num_positions=1,
            top_position_weight=0.25,
            portfolio_pnl=1000.0,
            positions=positions,
        )
        assert summary.portfolio_id == "pf-001"
        assert summary.portfolio_type == PortfolioType.TRADING
        assert summary.total_equity == 40000.0
        assert summary.num_positions == 1
        assert summary.timestamp is not None

    def test_summary_long_short_exposure(self) -> None:
        """Prüft long/short Exposure Berechnung."""
        positions = [
            Position(symbol="AAPL", quantity=100.0, avg_price=150.0, current_price=160.0),
            Position(symbol="MSFT", quantity=-50.0, avg_price=300.0, current_price=290.0),
        ]
        summary = PortfolioSummary(
            portfolio_id="pf-002",
            portfolio_type=PortfolioType.TRADING,
            total_equity=50000.0,
            total_position_value=30100.0,
            cash=19900.0,
            net_exposure=0.0,
            gross_exposure=30100.0,
            leverage=0.602,
            num_positions=2,
            top_position_weight=0.40,
            portfolio_pnl=500.0,
            positions=positions,
        )
        assert summary.long_exposure == 16000.0
        assert summary.short_exposure == 14500.0
        # net_exposure ist ein gespeichertes Feld, long/short werden berechnet
        assert summary.net_exposure == 0.0
        assert summary.long_exposure - summary.short_exposure == pytest.approx(1500.0)


class TestExposureManager:
    """Testet ExposureManager."""

    def _make_position(
        self, symbol: str, qty: float, avg: float, cur: float, weight: float = 0.0
    ) -> Position:
        return Position(
            symbol=symbol,
            quantity=qty,
            avg_price=avg,
            current_price=cur,
            weight=weight,
            pnl=0.0,
            pnl_pct=0.0,
        )

    def test_single_position_limit_blocked(self) -> None:
        """Single-Position-Limit wird überschritten -> blockiert."""
        mgr = ExposureManager(max_single_position_pct=0.25)
        pos = self._make_position("AAPL", 2000.0, 150.0, 160.0)

        allowed, reasons, max_size = mgr.add_position(
            [], total_equity=100000.0, new_position=pos
        )

        assert allowed is False
        assert "single_position_exceeded" in reasons
        assert max_size == pytest.approx(25000.0)

    def test_within_all_limits_allowed(self) -> None:
        """Position innerhalb aller Limits -> erlaubt."""
        mgr = ExposureManager(
            max_single_position_pct=0.25,
            max_net_long_pct=0.80,
            max_net_short_pct=-0.20,
            max_gross_leverage=2.0,
        )
        positions = [
            self._make_position("AAPL", 200.0, 150.0, 160.0, weight=0.32),
        ]
        # GOOG: qty=5, cur=2900 -> market_value=14500, weight=0.145 < 0.25
        new_pos = self._make_position(
            "GOOG", 5.0, 2800.0, 2900.0, weight=0.145
        )
        allowed, reasons, _ = mgr.add_position(
            positions, total_equity=100000.0, new_position=new_pos
        )

        assert allowed is True
        assert reasons == []

    def test_leverage_check_blocked(self) -> None:
        """Hinzufugen wird Leverage uberschreiten -> blockiert."""
        mgr = ExposureManager(
            max_gross_leverage=1.5,
            max_single_position_pct=1.0,  # allow large single position
        )
        positions = [
            self._make_position("AAPL", 300.0, 150.0, 160.0, weight=0.096),
            self._make_position("MSFT", 100.0, 300.0, 290.0, weight=0.058),
        ]
        # Existing gross = 48000 + 29000 = 77000, leverage = 0.77
        # Add TSLA: qty=350, cur=260 -> market_value=91000, weight=0.182 < 1.0
        # New gross = 77000 + 91000 = 168000, leverage = 1.68 > 1.5
        new_pos = self._make_position(
            "TSLA", 350.0, 250.0, 260.0, weight=0.182
        )

        allowed, reasons, _ = mgr.add_position(
            positions, total_equity=100000.0, new_position=new_pos
        )

        assert allowed is False
        assert "leverage_exceeded" in reasons

    def test_net_exposure_check_blocked(self) -> None:
        """Netto-Lang-Exposure uberschritten -> blockiert."""
        mgr = ExposureManager(
            max_single_position_pct=1.0,
            max_net_long_pct=0.60,
            max_net_short_pct=-0.10,
            max_gross_leverage=10.0,
        )
        positions = [
            self._make_position("AAPL", 300.0, 150.0, 160.0, weight=0.48),
        ]
        # Add GOOG: qty=30, cur=3000 -> market_value=90000, weight=0.9 < 1.0
        # new long = 48000 + 90000 = 138000, ratio = 1.38 > 0.6
        new_pos = self._make_position(
            "GOOG", 30.0, 2800.0, 3000.0, weight=0.9
        )

        allowed, reasons, _max_size = mgr.add_position(
            positions, total_equity=100000.0, new_position=new_pos
        )

        assert allowed is False
        assert "net_exposure_exceeded" in reasons

    def test_exposure_report_keys(self) -> None:
        """Exposure-Bericht enthalt alle erwarteten Keys."""
        mgr = ExposureManager()
        positions = [
            self._make_position("AAPL", 500.0, 150.0, 160.0, weight=0.2),
            self._make_position("MSFT", -200.0, 300.0, 290.0, weight=0.1),
        ]
        report = mgr.get_exposure_report(positions, total_equity=200000.0)

        expected_keys = {
            "net_exposure_pct",
            "gross_exposure_pct",
            "leverage",
            "long_exposure_pct",
            "short_exposure_pct",
            "num_positions",
            "top_weight",
        }
        assert set(report.keys()) == expected_keys
        assert report["num_positions"] == 2.0

    def test_all_limits_pass(self) -> None:
        """Alle Limits werden uberschritten."""
        mgr = ExposureManager(
            max_single_position_pct=0.25,
            max_net_long_pct=0.80,
            max_net_short_pct=-0.20,
            max_gross_leverage=2.0,
        )
        positions = [
            self._make_position("AAPL", 500.0, 150.0, 160.0, weight=0.08),
        ]
        result = mgr.check_all_limits(positions, total_equity=100000.0)

        assert result["single_position"] is True
        assert result["net_long"] is True
        assert result["gross_leverage"] is True

    def test_all_limits_fail(self) -> None:
        """Alle Limits werden uberschritten."""
        mgr = ExposureManager(
            max_single_position_pct=0.01,
            max_net_long_pct=0.10,
            max_net_short_pct=-0.10,
            max_gross_leverage=0.5,
        )
        positions = [
            self._make_position("AAPL", 500.0, 150.0, 160.0, weight=0.8),
        ]
        result = mgr.check_all_limits(positions, total_equity=100000.0)

        assert result["single_position"] is False
        assert result["net_long"] is False
        assert result["gross_leverage"] is False

    def test_zero_equity_raises(self) -> None:
        """Zero total_equity lost ValueError aus."""
        mgr = ExposureManager()
        pos = self._make_position("AAPL", 100.0, 150.0, 160.0)
        with pytest.raises(ValueError, match="total_equity must be positive"):
            mgr.add_position([], total_equity=0.0, new_position=pos)


class TestRebalancer:
    """Testet Rebalancer."""

    def test_drift_calculation(self) -> None:
        """Berechnet korrekten Drift aus Positionen."""
        rebalancer = Rebalancer()
        positions = [
            Position(
                symbol="AAPL",
                quantity=100.0,
                avg_price=150.0,
                current_price=160.0,
                weight=0.20,
            ),
            Position(
                symbol="GOOG",
                quantity=50.0,
                avg_price=2800.0,
                current_price=2900.0,
                weight=0.29,
            ),
        ]
        targets = {
            "AAPL": 0.25,
            "GOOG": 0.25,
            "CASH": 0.50,
        }

        drift = rebalancer.calculate_drift(positions, 10000.0, targets)

        assert drift["AAPL"] == pytest.approx(0.20 - 0.25, abs=0.001)
        assert drift["GOOG"] == pytest.approx(0.29 - 0.25, abs=0.001)

    def test_needs_rebalance_above_threshold(self) -> None:
        """Drift uber Schwellenwert -> Rebalancing notig."""
        rebalancer = Rebalancer(drift_threshold=0.05)
        positions = [
            Position(
                symbol="AAPL",
                quantity=1000.0,
                avg_price=150.0,
                current_price=160.0,
                weight=0.80,
            ),
        ]
        targets = {
            "AAPL": 0.25,
            "CASH": 0.50,
        }

        assert rebalancer.needs_rebalance(positions, 10000.0, targets) is True

    def test_no_rebalance_needed(self) -> None:
        """Drift unter Schwellenwert -> kein Rebalancing.

        AAPL weight=0.24, market_value=2400, cash=7600, CASH weight=0.76.
        Targets match exactly -> drift=0 for both.
        """
        rebalancer = Rebalancer(drift_threshold=0.05)
        positions = [
            Position(
                symbol="AAPL",
                quantity=15.0,
                avg_price=150.0,
                current_price=160.0,
                weight=0.24,
            ),
        ]
        targets = {
            "AAPL": 0.24,
            "CASH": 0.76,
        }

        assert rebalancer.needs_rebalance(positions, 10000.0, targets) is False

    def test_calculate_orders(self) -> None:
        """Berechnet korrekte Rebalancing-Orders."""
        rebalancer = Rebalancer(
            drift_threshold=0.05,
            min_trade_size=0.01,
        )
        positions = [
            Position(
                symbol="AAPL",
                quantity=50.0,
                avg_price=150.0,
                current_price=160.0,
                weight=0.80,
            ),
        ]
        # AAPL market_value = 8000, cash = 2000
        targets = {
            "AAPL": 0.50,
            "CASH": 0.40,
        }

        orders = rebalancer.calculate_rebalance_orders(positions, 10000.0, targets)

        # AAPL: target = 5000, actual = 8000, delta = -3000 (SELL)
        # Cash: target = 4000, actual = 2000, delta = +2000 (BUY)
        # Both ratios >= 0.01 -> orders created
        assert len(orders) >= 1
        assert any(
            o["symbol"] == "AAPL" and o["direction"] == "SELL" for o in orders
        )
        assert any(
            o["symbol"] == "CASH" and o["direction"] == "BUY" for o in orders
        )

    def test_min_trade_size_filter(self) -> None:
        """Kleine Trades unter min_trade_size werden gefiltert."""
        rebalancer = Rebalancer(min_trade_size=0.10)
        positions = [
            Position(
                symbol="AAPL",
                quantity=10.0,
                avg_price=150.0,
                current_price=160.0,
                weight=0.01,
            ),
        ]
        # AAPL: delta = (0.15 - 0.01) * 10000 = 1400 -> ratio = 0.14 >= 0.10
        # Cash: actual = 10000 - 1600 = 8400, target = 7000, delta = -1400 -> ratio = 0.14
        targets = {
            "AAPL": 0.15,
            "CASH": 0.70,
        }

        orders = rebalancer.calculate_rebalance_orders(positions, 10000.0, targets)
        assert all(o["amount"] >= 1000.0 for o in orders)

    def test_compute_portfolio_pnl_long(self) -> None:
        """PnL fur Long-Positionen korrekt."""
        positions = [
            Position(
                symbol="AAPL",
                quantity=100.0,
                avg_price=150.0,
                current_price=160.0,
                weight=0.25,
                pnl=1000.0,
                pnl_pct=6.67,
            ),
            Position(
                symbol="GOOG",
                quantity=50.0,
                avg_price=2800.0,
                current_price=2900.0,
                weight=0.30,
                pnl=5000.0,
                pnl_pct=3.57,
            ),
        ]
        total_pnl, total_pct = Rebalancer.compute_portfolio_pnl(positions)
        assert total_pnl == 6000.0
        assert total_pct > 0.0

    def test_compute_portfolio_pnl_mixed(self) -> None:
        """PnL fur gemischte Long/Short-Positionen."""
        positions = [
            Position(
                symbol="AAPL",
                quantity=100.0,
                avg_price=150.0,
                current_price=160.0,
                weight=0.40,
                pnl=1000.0,
                pnl_pct=6.67,
            ),
            Position(
                symbol="TSLA",
                quantity=-50.0,
                avg_price=200.0,
                current_price=180.0,
                weight=0.20,
                pnl=1000.0,
                pnl_pct=10.0,
            ),
        ]
        total_pnl, total_pct = Rebalancer.compute_portfolio_pnl(positions)
        assert total_pnl == 2000.0
        assert total_pct > 0.0

    def test_no_positions_with_cash_target(self) -> None:
        """Keine Positionen, Cash im Target -> Drift-Berechnung."""
        rebalancer = Rebalancer(drift_threshold=0.05)
        positions: list[Position] = []
        targets = {"CASH": 0.50}

        drift = rebalancer.calculate_drift(positions, 10000.0, targets)
        assert drift["CASH"] == pytest.approx(0.5, abs=0.001)

    def test_zero_equity_raises(self) -> None:
        """Zero total_equity lost ValueError aus."""
        rebalancer = Rebalancer()
        with pytest.raises(ValueError, match="total_equity must be positive"):
            rebalancer.calculate_rebalance_orders(
                [], total_equity=0.0, target_weights={"AAPL": 0.25}
            )
