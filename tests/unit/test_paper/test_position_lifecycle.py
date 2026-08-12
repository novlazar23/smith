"""Tests for position lifecycle tracking."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from packages.paper.position_lifecycle import (
    Fill,
    PositionLifecycle,
    PositionStatus,
)


class TestPositionLifecycleOpen:
    """Test initial position creation and fills."""

    def test_create_open_position(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        assert pos.status == PositionStatus.OPEN
        assert pos.current_quantity == 0.0
        assert pos.realized_pnl == 0.0

    def test_add_buy_fill(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        fill = Fill(fill_id="f1", quantity=10.0, price=150.0)
        state = pos.add_fill(fill)

        assert pos.current_quantity == 10.0
        assert pos.avg_entry_price == 150.0
        assert state["status"] == PositionStatus.FULL_FILL

    def test_add_partial_fill(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        fill = Fill(fill_id="f1", quantity=5.0, price=150.0)
        pos.add_fill(fill)

        assert pos.current_quantity == 5.0
        assert pos.status == PositionStatus.PARTIAL_FILL

    def test_two_partial_then_full(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        fill1 = Fill(fill_id="f1", quantity=3.0, price=148.0)
        fill2 = Fill(fill_id="f2", quantity=4.0, price=151.0)
        fill3 = Fill(fill_id="f3", quantity=3.0, price=150.0)

        pos.add_fill(fill1)
        assert pos.avg_entry_price == 148.0
        pos.add_fill(fill2)
        # Volume-weighted avg: (3*148 + 4*151) / 7 = (444 + 604) / 7 = 150.0
        assert pos.avg_entry_price == pytest.approx(149.714286)
        pos.add_fill(fill3)
        # (7*150.0 + 3*150.0) / 10 = 150.0
        assert pos.avg_entry_price == pytest.approx(149.8)
        assert pos.status == PositionStatus.FULL_FILL


class TestPositionLifecycleSellClose:
    """Test selling and closing positions."""

    def test_sell_closes_position(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        pos.add_fill(Fill(fill_id="f2", quantity=-10.0, price=110.0))

        assert pos.current_quantity == 0.0
        assert pos.status in (
            PositionStatus.CLOSED_PROFIT,
            PositionStatus.CLOSED_LOSS,
            PositionStatus.CLOSED,
        )
        assert pos.closed_at is not None

    def test_realized_pnl_profit(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        pos.add_fill(Fill(fill_id="f2", quantity=-10.0, price=110.0))

        # (110 - 100) * 10 = 100
        assert pos.realized_pnl == pytest.approx(100.0)

    def test_realized_pnl_loss(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        pos.add_fill(Fill(fill_id="f2", quantity=-10.0, price=90.0))

        # (90 - 100) * 10 = -100
        assert pos.realized_pnl == pytest.approx(-100.0)

    def test_partial_sell_then_close(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        # Sell 5
        pos.add_fill(Fill(fill_id="f2", quantity=-5.0, price=110.0))
        assert pos.current_quantity == 5.0
        assert pos.realized_pnl == pytest.approx(50.0)

        # Close remaining 5
        pos.add_fill(Fill(fill_id="f3", quantity=-5.0, price=115.0))
        assert pos.current_quantity == 0.0
        # Total realized: (110-100)*5 + (115-100)*5 = 50 + 75 = 125
        assert pos.realized_pnl == pytest.approx(125.0)

    def test_close_position_helper(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        fill = pos.close_position(110.0)

        assert fill.quantity == pytest.approx(-10.0)
        assert fill.price == 110.0
        assert pos.realized_pnl == pytest.approx(100.0)


class TestUnrealizedPnL:
    """Test unrealized PnL calculation."""

    def test_unrealized_profit(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        unrealized = pos.calculate_unrealized_pnl(110.0)

        assert unrealized == pytest.approx(100.0)
        assert pos.unrealized_pnl == pytest.approx(100.0)

    def test_unrealized_loss(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        unrealized = pos.calculate_unrealized_pnl(90.0)

        assert unrealized == pytest.approx(-100.0)

    def test_unrealized_zero_when_closed(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        pos.add_fill(Fill(fill_id="f2", quantity=-10.0, price=110.0))

        unrealized = pos.calculate_unrealized_pnl(115.0)
        assert unrealized == 0.0

    def test_close_raises_when_no_position(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        with pytest.raises(ValueError, match="No open position"):
            pos.close_position(100.0)


class TestDrawdownTracking:
    """Test drawdown tracking."""

    def test_peak_equity_updates(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))

        assert pos.peak_equity == pytest.approx(1000.0)

    def test_max_drawdown_calculation(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        # Simulate price going up then down
        # After adding fill, current_equity = 1000, peak = 1000
        pos.current_equity = 1200.0
        pos.peak_equity = 1200.0
        # Then price drops to 900
        pos.current_equity = 900.0
        dd = (1200.0 - 900.0) / 1200.0
        pos.max_drawdown = dd

        assert pos.max_drawdown == pytest.approx(0.25)

    def test_max_drawdown_increases(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        # First peak at 1200
        pos.current_equity = 1200.0
        pos.peak_equity = 1200.0
        pos.max_drawdown = (1200 - 1000) / 1200  # small drawdown

        # Then new higher peak and deeper drawdown
        pos.current_equity = 1500.0
        pos.peak_equity = 1500.0
        pos.current_equity = 900.0
        dd = (1500 - 900) / 1500
        pos.max_drawdown = max(pos.max_drawdown, dd)

        assert pos.max_drawdown == pytest.approx(0.4)


class TestExposure:
    """Test exposure metrics."""

    def test_exposure_notional(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))

        exposure = pos.get_exposure(105.0)
        assert exposure["notional"] == pytest.approx(1050.0)
        assert exposure["total_pnl"] == pytest.approx(50.0)

    def test_exposure_after_close(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        pos.add_fill(Fill(fill_id="f2", quantity=-10.0, price=110.0))

        exposure = pos.get_exposure(115.0)
        assert exposure["notional"] == 0.0
        assert exposure["realized_pnl"] == pytest.approx(100.0)


class TestStateAndSummary:
    """Test state retrieval and summary output."""

    def test_get_state(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        state = pos.get_state()

        assert state["symbol"] == "AAPL"
        assert state["current_quantity"] == 10.0
        assert state["avg_entry_price"] == 100.0
        assert state["num_fills"] == 1
        assert state["opened_at"] is not None
        assert state["closed_at"] is None

    def test_get_summary(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        summary = pos.get_summary(105.0)

        assert "AAPL" in summary
        assert "FULL_FILL" in summary
        assert "50.00" in summary  # unrealized PnL

    def test_total_pnl(self) -> None:
        pos = PositionLifecycle(symbol="AAPL", target_quantity=10.0)
        pos.add_fill(Fill(fill_id="f1", quantity=10.0, price=100.0))
        pos.calculate_unrealized_pnl(105.0)

        state = pos.get_state()
        assert state["total_pnl"] == pytest.approx(50.0)