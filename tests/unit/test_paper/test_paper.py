"""Tests for paper trading module — base types and executor."""

from __future__ import annotations

import pytest
from packages.paper import (
    OrderType,
    PaperAccount,
    PaperExecutor,
    PaperPosition,
    Trade,
    TradeDirection,
)


class TestTradeNotionalValue:
    """Test Trade notional value calculation."""

    def test_notional_from_filled_buy(self) -> None:
        trade = Trade(
            trade_id="t1",
            instrument="AAPL",
            direction=TradeDirection.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            price=150.0,
            filled_price=150.15,
            filled_quantity=10.0,
            status="filled",
        )
        assert trade.notional_value == pytest.approx(1501.50)

    def test_notional_zero_when_not_filled(self) -> None:
        trade = Trade(
            trade_id="t2",
            instrument="GOOG",
            direction=TradeDirection.BUY,
            order_type=OrderType.LIMIT,
            quantity=5.0,
            price=2800.0,
            status="pending",
        )
        assert trade.notional_value == 0.0


class TestPaperPositionMarketValue:
    """Test PaperPosition market value calculation."""

    def test_market_value_positive(self) -> None:
        pos = PaperPosition(symbol="AAPL", quantity=100.0, avg_price=150.0)
        assert pos.market_value == 15000.0

    def test_market_value_zero_avg(self) -> None:
        pos = PaperPosition(symbol="AAPL", quantity=100.0, avg_price=0.0)
        assert pos.market_value == 0.0

    def test_market_value_abs_quantity(self) -> None:
        pos = PaperPosition(symbol="AAPL", quantity=-50.0, avg_price=100.0)
        assert pos.market_value == 5000.0


class TestPaperAccountEquity:
    """Test PaperAccount equity calculation."""

    def test_equity_no_positions(self) -> None:
        account = PaperAccount(account_id="demo", cash=100000.0, initial_cash=100000.0)
        assert account.equity == 100000.0

    def test_equity_with_positions(self) -> None:
        pos = PaperPosition(symbol="AAPL", quantity=50.0, avg_price=175.0)
        account = PaperAccount(
            account_id="demo",
            cash=50000.0,
            initial_cash=100000.0,
            positions={"AAPL": pos},
        )
        assert account.equity == 50000.0 + 50.0 * 175.0

    def test_equity_multiple_positions(self) -> None:
        pos_a = PaperPosition(symbol="AAPL", quantity=10.0, avg_price=150.0)
        pos_b = PaperPosition(symbol="GOOG", quantity=5.0, avg_price=2800.0)
        account = PaperAccount(
            account_id="demo",
            cash=10000.0,
            initial_cash=50000.0,
            positions={"AAPL": pos_a, "GOOG": pos_b},
        )
        expected = 10000.0 + 10.0 * 150.0 + 5.0 * 2800.0
        assert account.equity == expected


class TestPaperAccountTotalPnl:
    """Test PaperAccount PnL calculation."""

    def test_total_pnl_combined(self) -> None:
        pos = PaperPosition(
            symbol="AAPL",
            quantity=10.0,
            avg_price=150.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0,
        )
        account = PaperAccount(
            account_id="demo",
            cash=10000.0,
            initial_cash=10000.0,
            positions={"AAPL": pos},
        )
        assert account.total_pnl == 700.0

    def test_total_pnl_zero(self) -> None:
        account = PaperAccount(account_id="demo", cash=10000.0, initial_cash=10000.0)
        assert account.total_pnl == 0.0


class TestExecutorBuyMarketOrder:
    """Test BUY order execution."""

    def test_buy_creates_position(self) -> None:
        executor = PaperExecutor(initial_cash=100000.0, default_slippage_pct=0.0, default_commission_pct=0.0)
        account = executor.create_account("acc1")
        trade = executor.submit_order(account, "AAPL", TradeDirection.BUY, 10.0, price=150.0)

        assert trade.status == "filled"
        assert "AAPL" in account.positions
        pos = account.positions["AAPL"]
        assert pos.quantity == 10.0
        assert pos.avg_price == 150.0

    def test_buy_reduces_cash(self) -> None:
        executor = PaperExecutor(
            initial_cash=100000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
        )
        account = executor.create_account("acc2")
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 10.0, price=150.0)
        assert account.cash == 100000.0 - 10.0 * 150.0


class TestExecutorSellMarketOrder:
    """Test SELL order execution."""

    def test_sell_closes_position(self) -> None:
        executor = PaperExecutor(
            initial_cash=100000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
        )
        account = executor.create_account("acc3")
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 10.0, price=150.0)
        executor.submit_order(account, "AAPL", TradeDirection.SELL, 10.0, price=160.0)

        assert "AAPL" not in account.positions

    def test_sell_increases_cash(self) -> None:
        executor = PaperExecutor(
            initial_cash=100000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
        )
        account = executor.create_account("acc4")
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 10.0, price=150.0)
        executor.submit_order(account, "AAPL", TradeDirection.SELL, 10.0, price=160.0)
        assert account.cash == 100000.0 - 10.0 * 150.0 + 10.0 * 160.0


class TestExecutorInsufficientCash:
    """Test BUY rejection on insufficient cash."""

    def test_buy_fails_when_insufficient(self) -> None:
        executor = PaperExecutor(initial_cash=1000.0)
        account = executor.create_account("acc5")
        with pytest.raises(ValueError, match="Insufficient cash"):
            executor.submit_order(account, "AAPL", TradeDirection.BUY, 100.0, price=150.0)


class TestExecutorInsufficientPosition:
    """Test SELL rejection on insufficient position."""

    def test_sell_fails_when_no_position(self) -> None:
        executor = PaperExecutor()
        account = executor.create_account("acc6")
        with pytest.raises(ValueError, match="No position to sell"):
            executor.submit_order(account, "AAPL", TradeDirection.SELL, 10.0, price=150.0)

    def test_sell_fails_when_partial(self) -> None:
        executor = PaperExecutor()
        account = executor.create_account("acc7")
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 5.0, price=150.0)
        with pytest.raises(ValueError, match="Insufficient position"):
            executor.submit_order(account, "AAPL", TradeDirection.SELL, 10.0, price=150.0)


class TestExecutorSlippageApplied:
    """Test slippage is applied to filled price."""

    def test_buy_slippage_increases_price(self) -> None:
        executor = PaperExecutor(default_slippage_pct=0.001)
        account = executor.create_account("acc8")
        trade = executor.submit_order(account, "AAPL", TradeDirection.BUY, 10.0, price=150.0)
        expected_filled = 150.0 * 1.001
        assert trade.filled_price == pytest.approx(expected_filled)

    def test_sell_slippage_decreases_price(self) -> None:
        executor = PaperExecutor(default_slippage_pct=0.001)
        account = executor.create_account("acc9")
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 10.0, price=150.0)
        trade = executor.submit_order(account, "AAPL", TradeDirection.SELL, 10.0, price=150.0)
        expected_filled = 150.0 * 0.999
        assert trade.filled_price == pytest.approx(expected_filled)


class TestExecutorCommissionCalculated:
    """Test commission is deducted from cash."""

    def test_buy_commission_deducted(self) -> None:
        executor = PaperExecutor(
            initial_cash=100000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.001,
        )
        account = executor.create_account("acc10")
        price = 150.0
        qty = 10.0
        trade = executor.submit_order(account, "AAPL", TradeDirection.BUY, qty, price=price)
        expected_commission = price * qty * 0.001
        assert trade.commission == pytest.approx(expected_commission)
        expected_deduction = price * qty + expected_commission
        assert account.cash == pytest.approx(100000.0 - expected_deduction)


class TestExecutorMaxPositionSize:
    """Test position size limit is enforced."""

    def test_position_capped_at_max_pct(self) -> None:
        executor = PaperExecutor(
            initial_cash=100000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
            max_position_size_pct=0.10,
        )
        account = executor.create_account("acc11")
        # 10% of 100k = 10k, at $100/share = 100 shares max
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 500.0, price=100.0)
        pos = account.positions["AAPL"]
        assert pos.quantity == 100.0  # capped to 10% of equity


class TestExecutorClosePositionPnl:
    """Test close position calculates realized PnL."""

    def test_close_with_profit(self) -> None:
        executor = PaperExecutor(
            initial_cash=100000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
        )
        account = executor.create_account("acc12")
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 10.0, price=100.0)
        trade = executor.close_position(account, "AAPL")
        assert trade is not None
        assert trade.direction == TradeDirection.SELL
        assert trade.filled_quantity == 10.0


class TestExecutorMultipleTrades:
    """Test handling multiple trades in sequence."""

    def test_buy_sell_buy_sequence(self) -> None:
        executor = PaperExecutor(
            initial_cash=100000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
        )
        account = executor.create_account("acc13")

        # Buy 10 AAPL at 100
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 10.0, price=100.0)
        # Sell 5 AAPL at 110
        executor.submit_order(account, "AAPL", TradeDirection.SELL, 5.0, price=110.0)
        # Buy 3 AAPL at 105
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 3.0, price=105.0)

        pos = account.positions["AAPL"]
        assert pos.quantity == 8.0  # 10 - 5 + 3
        assert account.total_trades == 3

    def test_multiple_instruments(self) -> None:
        executor = PaperExecutor(
            initial_cash=100000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
        )
        account = executor.create_account("acc14")
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 10.0, price=100.0)
        executor.submit_order(account, "GOOG", TradeDirection.BUY, 5.0, price=2800.0)

        assert len(account.positions) == 2
        assert "AAPL" in account.positions
        assert "GOOG" in account.positions


class TestExecutorGetSummary:
    """Test account summary generation."""

    def test_summary_after_buy(self) -> None:
        executor = PaperExecutor(
            initial_cash=100000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
        )
        account = executor.create_account("acc15")
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 10.0, price=100.0)
        summary = executor.get_account_summary(account)

        assert summary["account_id"] == "acc15"
        assert summary["cash"] == 99000.0
        assert summary["num_positions"] == 1
        assert summary["total_trades"] == 1
        assert len(summary["position_list"]) == 1
        assert summary["position_list"][0]["symbol"] == "AAPL"

    def test_summary_after_close(self) -> None:
        executor = PaperExecutor(
            initial_cash=100000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
        )
        account = executor.create_account("acc16")
        executor.submit_order(account, "AAPL", TradeDirection.BUY, 10.0, price=100.0)
        executor.close_position(account, "AAPL")
        summary = executor.get_account_summary(account)

        assert summary["num_positions"] == 0
        assert summary["total_trades"] == 2
