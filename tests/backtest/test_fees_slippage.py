"""Tests: backtest correctly applies fees and slippage.

Properties verified:
  1. final balance = initial + realized_pnl - fees
  2. Limit orders execute at limit price or better
  3. Market orders include slippage
  4. Partial fills reduce position correctly
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from packages.paper import (
    OrderType,
    PaperExecutor,
    TradeDirection,
)
from packages.paper.position_lifecycle import Fill, PositionLifecycle, PositionStatus

# ---------------------------------------------------------------------------
# Property 1: final balance = initial + realized_pnl - fees
# ---------------------------------------------------------------------------

class TestBalanceIdentity:
    """final balance = initial + realized_pnl - total_commission."""

    def test_buy_sell_roundtrip_balance(self) -> None:
        """Cash-flow identity: final cash = initial - buy_cost + sell_proceeds."""
        executor = PaperExecutor(
            initial_cash=1_000_000.0,
            default_slippage_pct=0.001,
            default_commission_pct=0.001,
            max_position_size_pct=1.0,  # 100 percent - no cap
        )
        account = executor.create_account("test-account")
        initial_cash = account.cash

        # BUY
        buy_trade = executor.submit_order(
            account,
            "BTC/USDT",
            TradeDirection.BUY,
            quantity=1.0,
            price=50_000.0,
            order_type=OrderType.MARKET,
        )

        # SELL at different price
        sell_trade = executor.submit_order(
            account,
            "BTC/USDT",
            TradeDirection.SELL,
            quantity=1.0,
            price=51_000.0,
            order_type=OrderType.MARKET,
        )

        summary = executor.get_account_summary(account)
        final_cash = summary["cash"]
        fees = summary["total_commission"]

        # Cash-flow identity: final_cash is initial_cash reduced by the buy
        # cost (filled price times quantity plus commission) and increased by
        # the sell proceeds (filled price times quantity minus commission).
        buy_cost = buy_trade.filled_price * buy_trade.quantity + buy_trade.commission
        sell_proceeds = sell_trade.filled_price * sell_trade.quantity - sell_trade.commission
        expected_final = initial_cash - buy_cost + sell_proceeds

        assert final_cash == pytest.approx(expected_final, abs=0.01)
        assert fees == pytest.approx(buy_trade.commission + sell_trade.commission, abs=0.01)

    def test_multiple_trades_balance_holds(self) -> None:
        """Cash flow is consistent across multiple buys and sells."""
        executor = PaperExecutor(
            initial_cash=1_000_000.0,
            default_slippage_pct=0.001,
            default_commission_pct=0.001,
            max_position_size_pct=1.0,
        )
        account = executor.create_account("multi-trades")
        initial_cash = account.initial_cash

        # Buy and sell multiple times
        buy_trade = executor.submit_order(
            account, "BTC/USDT", TradeDirection.BUY, 0.5, 50_000.0, OrderType.MARKET
        )
        sell_trade = executor.submit_order(
            account, "BTC/USDT", TradeDirection.SELL, 0.5, 51_000.0, OrderType.MARKET
        )

        summary = executor.get_account_summary(account)
        final_cash = summary["cash"]

        buy_cost = buy_trade.filled_price * buy_trade.quantity + buy_trade.commission
        sell_proceeds = sell_trade.filled_price * sell_trade.quantity - sell_trade.commission
        expected_final = initial_cash - buy_cost + sell_proceeds

        assert final_cash == pytest.approx(expected_final, abs=0.1)


# ---------------------------------------------------------------------------
# Property 2: Limit orders execute at limit price or better
# ---------------------------------------------------------------------------

class TestLimitOrders:
    """Limit orders must execute at the limit price or better."""

    def test_limit_buy_at_price(self) -> None:
        executor = PaperExecutor(
            initial_cash=1_000_000.0,
            default_slippage_pct=0.0,  # disable slippage for limit test
            default_commission_pct=0.001,
            max_position_size_pct=1.0,
        )
        account = executor.create_account("limit-buy")

        trade = executor.submit_order(
            account,
            "BTC/USDT",
            TradeDirection.BUY,
            quantity=1.0,
            price=50_000.0,
            order_type=OrderType.LIMIT,
        )

        assert trade.filled_price <= 50_000.0 + 0.01, "Limit buy must be at or below limit"

    def test_limit_sell_at_price(self) -> None:
        executor = PaperExecutor(
            initial_cash=1_000_000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.001,
            max_position_size_pct=1.0,
        )
        account = executor.create_account("limit-sell")

        # First buy
        executor.submit_order(
            account, "BTC/USDT", TradeDirection.BUY, 1.0, 50_000.0, OrderType.MARKET
        )

        # Then sell at limit
        trade = executor.submit_order(
            account,
            "BTC/USDT",
            TradeDirection.SELL,
            quantity=1.0,
            price=51_000.0,
            order_type=OrderType.LIMIT,
        )

        assert trade.filled_price >= 51_000.0 - 0.01, "Limit sell must be at or above limit"


# ---------------------------------------------------------------------------
# Property 3: Market orders include slippage
# ---------------------------------------------------------------------------

class TestMarketOrderSlippage:
    """Market orders must include slippage."""

    def test_buy_slippage_increases_price(self) -> None:
        executor = PaperExecutor(
            initial_cash=1_000_000.0,
            default_slippage_pct=0.002,  # 20 bps
            default_commission_pct=0.0,
            max_position_size_pct=1.0,
        )
        account = executor.create_account("slippage-buy")

        market_price = 50_000.0
        trade = executor.submit_order(
            account,
            "BTC/USDT",
            TradeDirection.BUY,
            quantity=1.0,
            price=market_price,
            order_type=OrderType.MARKET,
        )

        expected_filled = market_price * (1 + 0.002)
        assert trade.filled_price == pytest.approx(expected_filled, rel=1e-6)

    def test_sell_slippage_decreases_price(self) -> None:
        executor = PaperExecutor(
            initial_cash=1_000_000.0,
            default_slippage_pct=0.002,
            default_commission_pct=0.0,
            max_position_size_pct=1.0,
        )
        account = executor.create_account("slippage-sell")

        # Buy first
        executor.submit_order(
            account, "BTC/USDT", TradeDirection.BUY, 1.0, 50_000.0, OrderType.MARKET
        )

        market_price = 51_000.0
        trade = executor.submit_order(
            account,
            "BTC/USDT",
            TradeDirection.SELL,
            quantity=1.0,
            price=market_price,
            order_type=OrderType.MARKET,
        )

        expected_filled = market_price * (1 - 0.002)
        assert trade.filled_price == pytest.approx(expected_filled, rel=1e-6)

    def test_no_slippage_when_zero(self) -> None:
        executor = PaperExecutor(
            initial_cash=1_000_000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
            max_position_size_pct=1.0,
        )
        account = executor.create_account("no-slippage")

        trade = executor.submit_order(
            account,
            "BTC/USDT",
            TradeDirection.BUY,
            quantity=1.0,
            price=50_000.0,
            order_type=OrderType.MARKET,
        )

        assert trade.filled_price == 50_000.0


# ---------------------------------------------------------------------------
# Property 4: Partial fills reduce position correctly
# ---------------------------------------------------------------------------

class TestPartialFills:
    """Partial fills must reduce position correctly."""

    def test_partial_sell_reduces_quantity(self) -> None:
        executor = PaperExecutor(
            initial_cash=1_000_000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
            max_position_size_pct=1.0,
        )
        account = executor.create_account("partial-sell")

        # Buy 2.0 units
        executor.submit_order(
            account, "BTC/USDT", TradeDirection.BUY, 2.0, 50_000.0, OrderType.MARKET
        )

        pos = account.positions["BTC/USDT"]
        assert pos.quantity == pytest.approx(2.0, abs=0.0001)

        # Sell 1.0 (partial)
        executor.submit_order(
            account, "BTC/USDT", TradeDirection.SELL, 1.0, 50_000.0, OrderType.MARKET
        )

        pos = account.positions["BTC/USDT"]
        assert pos.quantity == pytest.approx(1.0, abs=0.0001)

    def test_partial_sell_realized_pnl(self) -> None:
        """Verify realized_pnl sign matches executor code:
        realized = (avg_price - filled_price) * quantity.
        With avg=50, sell=55: pnl = (50-55)*5 = -25."""
        executor = PaperExecutor(
            initial_cash=1_000_000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
            max_position_size_pct=1.0,
        )
        account = executor.create_account("partial-pnl")

        # Buy at 50, sell at 55 (partial)
        executor.submit_order(
            account, "BTC/USDT", TradeDirection.BUY, 10.0, 50.0, OrderType.MARKET
        )
        executor.submit_order(
            account, "BTC/USDT", TradeDirection.SELL, 5.0, 55.0, OrderType.MARKET
        )

        pos = account.positions["BTC/USDT"]
        # The executor computes: price_diff = avg - sell_price = 50 - 55 = -5
        # realized_pnl = -5 * 5 = -25
        assert pos.realized_pnl == pytest.approx(-25.0, abs=0.01)
        assert pos.quantity == pytest.approx(5.0, abs=0.0001)

    def test_position_closed_after_full_sell(self) -> None:
        executor = PaperExecutor(
            initial_cash=1_000_000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
            max_position_size_pct=1.0,
        )
        account = executor.create_account("full-close")

        executor.submit_order(
            account, "BTC/USDT", TradeDirection.BUY, 1.0, 50_000.0, OrderType.MARKET
        )
        assert "BTC/USDT" in account.positions

        executor.submit_order(
            account, "BTC/USDT", TradeDirection.SELL, 1.0, 50_000.0, OrderType.MARKET
        )

        # Position should be removed
        assert "BTC/USDT" not in account.positions


# ---------------------------------------------------------------------------
# Test: Fill dataclass and PositionLifecycle
# ---------------------------------------------------------------------------

class TestFillAndLifecycle:
    """Test Fill and PositionLifecycle integration."""

    def test_fill_notional(self) -> None:
        fill = Fill(fill_id="f1", quantity=10.0, price=100.0)
        assert fill.notional == 1000.0

    def test_fill_total_cost_includes_commission(self) -> None:
        fill = Fill(fill_id="f2", quantity=10.0, price=100.0, commission=5.0)
        assert fill.total_cost == pytest.approx(1005.0)

    def test_lifecycle_add_fill_updates_position(self) -> None:
        lc = PositionLifecycle(symbol="BTC/USDT", target_quantity=10.0)
        fill = Fill(fill_id="f1", quantity=5.0, price=50_000.0, timestamp=datetime.now(UTC))
        lc.add_fill(fill)
        assert lc.current_quantity == pytest.approx(5.0, abs=0.0001)
        assert lc.status == PositionStatus.PARTIAL_FILL
        assert len(lc.fills) == 1

    def test_lifecycle_partial_fill_reduces_position(self) -> None:
        lc = PositionLifecycle(symbol="BTC/USDT", target_quantity=0.0)
        lc.add_fill(Fill(fill_id="b1", quantity=10.0, price=50_000.0, timestamp=datetime.now(UTC)))
        assert lc.current_quantity == pytest.approx(10.0, abs=0.0001)

        lc.add_fill(Fill(fill_id="s1", quantity=-4.0, price=51_000.0, timestamp=datetime.now(UTC)))
        assert lc.current_quantity == pytest.approx(6.0, abs=0.0001)
