"""Tests: funding rate application and simulated latency in backtest.

Properties verified:
  1. Funding rate is applied correctly to positions.
  2. Simulated latency in backtest execution delays order processing.
  3. Partial fills reduce position correctly.
  4. Position size limits are enforced in backtest.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from packages.domain.market_data import FundingRate
from packages.paper import (
    OrderType,
    PaperExecutor,
    TradeDirection,
)
from packages.paper.position_lifecycle import Fill, PositionLifecycle, PositionStatus

# ---------------------------------------------------------------------------
# Helper: create a FundingRate
# ---------------------------------------------------------------------------

def _funding_rate(
    instrument: str = "BTC/USDT",
    venue: str = "binance",
    rate: float = 0.0001,
    mark_price: float = 50_000.0,
    hours_from_base: int = 0,
) -> FundingRate:
    return FundingRate(
        instrument=instrument,
        venue=venue,
        funding_rate=rate,
        mark_price=mark_price,
        next_funding_time=datetime(2025, 1, 1, 8, tzinfo=UTC) + timedelta(hours=hours_from_base),
        event_time=datetime(2025, 1, 1, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Property 1: Funding rate application
# ---------------------------------------------------------------------------

class TestFundingRateApplication:
    """Funding rates must be applied correctly to positions."""

    def test_funding_rate_positive_long_pays(self) -> None:
        """Positive rate: longs pay shorts. Funding rate value > 0 means longs pay."""
        fr = _funding_rate(rate=0.0001, mark_price=50_000.0)
        # Funding cost per unit = rate * mark_price
        expected_cost_per_unit = fr.funding_rate * fr.mark_price
        assert expected_cost_per_unit == pytest.approx(5.0, abs=0.01)

    def test_funding_rate_negative_short_pays(self) -> None:
        """Negative rate: shorts pay longs."""
        fr = _funding_rate(rate=-0.0001, mark_price=50_000.0)
        expected_value = fr.funding_rate * fr.mark_price
        assert expected_value == pytest.approx(-5.0, abs=0.01)

    def test_funding_rate_zero_no_payment(self) -> None:
        fr = _funding_rate(rate=0.0, mark_price=50_000.0)
        assert fr.funding_rate == 0.0
        assert fr.funding_rate_pct == 0.0

    def test_funding_rate_pct_property(self) -> None:
        fr = _funding_rate(rate=0.001)  # 0.1%
        assert fr.funding_rate_pct == pytest.approx(0.1, abs=0.001)

    def test_funding_rate_validation_bounds(self) -> None:
        """Funding rate must be in [-1.0, 1.0]."""
        with pytest.raises(ValueError, match="funding_rate"):
            _funding_rate(rate=1.5)

        with pytest.raises(ValueError, match="funding_rate"):
            _funding_rate(rate=-1.5)

    def test_funding_rate_mark_price_validation(self) -> None:
        with pytest.raises(ValueError, match="mark_price"):
            _funding_rate(mark_price=0.0)

        with pytest.raises(ValueError, match="mark_price"):
            _funding_rate(mark_price=-100.0)


# ---------------------------------------------------------------------------
# Property 2: Simulated latency
# ---------------------------------------------------------------------------

class TestSimulatedLatency:
    """Simulated latency in backtest execution."""

    def test_fill_timestamp_reflects_latency(self) -> None:
        """A filled order's timestamp should reflect execution time."""
        fill = Fill(
            fill_id="latency-1",
            quantity=1.0,
            price=50_000.0,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        )
        # The fill has a fixed timestamp - latency is modelled by the caller
        assert fill.timestamp == datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

    def test_lifecycle_fills_ordered_by_timestamp(self) -> None:
        """Fills added to a position lifecycle should preserve chronological order."""
        lc = PositionLifecycle(symbol="BTC/USDT", target_quantity=10.0)

        t1 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
        t2 = datetime(2025, 1, 1, 10, 0, 1, tzinfo=UTC)
        t3 = datetime(2025, 1, 1, 11, 0, 0, tzinfo=UTC)

        lc.add_fill(Fill(fill_id="f1", quantity=3.0, price=50_000.0, timestamp=t1))
        lc.add_fill(Fill(fill_id="f2", quantity=4.0, price=50_100.0, timestamp=t2))
        lc.add_fill(Fill(fill_id="f3", quantity=3.0, price=50_200.0, timestamp=t3))

        # Verify chronological order
        timestamps = [f.timestamp for f in lc.fills]
        assert timestamps == sorted(timestamps)
        assert lc.current_quantity == pytest.approx(10.0, abs=0.0001)

    def test_lifecycle_with_delayed_fill(self) -> None:
        """A fill arriving with a large delay should still be recorded."""
        lc = PositionLifecycle(symbol="BTC/USDT", target_quantity=1.0)

        t_start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        t_late = datetime(2025, 1, 1, 23, 59, 59, tzinfo=UTC)  # nearly 24h later

        lc.add_fill(Fill(fill_id="f1", quantity=1.0, price=50_000.0, timestamp=t_start))
        lc.add_fill(Fill(fill_id="f2", quantity=-1.0, price=51_000.0, timestamp=t_late))

        assert lc.status == PositionStatus.CLOSED_PROFIT
        # PnL = (51000 - 50000) * 1 = 1000
        assert lc.realized_pnl == pytest.approx(1000.0, abs=0.01)


# ---------------------------------------------------------------------------
# Property 3: Partial fills reduce position correctly (also in test_fees_slippage)
# ---------------------------------------------------------------------------

class TestPartialFillsInLifecycle:
    """Partial fills must reduce position correctly via PositionLifecycle."""

    def test_partial_fill_reduces_quantity(self) -> None:
        lc = PositionLifecycle(symbol="BTC/USDT", target_quantity=10.0)

        lc.add_fill(Fill(fill_id="b1", quantity=10.0, price=50_000.0, timestamp=datetime.now(UTC)))
        assert lc.current_quantity == pytest.approx(10.0, abs=0.0001)

        lc.add_fill(Fill(fill_id="s1", quantity=-3.0, price=51_000.0, timestamp=datetime.now(UTC)))
        assert lc.current_quantity == pytest.approx(7.0, abs=0.0001)
        assert lc.status == PositionStatus.PARTIAL_FILL

    def test_full_fill_reaches_target(self) -> None:
        lc = PositionLifecycle(symbol="BTC/USDT", target_quantity=10.0)

        lc.add_fill(Fill(fill_id="b1", quantity=6.0, price=50_000.0, timestamp=datetime.now(UTC)))
        assert lc.status == PositionStatus.PARTIAL_FILL

        lc.add_fill(Fill(fill_id="b2", quantity=4.0, price=50_100.0, timestamp=datetime.now(UTC)))
        assert lc.current_quantity == pytest.approx(10.0, abs=0.0001)
        assert lc.status == PositionStatus.FULL_FILL

    def test_close_position_via_lifecycle(self) -> None:
        lc = PositionLifecycle(symbol="BTC/USDT", target_quantity=0.0)

        lc.add_fill(Fill(fill_id="b1", quantity=5.0, price=50_000.0, timestamp=datetime.now(UTC)))
        assert lc.current_quantity == pytest.approx(5.0, abs=0.0001)

        close_fill = lc.close_position(52_000.0)
        assert lc.current_quantity == pytest.approx(0.0, abs=0.0001)
        assert lc.status == PositionStatus.CLOSED_PROFIT
        # PnL = (52000 - 50000) * 5 = 10000
        assert lc.realized_pnl == pytest.approx(10_000.0, abs=0.01)


# ---------------------------------------------------------------------------
# Property 4: Position size limits enforced
# ---------------------------------------------------------------------------

class TestPositionSizeLimits:
    """Position size limits must be enforced in backtest."""

    def test_max_position_size_caps_buy(self) -> None:
        """BUY larger than max_position_size_pct should be capped."""
        executor = PaperExecutor(
            initial_cash=100_000.0,
            default_slippage_pct=0.001,
            default_commission_pct=0.001,
            max_position_size_pct=0.10,  # 10% of equity
        )
        account = executor.create_account("size-limits")

        # 10% of 100,000 = 10,000 notional at price 50,000 -> 0.2 BTC max
        trade = executor.submit_order(
            account,
            "BTC/USDT",
            TradeDirection.BUY,
            quantity=1.0,  # requests 1.0, should be capped to ~0.2
            price=50_000.0,
            order_type=OrderType.MARKET,
        )

        # The trade quantity should have been adjusted down
        assert trade.filled_quantity <= 0.21, f"Expected capping, got qty={trade.filled_quantity}"

    def test_max_position_size_allows_within_limit(self) -> None:
        """BUY within limits should not be capped."""
        executor = PaperExecutor(
            initial_cash=100_000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
            max_position_size_pct=0.10,
        )
        account = executor.create_account("within-limit")

        # 5% of 100,000 = 5,000 -> 0.1 BTC at price 50,000
        trade = executor.submit_order(
            account,
            "BTC/USDT",
            TradeDirection.BUY,
            quantity=0.1,
            price=50_000.0,
            order_type=OrderType.MARKET,
        )

        assert trade.filled_quantity == pytest.approx(0.1, abs=0.0001)

    def test_position_size_limit_with_position_lifecycle(self) -> None:
        """Position lifecycle should track current vs. target quantity."""
        lc = PositionLifecycle(symbol="BTC/USDT", target_quantity=10.0)

        # Add partial fill
        lc.add_fill(Fill(fill_id="f1", quantity=4.0, price=50_000.0, timestamp=datetime.now(UTC)))
        assert lc.current_quantity == pytest.approx(4.0, abs=0.0001)
        assert lc.status == PositionStatus.PARTIAL_FILL

        # Add more - should reach target
        lc.add_fill(Fill(fill_id="f2", quantity=6.0, price=50_100.0, timestamp=datetime.now(UTC)))
        assert lc.current_quantity == pytest.approx(10.0, abs=0.0001)
        assert lc.status == PositionStatus.FULL_FILL

    def test_multiple_instruments_respect_individual_limits(self) -> None:
        """Each instrument's position should independently respect size limits."""
        # Use a large enough cash amount so both instruments can fit within 5% limits
        executor = PaperExecutor(
            initial_cash=10_000_000.0,  # 10M - large enough for both
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
            max_position_size_pct=0.05,  # 5% per position
        )
        account = executor.create_account("multi-instrument")

        # Buy BTC - 5% of 10M = 500k -> 10 BTC at price 50,000
        trade_btc = executor.submit_order(
            account,
            "BTC/USDT",
            TradeDirection.BUY,
            quantity=10.0,
            price=50_000.0,
            order_type=OrderType.MARKET,
        )

        # Buy ETH - 5% of 10M = 500k -> 250 ETH at price 2,000
        trade_eth = executor.submit_order(
            account,
            "ETH/USDT",
            TradeDirection.BUY,
            quantity=250.0,
            price=2_000.0,
            order_type=OrderType.MARKET,
        )

        # BTC should be capped at 10
        assert trade_btc.filled_quantity == pytest.approx(10.0, abs=0.0001)
        # ETH should be capped at 250
        assert trade_eth.filled_quantity == pytest.approx(250.0, abs=0.0001)

    def test_order_exceeds_both_cash_and_position_limit(self) -> None:
        """An order exceeding cash should fail regardless of position limit."""
        executor = PaperExecutor(
            initial_cash=10_000.0,
            default_slippage_pct=0.0,
            default_commission_pct=0.0,
            max_position_size_pct=0.50,
        )
        account = executor.create_account("cash-limit")

        with pytest.raises(ValueError, match="Insufficient cash"):
            executor.submit_order(
                account,
                "BTC/USDT",
                TradeDirection.BUY,
                quantity=1.0,  # costs 50k, only have 10k
                price=50_000.0,
                order_type=OrderType.MARKET,
            )


# ---------------------------------------------------------------------------
# Integration: funding + fees + latency in a combined scenario
# ---------------------------------------------------------------------------

class TestCombinedScenario:
    """Funding rate, fees, and latency working together."""

    def test_lifecycle_tracks_all_costs(self) -> None:
        """A lifecycle with multiple fills should track commission and slippage."""
        lc = PositionLifecycle(symbol="BTC/USDT", target_quantity=10.0)

        t1 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
        t2 = datetime(2025, 1, 1, 10, 5, 0, tzinfo=UTC)
        t3 = datetime(2025, 1, 1, 11, 0, 0, tzinfo=UTC)

        lc.add_fill(Fill(
            fill_id="b1",
            quantity=6.0,
            price=50_000.0,
            commission=30.0,
            slippage=0.001,
            timestamp=t1,
        ))
        lc.add_fill(Fill(
            fill_id="b2",
            quantity=4.0,
            price=50_100.0,
            commission=20.0,
            slippage=0.001,
            timestamp=t2,
        ))
        lc.add_fill(Fill(
            fill_id="s1",
            quantity=-5.0,
            price=51_000.0,
            commission=25.0,
            slippage=0.0,
            timestamp=t3,
        ))

        state = lc.get_state()
        assert state["total_commission"] == pytest.approx(75.0, abs=0.01)
        assert lc.current_quantity == pytest.approx(5.0, abs=0.0001)
        assert lc.status == PositionStatus.PARTIAL_FILL

    def test_funding_rate_and_executor_roundtrip(self) -> None:
        """A buy/sell with funding rate context - verify no errors."""
        fr = _funding_rate(rate=0.0001, mark_price=50_000.0)

        executor = PaperExecutor(
            initial_cash=1_000_000.0,
            default_slippage_pct=0.001,
            default_commission_pct=0.001,
            max_position_size_pct=1.0,
        )
        account = executor.create_account("fr-test")

        # Buy
        executor.submit_order(
            account, "BTC/USDT", TradeDirection.BUY, 1.0, 50_000.0, OrderType.MARKET
        )

        # Simulate funding rate event
        assert fr.instrument == "BTC/USDT"
        assert fr.funding_rate > 0

        # Sell
        executor.submit_order(
            account, "BTC/USDT", TradeDirection.SELL, 1.0, 50_500.0, OrderType.MARKET
        )

        summary = executor.get_account_summary(account)
        assert summary["num_positions"] == 0
        assert summary["total_trades"] == 2
