"""Deterministic tests for :class:`LivePnlTracker`."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from packages.live_execution import LivePnlTracker, OrderResult, OrderState

MARK = ("binance", "BTC/USDT")


def _fill(
    *,
    side: str,
    quantity: float,
    price: float,
    symbol: str = "BTC/USDT",
    venue: str = "binance",
    filled: float | None = None,
    when: datetime | None = None,
    order_id: str = "o1",
) -> OrderResult:
    result = OrderResult(
        idempotency_key="k",
        symbol=symbol,
        venue=venue,
        side=side,
        order_type="limit",
        quantity=quantity,
        price=price,
    )
    result.order_id = order_id
    result.state = OrderState.FILLED
    result.status = "filled"
    result.filled_quantity = quantity if filled is None else filled
    result.fill_price = price
    if when is not None:
        result.submitted_at = when
    return result


class TestEmptyTracker:
    def test_summary_empty(self) -> None:
        summary = LivePnlTracker().summary()
        assert summary["realized"] == 0.0
        assert summary["unrealized"] == 0.0
        assert summary["sharpe"] is None
        assert summary["sortino"] is None
        assert summary["max_drawdown"] == 0.0
        assert summary["win_rate"] == 0.0
        assert summary["profit_factor"] is None

    def test_daily_empty(self) -> None:
        assert LivePnlTracker().daily() == []


class TestPositions:
    def test_buy_opens_long_position(self) -> None:
        tracker = LivePnlTracker()
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0))
        summary = tracker.summary()
        assert summary["realized"] == 0.0
        # no mark price → no invented estimate
        assert summary["unrealized"] == 0.0
        marked = tracker.summary({MARK: 110.0})
        assert marked["unrealized"] == pytest.approx(100.0)

    def test_sell_opens_short_position(self) -> None:
        tracker = LivePnlTracker()
        tracker.process_order(_fill(side="sell", quantity=5.0, price=200.0))
        marked = tracker.summary({MARK: 190.0})
        assert marked["unrealized"] == pytest.approx(50.0)

    def test_sell_closes_long_realizes_positive_pnl(self) -> None:
        tracker = LivePnlTracker()
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, order_id="a"))
        tracker.process_order(_fill(side="sell", quantity=10.0, price=110.0, order_id="b"))
        summary = tracker.summary()
        assert summary["realized"] == pytest.approx(100.0)
        assert summary["unrealized"] == 0.0
        assert summary["win_rate"] == pytest.approx(1.0)
        assert summary["profit_factor"] is None  # no losses

    def test_buy_closes_short_realizes_positive_pnl(self) -> None:
        tracker = LivePnlTracker()
        tracker.process_order(_fill(side="sell", quantity=5.0, price=200.0, order_id="a"))
        tracker.process_order(_fill(side="buy", quantity=5.0, price=190.0, order_id="b"))
        summary = tracker.summary()
        assert summary["realized"] == pytest.approx(50.0)
        assert summary["unrealized"] == 0.0

    def test_partial_close_leaves_remaining_position(self) -> None:
        tracker = LivePnlTracker()
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, order_id="a"))
        tracker.process_order(_fill(side="sell", quantity=4.0, price=110.0, order_id="b"))
        summary = tracker.summary({MARK: 105.0})
        assert summary["realized"] == pytest.approx(40.0)  # 4 * (110-100)
        assert summary["unrealized"] == pytest.approx(30.0)  # 6 * (105-100)

    def test_opposite_fill_larger_than_position_flips_position(self) -> None:
        tracker = LivePnlTracker()
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, order_id="a"))
        tracker.process_order(_fill(side="sell", quantity=15.0, price=120.0, order_id="b"))
        summary = tracker.summary({MARK: 130.0})
        assert summary["realized"] == pytest.approx(200.0)  # 10 * (120-100)
        assert summary["unrealized"] == pytest.approx(-50.0)  # short 5 @120 vs 130

    def test_same_side_fills_update_average_entry_price(self) -> None:
        tracker = LivePnlTracker()
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, order_id="a"))
        tracker.process_order(_fill(side="buy", quantity=10.0, price=200.0, order_id="b"))
        # avg entry 150, position 20 @ mark 250 → 20 * 100
        summary = tracker.summary({MARK: 250.0})
        assert summary["unrealized"] == pytest.approx(2000.0)


class TestIgnoredFills:
    def test_zero_filled_quantity_ignored(self) -> None:
        tracker = LivePnlTracker()
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, filled=0.0))
        summary = tracker.summary({MARK: 110.0})
        assert summary["realized"] == 0.0
        assert summary["unrealized"] == 0.0

    def test_missing_fill_price_ignored(self) -> None:
        tracker = LivePnlTracker()
        result = _fill(side="buy", quantity=10.0, price=100.0)
        result.fill_price = None
        tracker.process_order(result)
        summary = tracker.summary({MARK: 110.0})
        assert summary["realized"] == 0.0
        assert summary["unrealized"] == 0.0


class TestUnrealized:
    def test_unrealized_with_mark_prices(self) -> None:
        tracker = LivePnlTracker()
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, order_id="a"))
        tracker.process_order(_fill(side="sell", quantity=2.0, price=95.0, order_id="b",
                                    venue="bybit", symbol="ETH/USDT"))
        summary = tracker.summary({MARK: 102.0, ("bybit", "ETH/USDT"): 90.0})
        assert summary["realized"] == 0.0
        assert summary["unrealized"] == pytest.approx(20.0 + 10.0)

    def test_unrealized_zero_without_mark_prices(self) -> None:
        tracker = LivePnlTracker()
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, order_id="a"))
        assert tracker.summary()["unrealized"] == 0.0
        assert tracker.summary({})["unrealized"] == 0.0


class TestDaily:
    def test_daily_realized_aggregation(self) -> None:
        tracker = LivePnlTracker()
        d = lambda day: datetime(2025, 1, day, tzinfo=UTC)  # noqa: E731
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, when=d(1), order_id="a"))
        tracker.process_order(_fill(side="sell", quantity=10.0, price=110.0, when=d(2), order_id="b"))
        tracker.process_order(_fill(side="buy", quantity=10.0, price=110.0, when=d(2), order_id="c"))
        tracker.process_order(_fill(side="sell", quantity=10.0, price=90.0, when=d(3), order_id="d"))
        rows = tracker.daily()
        assert rows == [
            {"date": "2025-01-03", "pnl": -200.0, "realized": -200.0, "unrealized": 0.0},
            {"date": "2025-01-02", "pnl": 100.0, "realized": 100.0, "unrealized": 0.0},
        ]

    def test_daily_unrealized_reported_on_latest_order_date(self) -> None:
        tracker = LivePnlTracker()
        d = lambda day: datetime(2025, 1, day, tzinfo=UTC)  # noqa: E731
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, when=d(1), order_id="a"))
        tracker.process_order(_fill(side="sell", quantity=10.0, price=110.0, when=d(2), order_id="b"))
        tracker.process_order(_fill(side="buy", quantity=10.0, price=110.0, when=d(2), order_id="c"))
        rows = tracker.daily({MARK: 120.0})
        assert len(rows) == 1
        row = rows[0]
        assert row["date"] == "2025-01-02"
        assert row["realized"] == pytest.approx(100.0)
        assert row["unrealized"] == pytest.approx(100.0)
        assert row["pnl"] == pytest.approx(200.0)


class TestRiskMetrics:
    def _tracker_with_daily_pnl(self) -> LivePnlTracker:
        # Realized daily series: d2: -100, d4: +200, d6: -50
        tracker = LivePnlTracker()
        d = lambda day: datetime(2025, 3, day, tzinfo=UTC)  # noqa: E731
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, when=d(1), order_id="a"))
        tracker.process_order(_fill(side="sell", quantity=10.0, price=90.0, when=d(2), order_id="b"))
        tracker.process_order(_fill(side="buy", quantity=10.0, price=90.0, when=d(3), order_id="c"))
        tracker.process_order(_fill(side="sell", quantity=10.0, price=110.0, when=d(4), order_id="e"))
        tracker.process_order(_fill(side="buy", quantity=10.0, price=110.0, when=d(5), order_id="f"))
        tracker.process_order(_fill(side="sell", quantity=10.0, price=105.0, when=d(6), order_id="g"))
        return tracker

    def test_metrics_on_controlled_daily_pnl(self) -> None:
        daily = [-100.0, 200.0, -50.0]
        mean = sum(daily) / len(daily)
        std = math.sqrt(sum((x - mean) ** 2 for x in daily) / (len(daily) - 1))
        downside_std = math.sqrt(sum(min(x, 0.0) ** 2 for x in daily) / len(daily))

        summary = self._tracker_with_daily_pnl().summary()
        assert summary["realized"] == pytest.approx(50.0)
        assert summary["sharpe"] == pytest.approx(mean / std * math.sqrt(252))
        assert summary["sortino"] == pytest.approx(mean / downside_std * math.sqrt(252))
        assert summary["max_drawdown"] == pytest.approx(0.5)
        assert summary["win_rate"] == pytest.approx(1 / 3)
        assert summary["profit_factor"] == pytest.approx(200.0 / 150.0)

    def test_sortino_none_when_no_downside(self) -> None:
        tracker = LivePnlTracker()
        d = lambda day: datetime(2025, 3, day, tzinfo=UTC)  # noqa: E731
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, when=d(1), order_id="a"))
        tracker.process_order(_fill(side="sell", quantity=10.0, price=110.0, when=d(2), order_id="b"))
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, when=d(3), order_id="c"))
        tracker.process_order(_fill(side="sell", quantity=10.0, price=120.0, when=d(4), order_id="e"))
        daily = [100.0, 200.0]
        mean = sum(daily) / len(daily)
        std = math.sqrt(sum((x - mean) ** 2 for x in daily) / (len(daily) - 1))
        summary = tracker.summary()
        assert summary["sortino"] is None
        assert summary["sharpe"] == pytest.approx(mean / std * math.sqrt(252))

    def test_sharpe_none_with_single_daily_point(self) -> None:
        tracker = LivePnlTracker()
        d = lambda day: datetime(2025, 3, day, tzinfo=UTC)  # noqa: E731
        tracker.process_order(_fill(side="buy", quantity=10.0, price=100.0, when=d(1), order_id="a"))
        tracker.process_order(_fill(side="sell", quantity=10.0, price=110.0, when=d(2), order_id="b"))
        summary = tracker.summary()
        assert summary["sharpe"] is None
        assert summary["sortino"] is None
        assert summary["max_drawdown"] == 0.0
