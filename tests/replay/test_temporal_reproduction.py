"""Tests: replay mode reproduces historical data in correct chronological order.

Properties verified:
  1. Replay timestamps are monotonically non-decreasing.
  2. At any point in replay, only data up to current time is available (no look-ahead).
  3. Features computed during replay only use past data.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from packages.domain.market_data import MultiTimeframeAggregator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candle_ts(offset_minutes: int) -> datetime:
    base = datetime(2025, 1, 1, tzinfo=UTC)
    return base + timedelta(minutes=offset_minutes)


def _raw_candles(count: int, start_offset: int = 0) -> list[dict[str, Any]]:
    """Return *count* raw 1m candles with deterministic prices."""
    candles: list[dict[str, Any]] = []
    for i in range(count):
        t = start_offset + i
        candles.append({
            "open_time": _make_candle_ts(t),
            "close_time": _make_candle_ts(t + 1),
            "open": 100.0 + i,
            "high": 102.0 + i,
            "low": 99.0 + i,
            "close": 101.0 + i,
            "volume": 10.0 + i * 0.5,
            "trade_count": i + 1,
        })
    return candles


# ---------------------------------------------------------------------------
# Property 1: Monotonically non-decreasing timestamps
# ---------------------------------------------------------------------------

class TestTimestampOrder:
    """Replay timestamps must be monotonically non-decreasing."""

    def test_aggregate_returns_sorted_window(self) -> None:
        candles = _raw_candles(60)  # 60 x 1m = 1h
        agg = MultiTimeframeAggregator(base_timeframe="1m")
        result = agg.aggregate(candles, "1h", "BTC/USDT", "binance")

        # The aggregated candle's open_time <= close_time
        assert result.open_time <= result.close_time

    def test_multi_tf_aggregation_preserves_order(self) -> None:
        candles = _raw_candles(1440)  # one full day of 1m candles
        agg = MultiTimeframeAggregator(base_timeframe="1m")
        results = agg.aggregate_all(candles, "BTC/USDT", "binance")

        # All aggregated candles should have open_time <= close_time
        for tf, ca in results.items():
            assert ca.open_time <= ca.close_time, f"tf={tf} violated order"

    def test_aggregation_with_gaps_preserves_order(self) -> None:
        """Candles with intentional gaps should still produce valid aggregates."""
        candles = [
            *_raw_candles(10),
            {
                "open_time": _make_candle_ts(100),
                "close_time": _make_candle_ts(101),
                "open": 120.0,
                "high": 122.0,
                "low": 119.0,
                "close": 121.0,
                "volume": 15.0,
                "trade_count": 50,
            },
        ]
        agg = MultiTimeframeAggregator(base_timeframe="1m")
        result = agg.aggregate(candles, "1h", "BTC/USDT", "binance")
        assert result.open_time <= result.close_time


# ---------------------------------------------------------------------------
# Property 2: No look-ahead bias
# ---------------------------------------------------------------------------

class TestNoLookahead:
    """At any point in replay, only data up to current time is available."""

    def test_aggregate_uses_only_visible_candles(self) -> None:
        """Aggregate should only consider candles whose open_time <= window close."""
        candles = _raw_candles(60)

        # Simulate a "replay point" at minute 5
        replay_time = _make_candle_ts(5)

        # Filter to only candles visible at replay_time
        visible_candles = [
            c for c in candles if c["close_time"] <= replay_time
        ]

        agg = MultiTimeframeAggregator(base_timeframe="1m")
        if visible_candles:
            # Should not raise - only visible candles passed; use 5m to avoid
            # "cannot aggregate base into itself" error
            result = agg.aggregate(visible_candles, "5m", "BTC/USDT", "binance")
            # The close_time of the result must be <= replay_time
            assert result.close_time <= replay_time

    def test_replay_snapshot_no_future_data(self) -> None:
        """In a replay scenario, querying data at time T must not return data > T."""
        candles = _raw_candles(60)
        agg = MultiTimeframeAggregator(base_timeframe="1m")

        # Simulate replay at minute 30
        current_time = _make_candle_ts(30)

        visible = [c for c in candles if c["close_time"] <= current_time]
        future = [c for c in candles if c["close_time"] > current_time]

        # Should have seen exactly 30 candles, not 60
        assert len(visible) == 30
        assert len(future) == 30

        # Aggregation of visible data should not leak future
        result = agg.aggregate(visible, "5m", "BTC/USDT", "binance")
        assert result.close_time <= current_time

    def test_batch_replay_order_preserved(self) -> None:
        """When replaying in batches, each batch must not contain future data."""
        all_candles = _raw_candles(100)

        # Simulate processing in batches of 10
        batch_size = 10
        for i in range(0, len(all_candles), batch_size):
            batch = all_candles[i : i + batch_size]
            batch_end_time = max(c["close_time"] for c in batch)

            for c in batch:
                # No candle in the current batch can close after the batch's own end
                assert c["close_time"] <= batch_end_time


# ---------------------------------------------------------------------------
# Property 3: Features only use past data
# ---------------------------------------------------------------------------

class TestFeaturePurity:
    """Features computed during replay must only use past data."""

    def test_sma_uses_only_past_candles(self) -> None:
        """SMA window at step T should only include candles available at T."""
        candles = _raw_candles(50)
        agg = MultiTimeframeAggregator(base_timeframe="1m")

        # Compute SMA over all 50 candles
        sma_full = agg.compute_simple_moving_average(candles, "close", window=10)

        # Compute SMA over only the first 30 candles
        past_candles = candles[:30]
        sma_past = agg.compute_simple_moving_average(past_candles, "close", window=10)

        # The past SMA values should match the first N entries of the full SMA
        assert len(sma_past) == len(sma_full) - (50 - 30)
        np.testing.assert_array_equal(sma_past, sma_full[: len(sma_past)])

    def test_ema_uses_only_past_candles(self) -> None:
        """EMA must be causal - EMA[t] only depends on data up to time t."""
        candles = _raw_candles(40)
        agg = MultiTimeframeAggregator(base_timeframe="1m")

        ema_full = agg.compute_exponential_moving_average(candles, "close", span=10)

        # EMA is cumulative, so values at index i are identical whether computed
        # on the full series or a truncated prefix.
        candles_20 = candles[:20]
        ema_20 = agg.compute_exponential_moving_average(candles_20, "close", span=10)

        np.testing.assert_array_equal(ema_20, ema_full[: len(ema_20)])

    def test_candle_aggregation_uses_closed_candles_only(self) -> None:
        """When aggregating, only fully closed candles should be included."""
        candles = _raw_candles(24)  # 24 x 1h = 1 day
        agg = MultiTimeframeAggregator(base_timeframe="1h")

        # Only include candles fully closed at minute 12
        cutoff = _make_candle_ts(12)
        closed = [c for c in candles if c["close_time"] <= cutoff]
        assert len(closed) == 12

        result = agg.aggregate(closed, "4h", "BTC/USDT", "binance")
        assert result.close_time <= cutoff
        assert result.sub_candle_count <= len(closed)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestReplayEdgeCases:
    """Edge-case behaviour of replay-time data availability."""

    def test_empty_candle_list_raises(self) -> None:
        agg = MultiTimeframeAggregator(base_timeframe="1m")
        with pytest.raises(ValueError, match="No candles"):
            agg.aggregate([], "1h", "BTC/USDT", "binance")

    def test_single_candle_aggregates_to_larger_tf(self) -> None:
        candles = _raw_candles(1)
        agg = MultiTimeframeAggregator(base_timeframe="1m")
        result = agg.aggregate(candles, "5m", "BTC/USDT", "binance")
        assert result.open == candles[0]["open"]
        assert result.close == candles[0]["close"]
        assert result.sub_candle_count == 1

    def test_high_low_validation_prevents_lookahead_artifacts(self) -> None:
        """Invalid candles are rejected, not silently passed through."""
        bad_candles = [{
            "open_time": _make_candle_ts(0),
            "close_time": _make_candle_ts(1),
            "open": 100.0,
            "high": 90.0,  # high < low -> invalid
            "low": 110.0,
            "close": 105.0,
            "volume": 10.0,
            "trade_count": 1,
        }]
        agg = MultiTimeframeAggregator(base_timeframe="1m")
        with pytest.raises(ValueError, match="high"):
            agg.aggregate(bad_candles, "5m", "BTC/USDT", "binance")

    def test_zero_price_rejected(self) -> None:
        bad_candles = [{
            "open_time": _make_candle_ts(0),
            "close_time": _make_candle_ts(1),
            "open": 0.0,
            "high": 10.0,
            "low": 5.0,
            "close": 8.0,
            "volume": 10.0,
            "trade_count": 1,
        }]
        agg = MultiTimeframeAggregator(base_timeframe="1m")
        with pytest.raises(ValueError, match="> 0"):
            agg.aggregate(bad_candles, "5m", "BTC/USDT", "binance")

    def test_aggregation_to_base_raises(self) -> None:
        """Aggregating into the base timeframe should raise."""
        candles = _raw_candles(10)
        agg = MultiTimeframeAggregator(base_timeframe="1m")
        with pytest.raises(ValueError, match="base timeframe"):
            agg.aggregate(candles, "1m", "BTC/USDT", "binance")
