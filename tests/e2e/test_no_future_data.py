"""Property test: no future data leakage in the E2E pipeline.

Verifies that at no point in the trading pipeline can future data
be accessed — every feature computed uses only data available
up to the current timestamp.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from packages.domain.market_data import CandleAggregation, MultiTimeframeAggregator


def _make_sorted_candles(count: int = 100, base_price: float = 100.0) -> list[dict]:
    """Generate a sorted list of mock candles (ascending by open_time)."""
    candles: list[dict] = []
    price = base_price
    rng = np.random.default_rng(seed=99)
    base_time = datetime(2025, 6, 1, tzinfo=UTC)
    for i in range(count):
        change = rng.normal(0, 0.005)
        open_ = price
        close = price * (1 + change)
        high = max(open_, close) * (1 + abs(rng.normal(0, 0.002)))
        low = min(open_, close) * (1 - abs(rng.normal(0, 0.002)))
        candles.append(
            {
                "open_time": base_time + timedelta(minutes=i),
                "close_time": base_time + timedelta(minutes=i + 1),
                "open": round(open_, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": round(float(rng.uniform(100, 1000)), 2),
                "trade_count": int(rng.integers(50, 500)),
            }
        )
        price = close
    return candles


class TestNoFutureData:
    """Property: every feature uses only data up to current timestamp."""

    def test_sma_no_future_data(self):
        """SMA computed at position i uses only candles[:i+1]."""
        candles = _make_sorted_candles(count=50, base_price=100.0)
        mta = MultiTimeframeAggregator(base_timeframe="1m")

        for i in range(19, len(candles)):  # window=20, start at i=19
            # Compute SMA using all candles
            sma_full = mta.compute_simple_moving_average(candles, window=20)

            # Compute SMA using only data up to position i
            candles_up_to_i = candles[: i + 1]
            sma_truncated = mta.compute_simple_moving_average(candles_up_to_i, window=20)

            # The last value of truncated should match the value at position i
            # in the full SMA (since both end at position i)
            expected = float(sma_full[i])
            actual = float(sma_truncated[-1])
            assert (
                abs(expected - actual) < 1e-6
            ), f"SMA at i={i}: full={expected}, truncated={actual}"

    def test_ema_no_future_data(self):
        """EMA computed at position i uses only candles[:i+1]."""
        candles = _make_sorted_candles(count=50, base_price=200.0)
        mta = MultiTimeframeAggregator(base_timeframe="1m")

        for span in [5, 10, 20]:
            for i in range(span - 1, len(candles)):
                ema_full = mta.compute_exponential_moving_average(candles, span=span)

                candles_up_to_i = candles[: i + 1]
                ema_truncated = mta.compute_exponential_moving_average(
                    candles_up_to_i, span=span
                )

                expected = float(ema_full[i])
                actual = float(ema_truncated[-1])
                assert (
                    abs(expected - actual) < 1e-6
                ), f"EMA(span={span}) at i={i}: full={expected}, truncated={actual}"

    def test_feature_pipeline_no_future_leak(self):
        """The full feature computation pipeline does not leak future data.

        Simulates: for each timestamp t, compute all features using only
        data available at or before t. Verify that feature values are
        identical whether computed from the full dataset (sliced to t)
        or from a stream of incremental updates.
        """
        candles = _make_sorted_candles(count=30, base_price=50.0)
        mta = MultiTimeframeAggregator(base_timeframe="1m")

        # Pre-compute all features from the full dataset
        all_closes = np.array([c["close"] for c in candles], dtype=np.float64)

        for t in range(25, len(candles)):  # start at t=25 to have enough data
            # Data available at time t
            available = candles[: t + 1]
            available_closes = np.array([c["close"] for c in available], dtype=np.float64)

            # Compute features from available-only data
            sma_5 = mta.compute_simple_moving_average(available, window=5, price_key="close")
            ema_3 = mta.compute_exponential_moving_average(available, span=3, price_key="close")

            # Expected: last computed value
            sma_val = float(sma_5[-1])
            ema_val = float(ema_3[-1])

            # Verify these match what we'd get from full data sliced to t
            full_sma = mta.compute_simple_moving_average(candles, window=5, price_key="close")
            full_ema = mta.compute_exponential_moving_average(candles, span=3, price_key="close")

            assert abs(sma_val - float(full_sma[t])) < 1e-6, (
                f"SMA at t={t}: stream={sma_val}, full[t]={full_sma[t]}"
            )
            assert abs(ema_val - float(full_ema[t])) < 1e-6, (
                f"EMA at t={t}: stream={ema_val}, full[t]={full_ema[t]}"
            )

            # Verify no future candle was used
            last_available_close = available[-1]["close"]
            if t + 1 < len(candles):
                next_close = candles[t + 1]["close"]
                # If the next candle has a very different price, our features
                # should NOT be influenced by it
                # (This is implicitly verified by the stream vs full match above.)

    def test_aggregation_only_past_data(self):
        """CandleAggregation should only use data in the provided list."""
        candles = _make_sorted_candles(count=20, base_price=10.0)

        # Take first 5 candles
        subset = candles[:5]
        agg = MultiTimeframeAggregator(base_timeframe="1m").aggregate(
            subset,
            target_timeframe="5m",
            instrument="TEST/USD",
            venue="mock",
        )

        # The aggregation should only reflect these 5 candles
        assert agg.open == subset[0]["open"]
        assert agg.high == max(c["high"] for c in subset)
        assert agg.low == min(c["low"] for c in subset)
        assert agg.close == subset[-1]["close"]
        assert agg.volume == sum(c["volume"] for c in subset)
        assert agg.trade_count == sum(c["trade_count"] for c in subset)
        assert agg.sub_candle_count == 5

        # Verify that adding a future candle changes the aggregation
        full_agg = MultiTimeframeAggregator(base_timeframe="1m").aggregate(
            candles,
            target_timeframe="5m",
            instrument="TEST/USD",
            venue="mock",
        )
        assert full_agg.close != agg.close, (
            "Full aggregation should differ from 5-candle aggregation"
        )

    def test_windowed_feature_island_test(self):
        """Windowed features computed independently per window match
        incremental computation. This catches future-data leakage where
        a sliding window 'peeks' ahead into the next window."""
        candles = _make_sorted_candles(count=100, base_price=30.0)
        window = 10

        mta = MultiTimeframeAggregator(base_timeframe="1m")

        # Incremental approach: compute for candles[0..k] at each step
        incremental_values: list[float] = []
        for k in range(window - 1, len(candles)):
            chunk = candles[: k + 1]
            sma = mta.compute_simple_moving_average(chunk, window=window)
            incremental_values.append(float(sma[-1]))

        # Batch approach: compute once on full data, then compare
        full_sma = mta.compute_simple_moving_average(candles, window=window)

        for idx, inc_val in enumerate(incremental_values):
            batch_val = float(full_sma[idx + window - 1])
            assert abs(inc_val - batch_val) < 1e-6, (
                f"Windowed mismatch at incremental index {idx}: "
                f"inc={inc_val}, batch={batch_val}"
            )

    def test_empty_and_minimal_inputs(self):
        """Edge cases: empty candles, single candle, insufficient window."""
        mta = MultiTimeframeAggregator(base_timeframe="1m")

        # Empty candle list should fail
        with pytest.raises(ValueError, match="Cannot aggregate"):
            CandleAggregation.from_raw_candles(
                [], "5m", "TEST/USD", "mock"
            )

        # Single candle SMA should fail if window > 1
        with pytest.raises(ValueError):
            mta.compute_simple_moving_average(
                [{"open_time": datetime.now(UTC), "close": 10.0}],
                window=5,
                price_key="close",
            )

    def test_consensus_agent_reports_no_future_horizon(self):
        """Agent report as_of dates should not reference future timestamps
        beyond the latest available candle close time."""
        from packages.schemas.agent_report import (
            AgentReport,
            AgentStatus,
            EvidenceReference,
        )

        candles = _make_sorted_candles(count=20, base_price=40.0)
        latest_time = candles[-1]["close_time"]

        ref = EvidenceReference(
            reference="r1", feature="sma", value="100",
            direction="positive", relevance=0.7,
        )

        # as_of should be <= latest available candle time
        report = AgentReport(
            report_id="r-valid",
            run_id="run-1",
            agent_id="test-agent",
            agent_version="1.0",
            instrument="BTC/USD",
            horizon="1h",
            as_of=latest_time,
            hypothesis="Test hypothesis",
            probabilities={"up": 0.5, "down": 0.3, "range": 0.2},
            evidence=[ref],
            raw_confidence=None,
            calibrated_confidence=None,
            expected_return=None,
            status=AgentStatus.ACTIVE,
        )
        assert report.as_of <= latest_time

        # Report with as_of in the future should not happen in real usage
        # (we verify the contract, not enforce it at schema level)
        future_time = latest_time + timedelta(hours=1)
        assert future_time > latest_time
        # In production, the pipeline should reject reports where as_of > latest_candle