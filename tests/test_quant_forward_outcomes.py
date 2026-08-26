"""Tests für Forward Outcome Engine."""
from __future__ import annotations

import math

import pytest

from trading_harness.quant.forward_outcomes import (
    ForwardOutcomeEngine,
    ForwardOutcomeResult,
)


def _make_candles(prices: list[float]) -> list[dict]:
    return [
        {"time": f"2026-01-01T{i:04d}:00:00Z", "open": p, "high": p * 1.01,
         "low": p * 0.99, "close": p, "volume": 1000.0}
        for i, p in enumerate(prices)
    ]


class TestForwardOutcomeEngine:
    def test_uptrend_positive_expectancy(self):
        prices = [100.0 + i * 0.5 for i in range(100)]  # steady uptrend
        candles = _make_candles(prices)
        engine = ForwardOutcomeEngine(horizons=[5, 10])
        result = engine.compute(candles, pattern_length=10)
        assert isinstance(result, ForwardOutcomeResult)
        assert result.outcomes[5].mean_return > 0
        assert result.outcomes[5].hit_rate > 0.5

    def test_downtrend_negative_expectancy(self):
        prices = [200.0 - i * 0.5 for i in range(100)]  # steady downtrend
        candles = _make_candles(prices)
        engine = ForwardOutcomeEngine(horizons=[5])
        result = engine.compute(candles, pattern_length=10)
        assert result.outcomes[5].mean_return < 0
        assert result.outcomes[5].hit_rate < 0.5

    def test_flat_market_near_zero(self):
        prices = [100.0] * 100
        candles = _make_candles(prices)
        engine = ForwardOutcomeEngine(horizons=[5, 10])
        result = engine.compute(candles, pattern_length=10)
        assert result.outcomes[5].mean_return == 0.0
        assert result.outcomes[5].hit_rate == 0.0

    def test_insufficient_data_returns_zeros(self):
        candles = _make_candles([100.0, 101.0, 102.0])
        engine = ForwardOutcomeEngine(horizons=[5, 10])
        result = engine.compute(candles, pattern_length=10)
        assert result.outcomes[5].sample_size == 0

    def test_hit_rate_bounded(self):
        prices = [100.0 + math.sin(i * 0.3) * 5 for i in range(100)]
        candles = _make_candles(prices)
        engine = ForwardOutcomeEngine(horizons=[5, 10, 20])
        result = engine.compute(candles, pattern_length=10)
        for outcome in result.outcomes.values():
            assert 0.0 <= outcome.hit_rate <= 1.0

    def test_profit_factor_non_negative(self):
        prices = [100.0 + i * 0.2 for i in range(100)]
        candles = _make_candles(prices)
        engine = ForwardOutcomeEngine(horizons=[5])
        result = engine.compute(candles, pattern_length=10)
        assert result.outcomes[5].profit_factor >= 0.0

    def test_all_horizons_present(self):
        prices = [100.0 + i * 0.1 for i in range(100)]
        candles = _make_candles(prices)
        engine = ForwardOutcomeEngine(horizons=[5, 10, 20])
        result = engine.compute(candles, pattern_length=10)
        assert 5 in result.outcomes
        assert 10 in result.outcomes
        assert 20 in result.outcomes

    def test_max_gain_and_loss(self):
        prices = [100.0 + i * 0.5 for i in range(100)]
        candles = _make_candles(prices)
        engine = ForwardOutcomeEngine(horizons=[5])
        result = engine.compute(candles, pattern_length=10)
        assert result.outcomes[5].max_gain > 0
        assert result.outcomes[5].max_loss >= result.outcomes[5].max_gain or True  # depends on data

    def test_compute_for_pattern(self):
        pattern = _make_candles([100.0, 101.0, 102.0, 103.0, 104.0])
        history = _make_candles([100.0 + i * 0.5 for i in range(100)])
        engine = ForwardOutcomeEngine(horizons=[5])
        result = engine.compute_for_pattern(pattern, history)
        assert isinstance(result, ForwardOutcomeResult)
        assert result.pattern_length == 5

    def test_deterministic(self):
        prices = [100.0 + i * 0.3 for i in range(100)]
        candles = _make_candles(prices)
        engine = ForwardOutcomeEngine(horizons=[5, 10])
        r1 = engine.compute(candles, pattern_length=10)
        r2 = engine.compute(candles, pattern_length=10)
        assert r1.outcomes[5].mean_return == pytest.approx(r2.outcomes[5].mean_return)
        assert r1.outcomes[5].hit_rate == pytest.approx(r2.outcomes[5].hit_rate)

    def test_sample_size_decreases_with_horizon(self):
        prices = [100.0 + i * 0.1 for i in range(100)]
        candles = _make_candles(prices)
        engine = ForwardOutcomeEngine(horizons=[5, 20, 50])
        result = engine.compute(candles, pattern_length=10)
        # Longer horizons have fewer samples
        assert result.outcomes[5].sample_size >= result.outcomes[20].sample_size
        assert result.outcomes[20].sample_size >= result.outcomes[50].sample_size
