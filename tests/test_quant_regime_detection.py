"""Tests für Regime Detection Engine."""
from __future__ import annotations

import math

import pytest

from trading_harness.quant.regime_detection import (
    REGIME_NAMES,
    RegimeDetector,
    RegimeResult,
)


def _make_closes(n: int, start: float = 100.0, drift: float = 0.0,
                  vol: float = 0.01) -> list[dict]:
    """Create n candles with controlled price path."""
    import random
    rng = random.Random(42)  # deterministic
    candles = []
    price = start
    for i in range(n):
        ret = drift + vol * rng.gauss(0, 1)
        price *= math.exp(ret)
        h = price * (1 + abs(rng.gauss(0, 0.005)))
        l = price * (1 - abs(rng.gauss(0, 0.005)))
        candles.append({
            "time": f"2026-01-01T{i:04d}:00:00Z",
            "open": price / math.exp(ret),
            "high": h,
            "low": l,
            "close": price,
            "volume": 1000.0,
        })
    return candles


def _uptrend(n: int = 100) -> list[dict]:
    return _make_closes(n, drift=0.005, vol=0.01)


def _downtrend(n: int = 100) -> list[dict]:
    return _make_closes(n, drift=-0.005, vol=0.01)


def _sideways(n: int = 100) -> list[dict]:
    return _make_closes(n, drift=0.0, vol=0.005)


class TestRegimeDetector:
    def test_uptrend_bullish(self):
        candles = _uptrend(100)
        result = RegimeDetector().detect(candles)
        assert result.regime in ("strong_bull", "weak_bull", "range")

    def test_downtrend_bearish(self):
        candles = _downtrend(100)
        result = RegimeDetector().detect(candles)
        assert result.regime in ("strong_bear", "weak_bear", "range")

    def test_sideways_range(self):
        candles = _sideways(100)
        result = RegimeDetector().detect(candles)
        assert result.regime in ("range", "low_volatility")

    def test_insufficient_data_returns_range(self):
        candles = _make_closes(5)
        result = RegimeDetector().detect(candles)
        assert result.regime == "range"
        assert result.confidence == 0.5

    def test_detect_returns_regime_result(self):
        candles = _uptrend(100)
        result = RegimeDetector().detect(candles)
        assert isinstance(result, RegimeResult)
        assert result.regime in REGIME_NAMES
        assert 0.0 <= result.confidence <= 1.0
        assert result.duration >= 1

    def test_detect_series_returns_list(self):
        candles = _uptrend(100)
        results = RegimeDetector().detect_series(candles)
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert r.regime in REGIME_NAMES

    def test_regime_names_valid(self):
        for name in REGIME_NAMES:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_deterministic(self):
        candles = _uptrend(100)
        r1 = RegimeDetector().detect(candles)
        r2 = RegimeDetector().detect(candles)
        assert r1.regime == r2.regime
        assert r1.confidence == pytest.approx(r2.confidence)

    def test_indicators_populated(self):
        candles = _uptrend(100)
        result = RegimeDetector().detect(candles)
        assert "sma_fast" in result.indicators
        assert "sma_slow" in result.indicators
        assert "adx" in result.indicators
        assert "volatility_class" in result.indicators

    def test_custom_parameters(self):
        candles = _uptrend(100)
        r1 = RegimeDetector(sma_fast=5, sma_slow=20).detect(candles)
        r2 = RegimeDetector(sma_fast=20, sma_slow=50).detect(candles)
        # Both should produce valid results
        assert r1.regime in REGIME_NAMES
        assert r2.regime in REGIME_NAMES

    def test_confidence_bounded(self):
        for _ in range(5):
            candles = _make_closes(100, drift=0.003 * (_ - 2), vol=0.01)
            result = RegimeDetector().detect(candles)
            assert 0.0 <= result.confidence <= 1.0
