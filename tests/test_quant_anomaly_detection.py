"""Tests für Anomaly Detection Engine."""
from __future__ import annotations

import math

import pytest

from trading_harness.quant.anomaly_detection import AnomalyDetector


def _make_candles(prices: list[float], volumes: list[float] | None = None) -> list[dict]:
    """Helper: create candle list from prices."""
    if volumes is None:
        volumes = [1000.0] * len(prices)
    return [
        {"time": f"2026-01-01T{i:02d}:00:00Z", "open": p, "high": p * 1.01,
         "low": p * 0.99, "close": p, "volume": v}
        for i, (p, v) in enumerate(zip(prices, volumes))
    ]


class TestAnomalyDetector:
    def test_empty_candles_returns_empty(self):
        assert AnomalyDetector().detect([]) == []

    def test_insufficient_data_returns_empty(self):
        candles = _make_candles([100.0] * 5)
        assert AnomalyDetector(window_size=20).detect(candles) == []

    def test_normal_data_no_anomalies(self):
        # Stable prices, constant volume → no anomalies
        prices = [100.0 + i * 0.1 for i in range(30)]
        candles = _make_candles(prices)
        anomalies = AnomalyDetector().detect(candles)
        assert len(anomalies) == 0

    def test_price_shock_detected(self):
        # Stable then sudden jump
        prices = [100.0] * 25 + [200.0]  # 100% jump
        candles = _make_candles(prices)
        anomalies = AnomalyDetector(zscore_threshold=2.0).detect(candles)
        price_shocks = [a for a in anomalies if a.anomaly_type == "price_shock"]
        assert len(price_shocks) >= 1
        assert price_shocks[-1].severity > 0

    def test_volume_spike_detected(self):
        # Normal volume then spike
        prices = [100.0] * 30
        volumes = [1000.0] * 25 + [50000.0]  # 50x spike
        candles = _make_candles(prices, volumes)
        anomalies = AnomalyDetector(zscore_threshold=2.0).detect(candles)
        vol_spikes = [a for a in anomalies if a.anomaly_type == "volume_spike"]
        assert len(vol_spikes) >= 1

    def test_severity_bounded_0_1(self):
        prices = [100.0] * 25 + [500.0]  # extreme jump
        candles = _make_candles(prices)
        anomalies = AnomalyDetector(zscore_threshold=2.0).detect(candles)
        for a in anomalies:
            assert 0.0 <= a.severity <= 1.0

    def test_zscore_threshold_configurable(self):
        prices = [100.0] * 25 + [110.0]  # 10% jump
        candles = _make_candles(prices)
        strict = AnomalyDetector(zscore_threshold=1.0).detect(candles)
        loose = AnomalyDetector(zscore_threshold=5.0).detect(candles)
        assert len(strict) >= len(loose)

    def test_window_size_configurable(self):
        prices = [100.0] * 30
        candles = _make_candles(prices)
        small_window = AnomalyDetector(window_size=5).detect(candles)
        large_window = AnomalyDetector(window_size=25).detect(candles)
        # Both should work, different results possible
        assert isinstance(small_window, list)
        assert isinstance(large_window, list)

    def test_detect_single_returns_list(self):
        history = _make_candles([100.0] * 20)
        candle = {"time": "2026-01-01T20:00:00Z", "open": 100.0, "high": 100.1,
                  "low": 99.9, "close": 100.0, "volume": 1000.0}
        result = AnomalyDetector().detect_single(candle, history)
        assert isinstance(result, list)

    def test_deterministic(self):
        prices = [100.0 + math.sin(i * 0.3) * 5 for i in range(30)]
        candles = _make_candles(prices)
        r1 = AnomalyDetector().detect(candles)
        r2 = AnomalyDetector().detect(candles)
        assert len(r1) == len(r2)
        for a1, a2 in zip(r1, r2):
            assert a1.anomaly_type == a2.anomaly_type
            assert a1.severity == pytest.approx(a2.severity)
