"""Tests für ML Feature Builder."""
from __future__ import annotations

import math
import pytest
from trading_harness.quant.ml_features import MLFeatureBuilder, MLFeatures


class TestMLFeatureBuilder:
    def test_build_basic(self):
        builder = MLFeatureBuilder(normalize=False)
        result = builder.build("BTCUSDT", "1m", {"rsi": 65.0, "macd": 1.2})
        assert isinstance(result, MLFeatures)
        assert result.symbol == "BTCUSDT"
        assert result.features["rsi"] == 65.0

    def test_build_normalized(self):
        builder = MLFeatureBuilder(normalize=True)
        result = builder.build("BTCUSDT", "1m", {"a": 100.0, "b": 200.0, "c": 300.0})
        values = list(result.features.values())
        mean = sum(values) / len(values)
        assert abs(mean) < 0.01

    def test_build_nan_handling(self):
        builder = MLFeatureBuilder(normalize=False, fill_nan=0.0)
        result = builder.build("BTCUSDT", "1m", {"a": float("nan"), "b": 1.0})
        assert result.features["a"] == 0.0
        assert result.features["b"] == 1.0

    def test_build_inf_handling(self):
        builder = MLFeatureBuilder(normalize=False)
        result = builder.build("BTCUSDT", "1m", {"a": float("inf"), "b": 1.0})
        assert result.features["a"] == 0.0

    def test_feature_names_sorted(self):
        builder = MLFeatureBuilder(normalize=False)
        result = builder.build("BTCUSDT", "1m", {"z": 1.0, "a": 2.0, "m": 3.0})
        assert result.feature_names == ["a", "m", "z"]

    def test_build_from_components(self):
        builder = MLFeatureBuilder(normalize=False)
        result = builder.build_from_components(
            "BTCUSDT", "1m",
            price_features={"close": 100.0},
            momentum_features={"rsi": 65.0},
        )
        assert "price_close" in result.features
        assert "momentum_rsi" in result.features

    def test_group_weights_applied(self):
        builder = MLFeatureBuilder(normalize=False)
        result = builder.build_from_components(
            "BTCUSDT", "1m",
            momentum_features={"rsi": 1.0},
            anomaly_features={"score": 1.0},
        )
        assert result.features["momentum_rsi"] == pytest.approx(1.2)
        assert result.features["anomaly_score"] == pytest.approx(0.7)

    def test_empty_features(self):
        builder = MLFeatureBuilder(normalize=True)
        result = builder.build("BTCUSDT", "1m", {})
        assert result.features == {}

    def test_feature_vector(self):
        builder = MLFeatureBuilder(normalize=False)
        result = builder.build("BTCUSDT", "1m", {"z": 1.0, "a": 2.0})
        vec = builder.feature_vector(result.features)
        assert vec == [2.0, 1.0]

    def test_merge(self):
        builder = MLFeatureBuilder(normalize=False)
        merged = builder.merge({"a": 1.0, "b": 2.0}, {"b": 3.0, "c": 4.0})
        assert merged == {"a": 1.0, "b": 3.0, "c": 4.0}

    def test_deterministic(self):
        builder = MLFeatureBuilder(normalize=True)
        r1 = builder.build("BTCUSDT", "1m", {"a": 10.0, "b": 20.0})
        r2 = builder.build("BTCUSDT", "1m", {"a": 10.0, "b": 20.0})
        assert r1.features == r2.features

    def test_metadata(self):
        builder = MLFeatureBuilder(normalize=False)
        result = builder.build("BTCUSDT", "1m", {"a": 1.0}, metadata={"version": 2})
        assert result.metadata == {"version": 2}
