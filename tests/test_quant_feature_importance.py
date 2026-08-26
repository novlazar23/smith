"""Tests für Feature Importance Engine."""
from __future__ import annotations

import pytest
from trading_harness.quant.feature_importance import (
    FeatureImportance,
    FeatureImportanceEngine,
    FeatureImportanceResult,
)


class TestFeatureImportanceEngine:
    def test_perfect_correlation(self):
        engine = FeatureImportanceEngine()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = engine.compute({"x": x}, y)
        assert result.features[0].correlation == pytest.approx(1.0)

    def test_negative_correlation(self):
        engine = FeatureImportanceEngine()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = engine.compute({"x": x}, y)
        assert result.features[0].correlation == pytest.approx(-1.0)
        assert result.features[0].importance == pytest.approx(1.0)

    def test_no_correlation(self):
        engine = FeatureImportanceEngine(threshold=0.5)
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 1.0, 4.0, 2.0, 3.0]
        result = engine.compute({"x": x}, y)
        assert result.features[0].importance < 0.5

    def test_ranking(self):
        engine = FeatureImportanceEngine(threshold=0.0)
        features = {
            "perfect": [1.0, 2.0, 3.0, 4.0, 5.0],
            "weak": [5.0, 1.0, 4.0, 2.0, 3.0],
        }
        target = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = engine.compute(features, target)
        assert result.features[0].name == "perfect"
        assert result.features[0].rank == 1

    def test_top_features(self):
        engine = FeatureImportanceEngine(threshold=0.8)
        features = {
            "high": [1.0, 2.0, 3.0, 4.0, 5.0],
            "low": [5.0, 1.0, 4.0, 2.0, 3.0],
        }
        target = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = engine.compute(features, target)
        assert "high" in result.top_features

    def test_group_importance(self):
        engine = FeatureImportanceEngine(threshold=0.0)
        features = {
            "momentum_rsi": [1.0, 2.0, 3.0, 4.0, 5.0],
            "momentum_macd": [2.0, 4.0, 6.0, 8.0, 10.0],
            "volatility_atr": [5.0, 1.0, 4.0, 2.0, 3.0],
        }
        target = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = engine.compute(features, target)
        assert "momentum" in result.feature_groups

    def test_empty_input(self):
        engine = FeatureImportanceEngine()
        result = engine.compute({}, [])
        assert result.features == []

    def test_select_features(self):
        engine = FeatureImportanceEngine(threshold=0.0)
        features = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
        target = [2.0, 4.0, 6.0, 8.0, 10.0]
        selected = engine.select_features(features, target, max_features=1)
        assert len(selected) == 1

    def test_mutual_information(self):
        engine = FeatureImportanceEngine()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        mi = engine.mutual_information_approx(x, y)
        assert mi > 0

    def test_deterministic(self):
        engine = FeatureImportanceEngine()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r1 = engine.compute({"x": x}, y)
        r2 = engine.compute({"x": x}, y)
        assert r1.features[0].correlation == r2.features[0].correlation

    def test_mismatched_lengths(self):
        engine = FeatureImportanceEngine()
        result = engine.compute({"x": [1.0, 2.0]}, [1.0, 2.0, 3.0])
        assert result.features == []
