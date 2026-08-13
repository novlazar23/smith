"""Tests für packages.chart_structure.resistance — Support/Resistance Level Detection."""

from __future__ import annotations

import numpy as np
import pytest
from packages.chart_structure import (
    SupportResistanceDetector,
    SupportResistanceLevel,
)


class TestSupportResistanceDetectorInit:
    """Testet SupportResistanceDetector-Konstruktion."""

    def test_default_init(self) -> None:
        det = SupportResistanceDetector()
        assert det.price_proximity == 0.002
        assert det.min_touches == 2

    def test_custom_params(self) -> None:
        det = SupportResistanceDetector(price_proximity=0.01, min_touches=3)
        assert det.price_proximity == 0.01
        assert det.min_touches == 3

    def test_invalid_proximity_zero(self) -> None:
        with pytest.raises(ValueError, match="price_proximity muss > 0"):
            SupportResistanceDetector(price_proximity=0)

    def test_invalid_proximity_negative(self) -> None:
        with pytest.raises(ValueError, match="price_proximity muss > 0"):
            SupportResistanceDetector(price_proximity=-0.01)

    def test_invalid_min_touches_zero(self) -> None:
        with pytest.raises(ValueError, match="min_touches muss >= 1"):
            SupportResistanceDetector(min_touches=0)

    def test_invalid_min_touches_negative(self) -> None:
        with pytest.raises(ValueError, match="min_touches muss >= 1"):
            SupportResistanceDetector(min_touches=-5)


class TestSupportResistanceDetectorDetectLevels:
    """Testet detect_levels."""

    def test_detect_levels_missing_close_raises(self) -> None:
        """ValueError wenn 'close' fehlt."""
        detector = SupportResistanceDetector()
        data: dict[str, np.ndarray] = {
            "open": np.array([100.0]),
            "high": np.array([101.0]),
            "low": np.array([99.0]),
        }
        with pytest.raises(ValueError, match="Missing required"):
            detector.detect_levels(data)

    def test_detect_levels_empty_data(self) -> None:
        """Leere close-Array → leere Ergebnisliste."""
        detector = SupportResistanceDetector()
        data = {"close": np.array([])}
        levels = detector.detect_levels(data)
        assert levels == []

    def test_detect_levels_returns_list(self, ) -> None:
        """Ergebnis ist immer Liste."""
        rng = np.random.RandomState(42)
        n = 50
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
        levels = SupportResistanceDetector().detect_levels({"close": close})
        assert isinstance(levels, list)

    def test_detect_levels_strength_range(self) -> None:
        """Stärke-Werte liegen zwischen 0 und 1."""
        rng = np.random.RandomState(42)
        n = 100
        # Data mit klaren Häufungen
        base = np.concatenate([np.full(25, 100.0), np.full(25, 105.0), np.full(25, 95.0), np.full(25, 100.0)])
        close = base + rng.randn(n) * 0.5
        levels = SupportResistanceDetector(price_proximity=0.01).detect_levels({"close": close})
        for level in levels:
            assert 0.0 < level.strength <= 1.0

    def test_detect_levels_touch_count_min(self) -> None:
        """touch_count entspricht min_touches oder höher."""
        rng = np.random.RandomState(42)
        n = 100
        base = np.concatenate([np.full(25, 100.0), np.full(25, 105.0), np.full(25, 95.0), np.full(25, 100.0)])
        close = base + rng.randn(n) * 0.5
        levels = SupportResistanceDetector(price_proximity=0.01, min_touches=3).detect_levels({"close": close})
        for level in levels:
            assert level.touch_count >= 3

    def test_detect_levels_type_classification(self) -> None:
        """Levels sind entweder 'support' oder 'resistance'."""
        rng = np.random.RandomState(42)
        n = 100
        base = np.concatenate([np.full(25, 100.0), np.full(25, 105.0), np.full(25, 95.0), np.full(25, 100.0)])
        close = base + rng.randn(n) * 0.5
        levels = SupportResistanceDetector(price_proximity=0.01).detect_levels({"close": close})
        for level in levels:
            assert level.level_type in ("support", "resistance")

    def test_detect_levels_resistance_above_median(self) -> None:
        """Resistance-Levels liegen über dem Median."""
        rng = np.random.RandomState(42)
        n = 100
        base = np.concatenate([np.full(25, 100.0), np.full(25, 105.0), np.full(25, 95.0), np.full(25, 100.0)])
        close = base + rng.randn(n) * 0.5
        levels = SupportResistanceDetector(price_proximity=0.01).detect_levels({"close": close})
        med = float(np.median(close))
        for level in levels:
            if level.level_type == "resistance":
                assert level.price > med
            elif level.level_type == "support":
                assert level.price <= med

    def test_detect_levels_attributes(self) -> None:
        """Alle Level-Attribute sind korrekt typisiert."""
        rng = np.random.RandomState(42)
        n = 100
        base = np.concatenate([np.full(25, 100.0), np.full(25, 105.0), np.full(25, 95.0), np.full(25, 100.0)])
        close = base + rng.randn(n) * 0.5
        levels = SupportResistanceDetector(price_proximity=0.01).detect_levels({"close": close})
        for level in levels:
            assert isinstance(level, SupportResistanceLevel)
            assert isinstance(level.price, float)
            assert isinstance(level.level_type, str)
            assert isinstance(level.strength, float)
            assert isinstance(level.touch_count, int)

    def test_detect_levels_high_min_touches_few_results(self) -> None:
        """Hohes min_touches reduziert Ergebnismenge."""
        rng = np.random.RandomState(42)
        n = 100
        base = np.concatenate([np.full(25, 100.0), np.full(25, 105.0), np.full(25, 95.0), np.full(25, 100.0)])
        close = base + rng.randn(n) * 0.5
        levels_2 = SupportResistanceDetector(price_proximity=0.01, min_touches=2).detect_levels({"close": close})
        levels_10 = SupportResistanceDetector(price_proximity=0.01, min_touches=10).detect_levels({"close": close})
        assert len(levels_10) <= len(levels_2)


class TestClusterPrices:
    """Testet _cluster_prices (statische Methode)."""

    def test_cluster_empty(self) -> None:
        assert SupportResistanceDetector._cluster_prices(np.array([])) == []

    def test_cluster_single_value(self) -> None:
        result = SupportResistanceDetector._cluster_prices(np.array([100.0]))
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_cluster_identical_values(self) -> None:
        arr = np.full(50, 100.0)
        clusters = SupportResistanceDetector._cluster_prices(arr)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_cluster_separate_groups(self) -> None:
        """Gut getrennte Cluster ergeben separate Gruppen."""
        arr = np.array([100.0] * 10 + [200.0] * 10)
        clusters = SupportResistanceDetector._cluster_prices(arr)
        assert len(clusters) == 2

    def test_cluster_sorted_unique(self) -> None:
        """Cluster verwenden sortierte, eindeutige Werte."""
        arr = np.array([110.0, 90.0, 100.0, 95.0, 105.0, 100.5, 101.0])
        clusters = SupportResistanceDetector._cluster_prices(arr)
        for cluster in clusters:
            for i in range(1, len(cluster)):
                assert cluster[i] >= cluster[i - 1]
