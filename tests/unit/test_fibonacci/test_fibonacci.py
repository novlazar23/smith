"""Tests für Fibonacci-Modul — Pivots, Retracements, Extensions, Konfluenz."""

from __future__ import annotations

import numpy as np
import pytest
from packages.chart_structure.base import SupportResistanceLevel
from packages.fibonacci import (
    FIBONACCI_EXTENSIONS,
    FIBONACCI_RETRACEMENTS,
    FibonacciArea,
    FibonacciPivot,
    FibonacciRetracement,
    PivotDetector,
)
from packages.fibonacci.confluence import ConfluenceScanner

# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def alternating_pivots() -> list[FibonacciPivot]:
    """Erstellt alternierende High/Low Pivots für Tests."""
    return [
        FibonacciPivot(price=100.0, time=0, type="swing_high"),
        FibonacciPivot(price=90.0, time=10, type="swing_low"),
        FibonacciPivot(price=110.0, time=20, type="swing_high"),
        FibonacciPivot(price=85.0, time=30, type="swing_low"),
    ]


@pytest.fixture
def sample_data() -> dict[str, np.ndarray]:
    """Erstellt synthetische Marktdaten mit bekannten Pivots."""
    n = 50
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    return {"high": high, "low": low, "close": close}


# ── dataclass tests ───────────────────────────────────────────────────────


class TestFibonacciPivotDataclass:
    """Testet FibonacciPivot dataclass."""

    def test_fields_are_correct(self) -> None:
        pivot = FibonacciPivot(price=100.0, time=5, type="swing_high")
        assert pivot.price == 100.0
        assert pivot.time == 5
        assert pivot.type == "swing_high"


# ── pivot detection tests ─────────────────────────────────────────────────


class TestPivotDetector:
    """Testet PivotDetector."""

    def test_high_pivots(self) -> None:
        """Erkennt Swing-Highs in bekannten Daten."""
        # Known pattern: peak in the middle
        high = np.array(
            [100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0],
            dtype=np.float64,
        )
        low = np.array([95.0] * 9, dtype=np.float64)
        close = np.arange(9.0)
        data = {"high": high, "low": low, "close": close}

        detector = PivotDetector(high_window=2, low_window=2)
        pivots = detector.detect_pivots(data)

        high_pivots = [p for p in pivots if p.type == "swing_high"]
        assert len(high_pivots) == 1
        assert high_pivots[0].price == 104.0
        assert high_pivots[0].time == 4

    def test_low_pivots(self) -> None:
        """Erkennt Swing-Lows in bekannten Daten."""
        low = np.array(
            [100.0, 99.0, 98.0, 97.0, 96.0, 97.0, 98.0, 99.0, 100.0],
            dtype=np.float64,
        )
        high = np.array([105.0] * 9, dtype=np.float64)
        close = np.arange(9.0)
        data = {"high": high, "low": low, "close": close}

        detector = PivotDetector(high_window=2, low_window=2)
        pivots = detector.detect_pivots(data)

        low_pivots = [p for p in pivots if p.type == "swing_low"]
        assert len(low_pivots) == 1
        assert low_pivots[0].price == 96.0
        assert low_pivots[0].time == 4

    def test_alternating_pivots(self) -> None:
        """Erkennt alternierende High/Low Pivots."""
        # Zigzag pattern with known pivots
        highs = np.array(
            [100.0, 95.0, 105.0, 98.0, 110.0, 102.0, 108.0],
            dtype=np.float64,
        )
        lows = np.array(
            [90.0, 85.0, 95.0, 88.0, 98.0, 90.0, 95.0],
            dtype=np.float64,
        )
        close = highs.copy()
        data = {"high": highs, "low": lows, "close": close}

        detector = PivotDetector(high_window=1, low_window=1)
        pivots = detector.detect_pivots(data)

        pivot_types = [p.type for p in pivots]
        assert "swing_high" in pivot_types
        assert "swing_low" in pivot_types

    def test_insufficient_data_raises(self) -> None:
        """Raise ValueError bei ungenügend Daten."""
        close = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        data = {"high": close, "low": close, "close": close}

        detector = PivotDetector(high_window=10, low_window=10)
        with pytest.raises(ValueError, match="Need at least 21 bars, got 3"):
            detector.detect_pivots(data)


# ── retracement tests ─────────────────────────────────────────────────────


class TestFibonacciRetracement:
    """Testet FibonacciRetracement."""

    def test_retracement_levels_correct(self) -> None:
        """Berechnet korrekte Fibonacci-Retracement-Preise."""
        # Uptrend: low=100, high=200 → range=100
        pivots = [
            FibonacciPivot(price=100.0, time=0, type="swing_low"),
            FibonacciPivot(price=200.0, time=10, type="swing_high"),
        ]
        fr = FibonacciRetracement(zone_band=0.005)
        areas = fr.calculate_retracements(pivots)

        # Expected retracement levels: 100 + 100 * factor
        # 0.236 → 123.6, 0.382 → 138.2, 0.5 → 150.0, 0.618 → 161.8, 0.786 → 178.6
        retracements = [a for a in areas if "retracement" in a.level_types]
        assert len(retracements) == 5
        # Check 0.5 retracement zone center
        center_05 = (retracements[2].lower + retracements[2].upper) / 2
        assert abs(center_05 - 150.0) < 0.001

    def test_extension_levels_correct(self) -> None:
        """Berechnet korrekte Fibonacci-Extension-Preise."""
        pivots = [
            FibonacciPivot(price=100.0, time=0, type="swing_low"),
            FibonacciPivot(price=200.0, time=10, type="swing_high"),
        ]
        fr = FibonacciRetracement(zone_band=0.005)
        areas = fr.calculate_retracements(pivots)

        extensions = [a for a in areas if "extension" in a.level_types]
        assert len(extensions) == 2
        # Expected extensions: 200 + 100 * factor
        # 1.272 → 327.2, 1.618 → 361.8
        center_ext0 = (extensions[0].lower + extensions[0].upper) / 2
        assert abs(center_ext0 - 327.2) < 0.001

    def test_retracement_zones_have_band(self) -> None:
        """Jede Zone hat lower < price < upper mit zone_band."""
        pivots = [
            FibonacciPivot(price=100.0, time=0, type="swing_low"),
            FibonacciPivot(price=200.0, time=10, type="swing_high"),
        ]
        fr = FibonacciRetracement(zone_band=0.005)
        areas = fr.calculate_retracements(pivots)

        for area in areas:
            center = (area.lower + area.upper) / 2
            assert area.lower < center < area.upper
            expected_band = center * 0.005
            assert abs(area.upper - area.lower - 2 * expected_band) < 1e-9

    def test_uptrend_leg(self) -> None:
        """Retracement von Swing-Low zu Swing-High im Uptrend."""
        pivots = [
            FibonacciPivot(price=100.0, time=0, type="swing_low"),
            FibonacciPivot(price=200.0, time=10, type="swing_high"),
        ]
        fr = FibonacciRetracement(zone_band=0.005)
        areas = fr.calculate_retracements(pivots)

        # All uptrend levels should be between ~87 and ~370
        for area in areas:
            assert area.lower >= 87.0
            assert area.upper <= 370.0

    def test_downtrend_leg(self) -> None:
        """Retracement von Swing-High zu Swing-Low im Downtrend."""
        pivots = [
            FibonacciPivot(price=200.0, time=0, type="swing_high"),
            FibonacciPivot(price=100.0, time=10, type="swing_low"),
        ]
        fr = FibonacciRetracement(zone_band=0.005)
        areas = fr.calculate_retracements(pivots)

        # Downtrend: retracements from high downward, extensions below low
        extensions = [a for a in areas if "extension" in a.level_types]

        # All should be below the high of 200
        for area in areas:
            assert area.upper <= 200.0

        # Extensions should be below the low of 100
        for area in extensions:
            assert area.upper < 100.0

    def test_fewer_than_2_pivots_returns_empty(self) -> None:
        """Gibt [] zurück bei weniger als 2 Pivots."""
        fr = FibonacciRetracement()
        assert fr.calculate_retracements([]) == []
        assert len(fr.calculate_retracements([FibonacciPivot(price=100.0, time=0, type="swing_high")])) == 0


# ── confluence tests ──────────────────────────────────────────────────────


class TestConfluenceScanner:
    """Testet ConfluenceScanner."""

    def test_finds_match(self) -> None:
        """Findet Konfluenz wenn SR-Level in Fibonacci-Zone fällt."""
        fib_areas = [
            FibonacciArea(lower=148.0, upper=152.0, level_types=["0.5", "retracement"]),
        ]
        sr_levels = [
            SupportResistanceLevel(price=150.0, level_type="support", strength=0.8, touch_count=3),
        ]

        scanner = ConfluenceScanner()
        results = scanner.find_confluence(fib_areas, sr_levels)

        assert len(results) == 1
        assert results[0].score > 0
        assert 150.0 in results[0].matching_prices

    def test_no_match_returns_empty(self) -> None:
        """Gibt [] zurück wenn kein Match."""
        fib_areas = [
            FibonacciArea(lower=148.0, upper=152.0, level_types=["0.5", "retracement"]),
        ]
        sr_levels = [
            SupportResistanceLevel(price=200.0, level_type="support", strength=0.8, touch_count=3),
        ]

        scanner = ConfluenceScanner()
        results = scanner.find_confluence(fib_areas, sr_levels)

        assert results == []

    def test_sorted_by_score(self) -> None:
        """Ergebnisse sind nach score absteigend sortiert."""
        fib_areas = [
            FibonacciArea(lower=148.0, upper=152.0, level_types=["0.5", "retracement"]),
            FibonacciArea(lower=122.0, upper=126.0, level_types=["0.382", "retracement"]),
        ]
        sr_levels = [
            SupportResistanceLevel(price=150.0, level_type="support", strength=0.8, touch_count=3),
            SupportResistanceLevel(price=124.0, level_type="support", strength=0.9, touch_count=5),
            SupportResistanceLevel(price=125.0, level_type="support", strength=0.7, touch_count=2),
        ]

        scanner = ConfluenceScanner()
        results = scanner.find_confluence(fib_areas, sr_levels)

        assert len(results) == 2
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_sr_levels_returns_empty(self) -> None:
        """Gibt [] zurück wenn sr_levels None ist."""
        fib_areas = [
            FibonacciArea(lower=148.0, upper=152.0, level_types=["0.5", "retracement"]),
        ]
        scanner = ConfluenceScanner()
        results = scanner.find_confluence(fib_areas, None)
        assert results == []


# ── constants tests ───────────────────────────────────────────────────────


class TestFibonacciConstants:
    """Testet Fibonacci-Konstanten."""

    def test_retracements_defined(self) -> None:
        expected = [0.236, 0.382, 0.5, 0.618, 0.786]
        assert expected == FIBONACCI_RETRACEMENTS
        assert len(FIBONACCI_RETRACEMENTS) == 5

    def test_extensions_defined(self) -> None:
        expected = [1.272, 1.618]
        assert expected == FIBONACCI_EXTENSIONS
        assert len(FIBONACCI_EXTENSIONS) == 2
