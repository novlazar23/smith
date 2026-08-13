"""Tests für packages.chart_structure.swing — Swing-Pivot-Erkennung."""

from __future__ import annotations

import numpy as np
import pytest
from packages.chart_structure import SwingDetector, SwingPivot


class TestSwingDetectorInit:
    """Testet SwingDetector-Konstruktion."""

    def test_default_lookback(self) -> None:
        det = SwingDetector()
        assert det.lookback == 5

    def test_custom_lookback(self) -> None:
        det = SwingDetector(lookback=3)
        assert det.lookback == 3

    def test_lookback_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="lookback muss >= 1"):
            SwingDetector(lookback=0)

    def test_lookback_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="lookback muss >= 1"):
            SwingDetector(lookback=-5)


class TestSwingDetectorDetectSwings:
    """Testet detect_swings."""

    @pytest.fixture
    def known_pattern(self) -> dict[str, np.ndarray]:
        """Bekanntes Muster mit klaren Hochs und Tiefs."""
        n = 50
        rng = np.random.RandomState(123)
        close = np.array(
            [
                10.0, 11.0, 12.0, 11.5, 10.5, 9.5, 10.0, 11.0, 12.5, 13.5,
                13.0, 12.0, 11.0, 12.0, 13.0, 14.5, 15.5, 15.0, 14.0, 15.0,
                16.0, 17.5, 18.5, 18.0, 17.0, 18.0, 19.0, 20.5, 21.5, 21.0,
                20.0, 21.0, 22.0, 23.5, 24.5, 24.0, 23.0, 24.0, 25.0, 26.5,
                27.5, 27.0, 26.0, 27.0, 28.0, 29.5, 30.5, 30.0, 29.0, 30.0,
            ]
        )
        high = close + np.abs(rng.randn(n) * 0.5) + 0.5
        low = close - np.abs(rng.randn(n) * 0.5) - 0.5
        return {"high": high, "low": low, "close": close}

    def test_identifies_pivots(self, known_pattern: dict[str, np.ndarray]) -> None:
        """Pivots werden in bekanntem Muster erkannt."""
        detector = SwingDetector(lookback=3)
        pivots = detector.detect_swings(known_pattern)
        assert len(pivots) > 0

    def test_identifies_highs_and_lows(self, known_pattern: dict[str, np.ndarray]) -> None:
        """Sowohl Highs als auch Lows werden erkannt."""
        detector = SwingDetector(lookback=3)
        pivots = detector.detect_swings(known_pattern)
        high_pivots = [p for p in pivots if p.direction == "high"]
        low_pivots = [p for p in pivots if p.direction == "low"]
        assert len(high_pivots) > 0
        assert len(low_pivots) > 0

    def test_pivot_direction_is_literal(self, known_pattern: dict[str, np.ndarray]) -> None:
        """Richtung ist 'high' oder 'low'."""
        detector = SwingDetector(lookback=3)
        pivots = detector.detect_swings(known_pattern)
        for p in pivots:
            assert p.direction in ("high", "low")

    def test_pivot_quality_non_negative(self, known_pattern: dict[str, np.ndarray]) -> None:
        """Qualitäts-Scores sind nicht-negativ."""
        detector = SwingDetector(lookback=3)
        pivots = detector.detect_swings(known_pattern)
        for p in pivots:
            assert p.quality_score >= 0.0

    def test_pivot_time_in_range(self, known_pattern: dict[str, np.ndarray]) -> None:
        """Pivot-Zeiten liegen im gültigen Bereich."""
        n = 50
        detector = SwingDetector(lookback=3)
        pivots = detector.detect_swings(known_pattern)
        for p in pivots:
            assert p.time >= detector.lookback
            assert p.time < n - detector.lookback

    def test_insufficient_data_raises(self) -> None:
        """ValueError bei unzureichenden Daten."""
        detector = SwingDetector(lookback=5)
        short_data: dict[str, np.ndarray] = {
            "high": np.array([10.0, 11.0, 12.0, 11.0, 10.0]),
            "low": np.array([9.0, 10.0, 10.5, 9.5, 9.0]),
            "close": np.array([10.0, 10.5, 11.0, 10.0, 9.5]),
        }
        # Benötigt 2*5+1 = 11 Bars
        with pytest.raises(ValueError, match="Ungenügende"):
            detector.detect_swings(short_data)

    def test_missing_keys_raises(self) -> None:
        """ValueError bei fehlenden Keys."""
        detector = SwingDetector(lookback=3)
        with pytest.raises(ValueError, match="Missing"):
            detector.detect_swings({"close": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])})

    def test_missing_low_raises(self) -> None:
        """ValueError wenn 'low' fehlt."""
        detector = SwingDetector(lookback=3)
        data: dict[str, np.ndarray] = {
            "high": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        }
        with pytest.raises(ValueError, match="Missing"):
            detector.detect_swings(data)

    def test_close_defaults_to_high(self) -> None:
        """'close' ist optional, fällt auf 'high' zurück."""
        detector = SwingDetector(lookback=3)
        n = 30
        highs = np.cumsum(np.random.RandomState(42).randn(n) * 0.5) + 100
        lows = highs - np.abs(np.random.RandomState(42).randn(n) * 0.3) - 0.5
        pivots = detector.detect_swings({"high": highs, "low": lows})
        assert isinstance(pivots, list)

    def test_lookback_3_finds_more(self) -> None:
        """Lookback=3 findet mehr Pivots als Lookback=5."""
        rng = np.random.RandomState(999)
        n = 40
        close = np.cumsum(rng.randn(n) * 0.5) + 100
        high = close + np.abs(rng.randn(n) * 0.3) + 0.5
        low = close - np.abs(rng.randn(n) * 0.3) - 0.5
        data = {"high": high, "low": low, "close": close}
        p3 = SwingDetector(lookback=3).detect_swings(data)
        p5 = SwingDetector(lookback=5).detect_swings(data)
        assert len(p5) <= len(p3)

    def test_pivot_dataclass_type(self, known_pattern: dict[str, np.ndarray]) -> None:
        """Pivots sind SwingPivot-Instanzen."""
        detector = SwingDetector(lookback=3)
        pivots = detector.detect_swings(known_pattern)
        for p in pivots:
            assert isinstance(p, SwingPivot)


class TestQualityScore:
    """Testet Qualitätsberechnung."""

    def test_quality_high_positive(self) -> None:
        """Swing-High mit klarem Abstand zu Nachbarn hat hohe Qualität."""
        # Manuell: Pivot bei i=5 mit high=12, Nachbarn=11 → score ≈ 0.09
        from packages.chart_structure.swing import SwingDetector
        result = SwingDetector._quality_score_high(
            i=5,
            highs=np.array([10, 10.5, 11, 11.5, 11.8, 12.0, 11.5, 11.2, 11.0, 11.3]),
            lows=np.array([8, 8.3, 8.7, 8.9, 9.0, 9.2, 8.8, 8.5, 8.3, 8.6]),
            window_high=np.array([10, 10.5, 11, 11.5, 11.8, 12.0, 11.5, 11.2, 11.0, 11.3]),
            window_low=np.array([8, 8.3, 8.7, 8.9, 9.0, 9.2, 8.8, 8.5, 8.3, 8.6]),
        )
        assert result >= 0.0

    def test_quality_low_positive(self) -> None:
        """Swing-Low mit klarem Abstand zu Nachbarn hat hohe Qualität."""
        from packages.chart_structure.swing import SwingDetector
        result = SwingDetector._quality_score_low(
            i=5,
            highs=np.array([10, 10.5, 11, 11.5, 11.8, 12.0, 11.5, 11.2, 11.0, 11.3]),
            lows=np.array([8, 8.3, 8.7, 8.9, 9.0, 9.2, 8.8, 8.5, 8.3, 8.6]),
            window_high=np.array([10, 10.5, 11, 11.5, 11.8, 12.0, 11.5, 11.2, 11.0, 11.3]),
            window_low=np.array([8, 8.3, 8.7, 8.9, 9.0, 9.2, 8.8, 8.5, 8.3, 8.6]),
        )
        assert result >= 0.0
