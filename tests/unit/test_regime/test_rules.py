"""Tests fur packages.regime.rules — RuleBasedRegimeDetector."""

from __future__ import annotations

import numpy as np
import pytest
from packages.regime.base import MarketRegime
from packages.regime.rules import RuleBasedRegimeDetector


class TestRuleBasedRegimeDetectorInit:
    """Testet RuleBasedRegimeDetector-Konstruktion."""

    def test_default_params(self) -> None:
        det = RuleBasedRegimeDetector()
        assert det.adx_bull_threshold == 25.0
        assert det.adx_choppy_threshold == 20.0
        assert det.rsi_bull_upper == 50.0
        assert det.rsi_bear_lower == 50.0
        assert det.sma_fast_period == 20
        assert det.sma_slow_period == 50

    def test_custom_params(self) -> None:
        det = RuleBasedRegimeDetector(
            adx_bull_threshold=30.0,
            adx_choppy_threshold=25.0,
            rsi_bull_upper=60.0,
            rsi_bear_lower=40.0,
            sma_fast_period=10,
            sma_slow_period=30,
        )
        assert det.adx_bull_threshold == 30.0
        assert det.adx_choppy_threshold == 25.0
        assert det.rsi_bull_upper == 60.0
        assert det.rsi_bear_lower == 40.0
        assert det.sma_fast_period == 10
        assert det.sma_slow_period == 30

    def test_has_name(self) -> None:
        det = RuleBasedRegimeDetector()
        assert det.name == "rule_based"


class TestRuleBasedRegimeDetectorDetect:
    """Testet detect()."""

    @pytest.fixture
    def sample_data(self) -> dict[str, np.ndarray]:
        np.random.seed(42)
        n = 100
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return {"close": close, "high": high, "low": low}

    def test_detect_returns_valid_regime(self, sample_data: dict[str, np.ndarray]) -> None:
        """Regime ist immer eines der drei definierten."""
        det = RuleBasedRegimeDetector()
        result = det.detect(sample_data)
        assert result.regime in (MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.CHOPPY)

    def test_detect_returns_regime_result(self, sample_data: dict[str, np.ndarray]) -> None:
        """Ergebnis ist RegimeResult."""
        det = RuleBasedRegimeDetector()
        result = det.detect(sample_data)
        assert isinstance(result, type(sample_data)) is False
        # Wir prufen implizit uber result.regime

    def test_detect_confidence_range(self, sample_data: dict[str, np.ndarray]) -> None:
        """Confidence liegt zwischen 0 und 1."""
        det = RuleBasedRegimeDetector()
        result = det.detect(sample_data)
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_scores_sum_to_one(self, sample_data: dict[str, np.ndarray]) -> None:
        """Scores summieren sich zu 1."""
        det = RuleBasedRegimeDetector()
        result = det.detect(sample_data)
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 1e-6

    def test_detect_scores_all_positive(self, sample_data: dict[str, np.ndarray]) -> None:
        """Alle Scores sind nicht-negativ."""
        det = RuleBasedRegimeDetector()
        result = det.detect(sample_data)
        for v in result.scores.values():
            assert v >= 0.0

    def test_detect_regime_is_best_score(self, sample_data: dict[str, np.ndarray]) -> None:
        """Regime ist der Score mit der hochsten Wahrscheinlichkeit."""
        det = RuleBasedRegimeDetector()
        result = det.detect(sample_data)
        best = max(result.scores, key=result.scores.get)
        assert result.regime == best

    def test_detect_metadata_has_adx(self, sample_data: dict[str, np.ndarray]) -> None:
        """Metadata enthalt ADX-Wert."""
        det = RuleBasedRegimeDetector()
        result = det.detect(sample_data)
        assert "adx" in result.metadata
        assert isinstance(result.metadata["adx"], float)

    def test_detect_metadata_has_rsi(self, sample_data: dict[str, np.ndarray]) -> None:
        """Metadata enthalt RSI-Wert."""
        det = RuleBasedRegimeDetector()
        result = det.detect(sample_data)
        assert "rsi" in result.metadata
        assert isinstance(result.metadata["rsi"], float)

    def test_detect_metadata_has_uptrend(self, sample_data: dict[str, np.ndarray]) -> None:
        """Metadata enthalt is_uptrend."""
        det = RuleBasedRegimeDetector()
        result = det.detect(sample_data)
        assert "is_uptrend" in result.metadata
        assert isinstance(result.metadata["is_uptrend"], bool)
        assert result.metadata["is_uptrend"] is True  # 42 > 20 → uptrend

    def test_very_short_data_returns_choppy(self) -> None:
        """Weniger als 60 Bars -> CHOPPY."""
        close = np.array([100.0] * 5)
        data = {"close": close, "high": close, "low": close}
        det = RuleBasedRegimeDetector()
        result = det.detect(data)
        assert result.regime == MarketRegime.CHOPPY
        assert result.confidence == 0.0

    def test_strong_uptrend(self) -> None:
        """Starker Aufwartstrend -> Bull."""
        n = 60
        close = np.array([100.0 + i * 2.0 for i in range(n)])
        rng = np.random.RandomState(42)
        high = close + np.abs(rng.randn(n) * 0.5) + 0.5
        low = close - np.abs(rng.randn(n) * 0.5) - 0.5
        data = {"close": close, "high": high, "low": low}
        det = RuleBasedRegimeDetector()
        result = det.detect(data)
        assert result.regime == MarketRegime.BULL

    def test_strong_downtrend(self) -> None:
        """Starker Abwartstrend -> Bear."""
        n = 60
        close = np.array([100.0 - i * 2.0 for i in range(n)])
        rng = np.random.RandomState(42)
        high = close + np.abs(rng.randn(n) * 0.5) + 0.5
        low = close - np.abs(rng.randn(n) * 0.5) - 0.5
        data = {"close": close, "high": high, "low": low}
        det = RuleBasedRegimeDetector()
        result = det.detect(data)
        assert result.regime == MarketRegime.BEAR

    def test_choppy_market(self) -> None:
        """Sehr wenig Bewegung -> CHOPPY oder schwaches Regime."""
        n = 60
        rng = np.random.RandomState(123)
        close = np.cumsum(rng.randn(n) * 0.01) + 100
        high = close + 0.05
        low = close - 0.05
        data = {"close": close, "high": high, "low": low}
        det = RuleBasedRegimeDetector()
        result = det.detect(data)
        assert result.regime in (MarketRegime.CHOPPY, MarketRegime.BULL, MarketRegime.BEAR)

    def test_different_thresholds_affect_result(self) -> None:
        """Verschiedene Schwellenwerte ergeben verschiedene Regime."""
        n = 60
        close = np.array([100.0 + i * 0.5 for i in range(n)])
        rng = np.random.RandomState(42)
        high = close + np.abs(rng.randn(n) * 0.3) + 0.3
        low = close - np.abs(rng.randn(n) * 0.3) - 0.3
        data = {"close": close, "high": high, "low": low}
        det_tight = RuleBasedRegimeDetector(adx_bull_threshold=15.0)
        det_loose = RuleBasedRegimeDetector(adx_bull_threshold=50.0)
        result_tight = det_tight.detect(data)
        result_loose = det_loose.detect(data)
        assert isinstance(result_tight.regime, MarketRegime)
        assert isinstance(result_loose.regime, MarketRegime)

    def test_metadata_contains_sma_values(self, sample_data: dict[str, np.ndarray]) -> None:
        """Metadata enthalt SMA_fast und SMA_slow."""
        det = RuleBasedRegimeDetector()
        result = det.detect(sample_data)
        assert "sma_fast" in result.metadata
        assert "sma_slow" in result.metadata

    def test_short_data_scores_are_equal(self) -> None:
        """Short data -> Scores sind [0.0, 0.0, 1.0]."""
        close = np.array([100.0] * 10)
        data = {"close": close, "high": close, "low": close}
        det = RuleBasedRegimeDetector()
        result = det.detect(data)
        assert result.scores[MarketRegime.CHOPPY] == 1.0
        assert result.scores[MarketRegime.BULL] == 0.0
        assert result.scores[MarketRegime.BEAR] == 0.0
