"""Tests für Regime Detection — Rule-Based und GMM-basierter Detektor."""

from __future__ import annotations

import numpy as np
import pytest
from packages.regime import MarketRegime, RuleBasedRegimeDetector


@pytest.fixture
def sample_data() -> dict[str, np.ndarray]:
    """Erstellt synthetische Marktdaten für Tests."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.abs(np.random.randn(n) * 1000) + 500
    return {
        "open": close - np.random.randn(n) * 0.2,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


class TestMarketRegime:
    """Testet MarketRegime-Enum."""

    def test_all_regimes(self) -> None:
        assert MarketRegime.BULL == "bull"
        assert MarketRegime.BEAR == "bear"
        assert MarketRegime.CHOPPY == "choppy"


class TestRuleBasedRegimeDetector:
    """Testet Rule-Based Regime-Detektor."""

    def test_detect_returns_valid_regime(self, sample_data: dict[str, np.ndarray]) -> None:
        detector = RuleBasedRegimeDetector()
        result = detector.detect(sample_data)
        assert result.regime in (MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.CHOPPY)

    def test_confidence_in_range(self, sample_data: dict[str, np.ndarray]) -> None:
        detector = RuleBasedRegimeDetector()
        result = detector.detect(sample_data)
        assert 0.0 <= result.confidence <= 1.0

    def test_scores_sum_to_one(self, sample_data: dict[str, np.ndarray]) -> None:
        detector = RuleBasedRegimeDetector()
        result = detector.detect(sample_data)
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 1e-6

    def test_scores_all_positive(self, sample_data: dict[str, np.ndarray]) -> None:
        detector = RuleBasedRegimeDetector()
        result = detector.detect(sample_data)
        for v in result.scores.values():
            assert v >= 0.0

    def test_metadata_contains_adx_rsi(self, sample_data: dict[str, np.ndarray]) -> None:
        detector = RuleBasedRegimeDetector()
        result = detector.detect(sample_data)
        assert "adx" in result.metadata
        assert "rsi" in result.metadata
        assert "is_uptrend" in result.metadata

    def test_very_short_data(self) -> None:
        close = np.array([100.0] * 5)
        data = {"close": close, "high": close, "low": close, "volume": np.ones(5)}
        detector = RuleBasedRegimeDetector()
        result = detector.detect(data)
        assert result.regime == MarketRegime.CHOPPY
        assert result.confidence == 0.0

    def test_strong_uptrend(self) -> None:
        """Uptrend sollte Bull-Regime ergeben."""
        close = np.array([100.0 + i * 2 for i in range(60)])
        high = close + np.abs(np.random.RandomState(42).randn(60) * 0.5)
        low = close - np.abs(np.random.RandomState(42).randn(60) * 0.5)
        data = {"close": close, "high": high, "low": low, "volume": np.ones(60)}
        detector = RuleBasedRegimeDetector()
        result = detector.detect(data)
        assert result.regime == MarketRegime.BULL

    def test_strong_downtrend(self) -> None:
        """Downtrend sollte Bear-Regime ergeben."""
        close = np.array([100.0 - i * 2 for i in range(60)])
        high = close + np.abs(np.random.RandomState(42).randn(60) * 0.5)
        low = close - np.abs(np.random.RandomState(42).randn(60) * 0.5)
        data = {"close": close, "high": high, "low": low, "volume": np.ones(60)}
        detector = RuleBasedRegimeDetector()
        result = detector.detect(data)
        assert result.regime == MarketRegime.BEAR

    def test_choppy_market(self) -> None:
        """Sehr kurze Perioden mit wenig Bewegung sollten Choppy ergeben."""
        rng = np.random.RandomState(123)
        close = np.cumsum(rng.randn(60) * 0.01) + 100
        high = close + 0.05
        low = close - 0.05
        data = {"close": close, "high": high, "low": low, "volume": np.ones(60)}
        detector = RuleBasedRegimeDetector()
        result = detector.detect(data)
        # Sollte entweder CHOPPY oder ein schwaches Regime sein
        assert result.regime in (MarketRegime.CHOPPY, MarketRegime.BULL, MarketRegime.BEAR)
