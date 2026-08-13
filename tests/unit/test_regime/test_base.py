"""Tests fur packages.regime.base — MarketRegime, RegimeResult, BaseRegimeDetector."""

from __future__ import annotations

import numpy as np
import pytest
from packages.regime.base import BaseRegimeDetector, MarketRegime, RegimeResult


class TestMarketRegime:
    """Testet MarketRegime-Enum."""

    def test_all_regimes(self) -> None:
        assert MarketRegime.BULL == "bull"
        assert MarketRegime.BEAR == "bear"
        assert MarketRegime.CHOPPY == "choppy"

    def test_regime_values_are_strings(self) -> None:
        assert isinstance(MarketRegime.BULL, str)
        assert isinstance(MarketRegime.BEAR, str)
        assert isinstance(MarketRegime.CHOPPY, str)

    def test_regime_can_compare(self) -> None:
        assert MarketRegime.BULL == "bull"
        assert MarketRegime.BEAR == "bear"
        assert MarketRegime.CHOPPY == "choppy"

    def test_no_extra_regimes(self) -> None:
        """Nur die drei definierten Regimes existieren."""
        members = list(MarketRegime)
        assert len(members) == 3


class TestRegimeResult:
    """Testet RegimeResult."""

    def test_result_construction(self) -> None:
        result = RegimeResult(
            regime=MarketRegime.BULL,
            confidence=0.75,
            scores={
                MarketRegime.BULL: 0.75,
                MarketRegime.BEAR: 0.15,
                MarketRegime.CHOPPY: 0.10,
            },
        )
        assert result.regime == MarketRegime.BULL
        assert result.confidence == 0.75
        assert len(result.scores) == 3
        assert result.metadata == {}

    def test_result_with_metadata(self) -> None:
        result = RegimeResult(
            regime=MarketRegime.CHOPPY,
            confidence=0.0,
            scores={
                MarketRegime.BULL: 0.0,
                MarketRegime.BEAR: 0.0,
                MarketRegime.CHOPPY: 1.0,
            },
            metadata={"n_samples": 42, "model": "rule_based"},
        )
        assert result.regime == MarketRegime.CHOPPY
        assert result.metadata["n_samples"] == 42
        assert result.metadata["model"] == "rule_based"

    def test_result_metadata_defaults_empty(self) -> None:
        """Standard-Metadata ist leer."""
        result = RegimeResult(
            regime=MarketRegime.BULL,
            confidence=0.5,
            scores={
                MarketRegime.BULL: 0.5,
                MarketRegime.BEAR: 0.3,
                MarketRegime.CHOPPY: 0.2,
            },
        )
        assert result.metadata == {}

    def test_result_confidence_range(self) -> None:
        """Confidence kann in jedem Bereich liegen."""
        result = RegimeResult(
            regime=MarketRegime.BULL,
            confidence=1.0,
            scores={MarketRegime.BULL: 1.0, MarketRegime.BEAR: 0.0, MarketRegime.CHOPPY: 0.0},
        )
        assert result.confidence == 1.0

    def test_result_all_three_scores(self) -> None:
        """Scores muss alle drei Regimes enthalten."""
        result = RegimeResult(
            regime=MarketRegime.BULL,
            confidence=0.6,
            scores={
                MarketRegime.BULL: 0.6,
                MarketRegime.BEAR: 0.25,
                MarketRegime.CHOPPY: 0.15,
            },
        )
        for regime in MarketRegime:
            assert regime in result.scores

    def test_result_regime_matches_best_score(self) -> None:
        """Regime stimmt ubers maximalem Score uberein."""
        best = MarketRegime.BEAR
        scores = {
            MarketRegime.BULL: 0.2,
            best: 0.5,
            MarketRegime.CHOPPY: 0.3,
        }
        result = RegimeResult(
            regime=best,
            confidence=scores[best],
            scores=scores,
        )
        assert result.regime == best


class TestBaseRegimeDetector:
    """Testet BaseRegimeDetector."""

    def test_is_abstract(self) -> None:
        """BaseRegimeDetector ist abstrakt."""
        with pytest.raises(TypeError):
            BaseRegimeDetector()

    def test_detect_is_abstract(self) -> None:
        """detect() ist abstrakt."""
        # Eine Unterklasse ohne implementierung wirft TypeError
        class Incomplete(BaseRegimeDetector):
            name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_detector_has_name(self) -> None:
        """Regime-Detektoren mussen name haben."""
        from packages.regime.rules import RuleBasedRegimeDetector

        det = RuleBasedRegimeDetector()
        assert det.name == "rule_based"
        assert isinstance(det.name, str)
        assert len(det.name) > 0

    def test_detect_requires_data(self) -> None:
        """detect() erwartet Daten-Dict."""
        from packages.regime.rules import RuleBasedRegimeDetector

        det = RuleBasedRegimeDetector()
        # Kurze Daten -> CHOPPY
        result = det.detect({
            "close": np.array([100.0] * 5),
            "high": np.array([100.0] * 5),
            "low": np.array([100.0] * 5),
        })
        assert isinstance(result, RegimeResult)
