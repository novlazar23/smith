"""Tests for packages.regime.hmm — HiddenMarkovModel (GMM-basiert)."""

from __future__ import annotations

import numpy as np
import pytest
from packages.regime.base import MarketRegime, RegimeResult
from packages.regime.hmm import HiddenMarkovModel


class TestHiddenMarkovModelInit:
    """Testet HiddenMarkovModel-Konstruktion."""

    def test_default_init(self) -> None:
        hmm = HiddenMarkovModel()
        assert hmm.n_components == 3
        assert hmm.n_init == 3
        assert hmm.covariance_type == "full"
        assert hmm.name == "hmm"
        assert hmm._model is None

    def test_custom_n_components(self) -> None:
        hmm = HiddenMarkovModel(n_components=2)
        assert hmm.n_components == 2

    def test_custom_covariance_type(self) -> None:
        hmm = HiddenMarkovModel(covariance_type="diag")
        assert hmm.covariance_type == "diag"

    def test_custom_n_init(self) -> None:
        hmm = HiddenMarkovModel(n_init=5)
        assert hmm.n_init == 5


class TestHiddenMarkovModelDetect:
    """Testet detect()."""

    @pytest.fixture
    def bull_market_data(self) -> dict[str, np.ndarray]:
        """Bull-Markt-Daten."""
        n = 100
        rng = np.random.RandomState(42)
        close = np.array([100.0 + i * 0.8 + rng.randn() * 0.5 for i in range(n)])
        high = close + np.abs(rng.randn(n) * 0.3) + 0.5
        low = close - np.abs(rng.randn(n) * 0.3) - 0.5
        return {"close": close, "high": high, "low": low}

    @pytest.fixture
    def bear_market_data(self) -> dict[str, np.ndarray]:
        """Bear-Markt-Daten."""
        n = 100
        rng = np.random.RandomState(100)
        close = np.array([200.0 - i * 0.8 + rng.randn() * 0.5 for i in range(n)])
        high = close + np.abs(rng.randn(n) * 0.3) + 0.5
        low = close - np.abs(rng.randn(n) * 0.3) - 0.5
        return {"close": close, "high": high, "low": low}

    @pytest.fixture
    def chopy_market_data(self) -> dict[str, np.ndarray]:
        """Choppy-Markt-Daten."""
        n = 100
        rng = np.random.RandomState(200)
        close = np.cumsum(rng.randn(n) * 0.1) + 100
        high = close + np.abs(rng.randn(n) * 0.05) + 0.05
        low = close - np.abs(rng.randn(n) * 0.05) - 0.05
        return {"close": close, "high": high, "low": low}

    def test_detect_returns_regime_result(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """detect gibt RegimeResult zuruck."""
        hmm = HiddenMarkovModel()
        result = hmm.detect(bull_market_data)
        assert isinstance(result, RegimeResult)

    def test_detect_returns_valid_regime(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """Regime ist eines der drei definierten Regimes."""
        hmm = HiddenMarkovModel()
        result = hmm.detect(bull_market_data)
        assert result.regime in (MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.CHOPPY)

    def test_detect_confidence_in_range(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """Confidence liegt zwischen 0 und 1."""
        hmm = HiddenMarkovModel()
        result = hmm.detect(bull_market_data)
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_scores_all_three(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """Scores enthalten alle drei Regimes."""
        hmm = HiddenMarkovModel()
        result = hmm.detect(bull_market_data)
        for regime in MarketRegime:
            assert regime in result.scores

    def test_detect_scores_sum_to_one(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """Scores summieren sich zu 1."""
        hmm = HiddenMarkovModel()
        result = hmm.detect(bull_market_data)
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 1e-6

    def test_detect_scores_all_positive(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """Alle Scores sind nicht-negativ."""
        hmm = HiddenMarkovModel()
        result = hmm.detect(bull_market_data)
        for v in result.scores.values():
            assert v >= 0.0

    def test_detect_regime_best_score(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """Regime ist der Score mit der hochsten Wahrscheinlichkeit."""
        hmm = HiddenMarkovModel()
        result = hmm.detect(bull_market_data)
        best = max(result.scores, key=result.scores.get)
        assert result.regime == best

    def test_detect_confidence_is_best_score(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """Confidence entspricht dem besten Score."""
        hmm = HiddenMarkovModel()
        result = hmm.detect(bull_market_data)
        best_score = max(result.scores.values())
        assert abs(result.confidence - best_score) < 1e-6

    def test_detect_metadata_present(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """Metadata enthalt n_components und cluster_rsi_means."""
        hmm = HiddenMarkovModel()
        result = hmm.detect(bull_market_data)
        assert "n_components" in result.metadata
        assert "cluster_rsi_means" in result.metadata
        assert result.metadata["n_components"] == 3

    def test_detect_trains_model(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """Modell wird nach detect() trainiert."""
        hmm = HiddenMarkovModel()
        assert hmm._model is None
        hmm.detect(bull_market_data)
        assert hmm._model is not None

    def test_detect_bull_market(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """Bull-Markt sollte hohes BULL-Score haben."""
        hmm = HiddenMarkovModel()
        result = hmm.detect(bull_market_data)
        # Bull-Market sollte mindestens ein Regime sein
        assert result.regime in (MarketRegime.BULL, MarketRegime.CHOPPY, MarketRegime.BEAR)

    def test_detect_choppy_market(self, chopy_market_data: dict[str, np.ndarray]) -> None:
        """Choppy-Markt wird erkannt."""
        hmm = HiddenMarkovModel()
        result = hmm.detect(chopy_market_data)
        assert result.regime in (MarketRegime.BULL, MarketRegime.CHOPPY, MarketRegime.BEAR)

    def test_different_n_components(self, bull_market_data: dict[str, np.ndarray]) -> None:
        """Verschiedene n_components liefern verschiedene Ergebnisse."""
        hmm_2 = HiddenMarkovModel(n_components=2)
        hmm_3 = HiddenMarkovModel(n_components=3)
        result_2 = hmm_2.detect(bull_market_data)
        result_3 = hmm_3.detect(bull_market_data)
        assert isinstance(result_2, RegimeResult)
        assert isinstance(result_3, RegimeResult)


class TestBuildFeatures:
    """Testet _build_features (interne Methode)."""

    def test_features_shape(self) -> None:
        """Features haben 3 Spalten [RSI, SMA-Delta, TrueRange]."""
        n = 100
        rng = np.random.RandomState(42)
        close = np.array([100.0 + i * 0.5 + rng.randn() * 0.3 for i in range(n)])
        high = close + 0.5
        low = close - 0.5
        data = {"close": close, "high": high, "low": low}
        hmm = HiddenMarkovModel()
        features = hmm._build_features(data)
        assert features.shape[1] == 3

    def test_features_length_matches_close(self) -> None:
        """Features haben gleiche Lange wie close."""
        n = 100
        close = np.cumsum(np.random.RandomState(42).randn(n) * 0.5) + 100
        high = close + 0.5
        low = close - 0.5
        data = {"close": close, "high": high, "low": low}
        hmm = HiddenMarkovModel()
        features = hmm._build_features(data)
        assert len(features) == n

    def test_features_no_nans(self) -> None:
        """Features enthalten keine NaNs."""
        n = 100
        close = np.cumsum(np.random.RandomState(42).randn(n) * 0.5) + 100
        high = close + 0.5
        low = close - 0.5
        data = {"close": close, "high": high, "low": low}
        hmm = HiddenMarkovModel()
        features = hmm._build_features(data)
        assert not np.any(np.isnan(features))

    def test_features_without_optional_keys(self) -> None:
        """Features arbeiten auch ohne 'high' und 'low'."""
        n = 100
        close = np.cumsum(np.random.RandomState(42).randn(n) * 0.5) + 100
        data = {"close": close}
        hmm = HiddenMarkovModel()
        features = hmm._build_features(data)
        assert features.shape[1] == 3
        assert not np.any(np.isnan(features))
