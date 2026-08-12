"""Unit tests for the HistoricalAnalogyAgent."""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from numpy.typing import NDArray
from packages.agents import AgentType
from packages.agents.base import AgentConfig
from packages.agents.historical_analogy_agent import HistoricalAnalogyAgent
from packages.schemas.agent_report import (
    AgentReport,
    EvidenceReference,
    InvalidationCondition,
)


@pytest.fixture
def agent() -> HistoricalAnalogyAgent:
    return HistoricalAnalogyAgent()


@pytest.fixture
def trend_data() -> dict[str, NDArray[np.float64]]:
    """Generate an uptrend dataset with 200 points."""
    np.random.seed(42)
    n = 200
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.3) + np.arange(n) * 0.5
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.abs(np.random.randn(n) * 1000) + 500
    return {"close": close, "high": high, "low": low, "volume": volume}


@pytest.fixture
def minimal_data() -> dict[str, NDArray[np.float64]]:
    """Minimal valid dataset (min_window + 1 = 21 points)."""
    np.random.seed(99)
    close = np.linspace(100, 120, 21)
    return {
        "close": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "volume": np.ones(21) * 500,
    }


# ── AgentType enum ─────────────────────────────────────────────────────────

class TestAgentTypeHistoricalAnalogy:
    def test_historical_analogy_in_enum(self) -> None:
        assert hasattr(AgentType, "HISTORICAL_ANALOGY")
        assert AgentType.HISTORICAL_ANALOGY == "historical_analogy"

    def test_historical_analogy_is_str_enum(self) -> None:
        assert isinstance(AgentType.HISTORICAL_ANALOGY, str)


# ── AgentConfig ────────────────────────────────────────────────────────────

class TestHistoricalAnalogyAgentConfig:
    def test_default_config_agent_id(self) -> None:
        agent = HistoricalAnalogyAgent()
        assert agent.agent_id == "historical_analogy"

    def test_default_config_agent_type(self) -> None:
        agent = HistoricalAnalogyAgent()
        assert agent.config.agent_type == AgentType.HISTORICAL_ANALOGY

    def test_custom_config(self) -> None:
        config = AgentConfig(
            agent_id="custom_ha",
            agent_type=AgentType.HISTORICAL_ANALOGY,
            instrument="BTC/USDT",
            horizon="4h",
        )
        agent = HistoricalAnalogyAgent(config=config)
        assert agent.agent_id == "custom_ha"
        assert agent.config.instrument == "BTC/USDT"
        assert agent.config.horizon == "4h"


# ── Basic Analysis ────────────────────────────────────────────────────────

class TestHistoricalAnalogyAgentBasic:
    def test_produces_agent_report(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        assert isinstance(report, AgentReport)
        assert report.agent_id == "historical_analogy"

    def test_probabilities_sum_to_one(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        prob_sum = sum(report.probabilities.values())
        assert abs(prob_sum - 1.0) <= 0.0001

    def test_probabilities_have_required_keys(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        assert "up" in report.probabilities
        assert "down" in report.probabilities
        assert "range" in report.probabilities

    def test_evidence_present(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        assert len(report.evidence) >= 1

    def test_evidence_is_evidence_reference(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        for ev in report.evidence:
            assert isinstance(ev, EvidenceReference)

    def test_counter_evidence_required(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        assert len(report.counter_evidence) >= 1

    def test_invalidations_present(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        assert len(report.invalidations) >= 1

    def test_status_shadow(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        assert report.status.value == "shadow"

    def test_report_id_is_unique(self, agent, trend_data) -> None:
        r1 = agent.analyze(trend_data)
        r2 = agent.analyze(trend_data)
        assert r1.report_id != r2.report_id

    def test_sample_size_set(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        assert report.sample_size is not None
        assert isinstance(report.sample_size, int)
        assert report.sample_size > 0

    def test_raw_confidence_valid(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        assert 0.0 <= report.raw_confidence <= 0.9

    def test_hypothesis_non_empty(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        assert len(report.hypothesis) > 0
        assert "Historical analogy" in report.hypothesis


# ── DTW Distance ──────────────────────────────────────────────────────────

class TestDTWDistance:
    def test_identical_sequences(self) -> None:
        from packages.agents.historical_analogy_agent import _dtw_distance
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = _dtw_distance(seq, seq)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_different_sequences(self) -> None:
        from packages.agents.historical_analogy_agent import _dtw_distance
        seq1 = np.array([1.0, 2.0, 3.0])
        seq2 = np.array([1.0, 2.0, 4.0])
        dist = _dtw_distance(seq1, seq2)
        assert dist > 0.0

    def test_empty_sequences(self) -> None:
        from packages.agents.historical_analogy_agent import _dtw_distance
        dist = _dtw_distance(np.array([]), np.array([1.0]))
        assert np.isinf(dist)

    def test_scaled_sequence(self) -> None:
        from packages.agents.historical_analogy_agent import _dtw_distance
        seq1 = np.array([1.0, 2.0, 3.0])
        seq2 = np.array([2.0, 4.0, 6.0])
        dist = _dtw_distance(seq1, seq2)
        assert dist > 0.0


# ── Regime Detection ──────────────────────────────────────────────────────

class TestRegimeDetection:
    def test_bull_regime(self) -> None:
        from packages.agents.historical_analogy_agent import _detect_regime
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.025])
        regime = _detect_regime(returns)
        assert regime == "bull"

    def test_bear_regime(self) -> None:
        from packages.agents.historical_analogy_agent import _detect_regime
        returns = np.array([-0.01, -0.02, -0.015, -0.01, -0.025])
        regime = _detect_regime(returns)
        assert regime == "bear"

    def test_choppy_regime(self) -> None:
        from packages.agents.historical_analogy_agent import _detect_regime
        returns = np.array([0.01, -0.01, 0.005, -0.005, 0.002])
        regime = _detect_regime(returns)
        assert regime == "choppy"

    def test_short_returns(self) -> None:
        from packages.agents.historical_analogy_agent import _detect_regime
        returns = np.array([0.01, -0.01])
        regime = _detect_regime(returns)
        assert regime == "choppy"


# ── Data Validation ───────────────────────────────────────────────────────

class TestHistoricalAnalogyValidation:
    def test_missing_close_raises(self, agent) -> None:
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze({"high": np.array([1.0, 2.0]), "low": np.array([0.5, 1.5])})

    def test_too_few_points_raises(self, agent) -> None:
        data = {
            "close": np.array([100.0, 101.0]),
            "high": np.array([100.5, 101.5]),
            "low": np.array([99.5, 100.5]),
        }
        with pytest.raises(ValueError, match="Need at least"):
            agent.analyze(data)

    def test_valid_minimal_data(self, agent, minimal_data) -> None:
        report = agent.analyze(minimal_data)
        assert isinstance(report, AgentReport)


# ── Historical Data Override ──────────────────────────────────────────────

class TestHistoricalDataOverride:
    def test_uses_external_historical_data(self) -> None:
        np.random.seed(123)
        hist_close = np.cumsum(np.random.randn(300) * 0.3) + 100.0
        hist_high = hist_close + np.abs(np.random.randn(300) * 0.3)
        hist_low = hist_close - np.abs(np.random.randn(300) * 0.3)

        hist_data = {
            "close": hist_close,
            "high": hist_high,
            "low": hist_low,
            "volume": np.abs(np.random.randn(300) * 1000) + 500,
        }

        # Need >= 21 current data points for min_window check
        current_close = np.array([hist_close[-1] + i * 0.1 for i in range(25)])
        current_high = current_close + 0.5
        current_low = current_close - 0.5

        agent = HistoricalAnalogyAgent(historical_data=hist_data)
        data = {
            "close": current_close,
            "high": current_high,
            "low": current_low,
            "volume": np.ones(25) * 500.0,
        }

        report = agent.analyze(data)
        assert isinstance(report, AgentReport)
        assert len(report.evidence) >= 1


# ── Evidence Structure ────────────────────────────────────────────────────

class TestEvidenceStructure:
    def test_evidence_direction_values(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        for ev in report.evidence:
            assert ev.direction in ("positive", "negative", "neutral")
            assert 0.0 <= ev.relevance <= 1.0
            assert ev.feature
            assert ev.value

    def test_counter_evidence_direction(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        for ev in report.counter_evidence:
            assert ev.direction == "negative"

    def test_invalidations_structure(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        for inv in report.invalidations:
            assert isinstance(inv, InvalidationCondition)
            assert inv.condition
            assert inv.indicator
            assert inv.direction in ("above", "below")


# ── Probability Distribution ──────────────────────────────────────────────

class TestProbabilityDistribution:
    def test_uniform_distribution_empty(self) -> None:
        agent = HistoricalAnalogyAgent()
        # When no matches, should return uniform
        prob = agent._compute_prob([])
        assert prob == {"up": 0.33, "down": 0.33, "range": 0.34}

    def test_all_up(self) -> None:
        agent = HistoricalAnalogyAgent()
        matches = [
            {"outcome": "up", "similarity": 0.5},
            {"outcome": "up", "similarity": 0.6},
        ]
        prob = agent._compute_prob(matches)
        assert prob["up"] > prob["down"]
        assert prob["up"] > prob["range"]

    def test_all_down(self) -> None:
        agent = HistoricalAnalogyAgent()
        matches = [
            {"outcome": "down", "similarity": 0.5},
            {"outcome": "down", "similarity": 0.6},
        ]
        prob = agent._compute_prob(matches)
        assert prob["down"] > prob["up"]
        assert prob["down"] > prob["range"]

    def test_weighted_by_similarity(self) -> None:
        agent = HistoricalAnalogyAgent()
        matches = [
            {"outcome": "up", "similarity": 0.9},   # high weight
            {"outcome": "down", "similarity": 0.1},  # low weight
        ]
        prob = agent._compute_prob(matches)
        assert prob["up"] > prob["down"]


# ── Confidence Score ──────────────────────────────────────────────────────

class TestConfidenceScore:
    def test_confidence_for_empty_matches(self) -> None:
        agent = HistoricalAnalogyAgent()
        conf = agent._compute_confidence([])
        assert conf == 0.1

    def test_confidence_increases_with_matches(self) -> None:
        agent = HistoricalAnalogyAgent()
        matches_low = [{"outcome": "up", "similarity": 0.1}]
        matches_high = [
            {"outcome": "up", "similarity": 0.8},
            {"outcome": "down", "similarity": 0.7},
        ]
        conf_low = agent._compute_confidence(matches_low)
        conf_high = agent._compute_confidence(matches_high)
        assert conf_high >= conf_low

    def test_confidence_capped_at_max(self) -> None:
        agent = HistoricalAnalogyAgent()
        matches = [
            {"outcome": "up", "similarity": 1.0},
            {"outcome": "up", "similarity": 1.0},
            {"outcome": "down", "similarity": 1.0},
            {"outcome": "down", "similarity": 1.0},
            {"outcome": "up", "similarity": 1.0},
        ]
        conf = agent._compute_confidence(matches)
        assert conf <= 0.9


# ── Integration Tests ─────────────────────────────────────────────────────

class TestHistoricalAnalogyIntegration:
    def test_full_analysis_pipeline(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)

        # All required fields present
        assert report.agent_id == "historical_analogy"
        assert report.hypothesis
        assert report.probabilities
        assert report.evidence
        assert report.counter_evidence
        assert report.invalidations

        # Probabilities valid
        prob_sum = sum(report.probabilities.values())
        assert abs(prob_sum - 1.0) <= 0.0001

        # Minimum counts
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1
        assert len(report.invalidations) >= 1

    def test_report_as_of_is_datetime(self, agent, trend_data) -> None:
        report = agent.analyze(trend_data)
        assert isinstance(report.as_of, datetime.datetime)
