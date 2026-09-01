"""Tests for the uncertainty package."""

from __future__ import annotations

import pytest
from packages.uncertainty.confidence import (
    BayesianConfidence,
    bootstrap_confidence_interval,
)
from packages.uncertainty.elicitation import (
    prior_from_confidence,
    prior_from_historical,
)
from packages.uncertainty.entropy import (
    entropy_score,
    normalized_entropy,
)


class TestEntropyScore:
    """Tests for Shannon entropy."""

    def test_uniform_distribution_max_entropy(self):
        probs = {"up": 0.333, "down": 0.333, "range": 0.334}
        result = entropy_score(probs)
        # 3 categories → max entropy = log2(3) ≈ 1.585
        assert result > 1.5

    def test_certain_distribution_zero_entropy(self):
        probs = {"up": 1.0, "down": 0.0, "range": 0.0}
        result = entropy_score(probs)
        assert result == 0.0

    def test_bimodal_distribution(self):
        probs = {"up": 0.5, "down": 0.5, "range": 0.0}
        result = entropy_score(probs)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_empty_dict(self):
        result = entropy_score({})
        assert result == 0.0


class TestNormalizedEntropy:
    """Tests for normalized entropy."""

    def test_uniform_distribution_norm_entropy_1(self):
        probs = {"up": 0.333, "down": 0.333, "range": 0.334}
        result = normalized_entropy(probs)
        assert result > 0.99

    def test_certain_distribution_norm_entropy_0(self):
        probs = {"up": 1.0, "down": 0.0, "range": 0.0}
        result = normalized_entropy(probs)
        assert result == 0.0

    def test_two_categories(self):
        probs = {"up": 0.5, "down": 0.5}
        result = normalized_entropy(probs)
        assert result > 0.99

    def test_in_range_0_1(self):
        probs = {"up": 0.6, "down": 0.3, "range": 0.1}
        result = normalized_entropy(probs)
        assert 0.0 < result < 1.0


class TestBayesianConfidence:
    """Tests for Bayesian confidence intervals."""

    def test_from_alpha_beta_basic(self):
        bc = BayesianConfidence.from_alpha_beta(alpha=10, beta=2, confidence_level=0.95)
        assert 0.0 <= bc.mean <= 1.0
        assert bc.lower <= bc.mean <= bc.upper
        assert bc.width > 0

    def test_symmetric_prior(self):
        bc = BayesianConfidence.from_alpha_beta(alpha=5, beta=5)
        assert abs(bc.mean - 0.5) < 0.1

    def test_zero_alpha(self):
        bc = BayesianConfidence.from_alpha_beta(alpha=0, beta=5)
        assert bc.mean == pytest.approx(0.5, abs=0.1)

    def test_invalid_returns_fallback(self):
        bc = BayesianConfidence.from_alpha_beta(alpha=-1, beta=-1)
        assert bc.width == 1.0


class TestBootstrapConfidence:
    """Tests for bootstrap confidence intervals."""

    def test_basic_bootstrap(self):
        samples = [0.7, 0.8, 0.75, 0.85, 0.65]
        bc = bootstrap_confidence_interval(samples)
        assert 0.0 <= bc.mean <= 1.0
        assert bc.width > 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            bootstrap_confidence_interval([])

    def test_uniform_samples_zero_width(self):
        samples = [0.5, 0.5, 0.5, 0.5]
        bc = bootstrap_confidence_interval(samples)
        assert bc.width == 0.0


class TestPriorFromConfidence:
    """Tests for elicitation from confidence."""

    def test_high_confidence(self):
        alpha, beta = prior_from_confidence(0.9, strength=10.0)
        assert alpha > beta
        assert alpha > 1.0
        assert beta > 1.0

    def test_low_confidence(self):
        alpha, beta = prior_from_confidence(0.1, strength=10.0)
        assert beta > alpha

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            prior_from_confidence(1.5)
        with pytest.raises(ValueError):
            prior_from_confidence(-0.1)

    def test_zero_strength_raises(self):
        with pytest.raises(ValueError):
            prior_from_confidence(0.5, strength=0)


class TestPriorFromHistorical:
    """Tests for elicitation from historical data."""

    def test_high_accuracy_prior(self):
        alpha, beta = prior_from_historical(0.8, 50)
        assert alpha > beta

    def test_low_accuracy_prior(self):
        alpha, beta = prior_from_historical(0.2, 50)
        assert beta > alpha

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            prior_from_historical(1.5, 10)

    def test_zero_observations_raises(self):
        with pytest.raises(ValueError):
            prior_from_historical(0.5, 0)
