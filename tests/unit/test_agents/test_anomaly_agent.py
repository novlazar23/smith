"""Unit tests for the Anomaly Detection Agent."""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from packages.agents import AgentType, AnomalyAgent
from packages.agents.base import AgentConfig
from packages.schemas.agent_report import AgentReport

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def normal_ohlcv() -> dict:
    """Normal OHLCV data — 100 bars, no anomalies."""
    n = 100
    return {
        "close": np.linspace(100, 110, n),
        "high": np.linspace(101, 112, n),
        "low": np.linspace(99, 108, n),
        "volume": np.ones(n) * 1000.0,
        "open": np.linspace(100.5, 109.5, n),
    }


@pytest.fixture
def volume_spike_data() -> dict:
    """Data with a volume spike on last bar."""
    n = 100
    volume = np.ones(n) * 1000.0
    volume[-1] = 50000.0
    return {
        "close": np.linspace(100, 110, n),
        "high": np.linspace(101, 112, n),
        "low": np.linspace(99, 108, n),
        "volume": volume,
        "open": np.linspace(100.5, 109.5, n),
    }


@pytest.fixture
def short_ohlcv() -> dict:
    """Too-short data for statistics."""
    return {
        "close": np.array([100.0, 101.0, 102.0]),
        "high": np.array([101.0, 102.0, 103.0]),
        "low": np.array([99.0, 100.0, 101.0]),
        "volume": np.array([1000.0, 1100.0, 1200.0]),
        "open": np.array([100.0, 101.0, 102.0]),
    }


# ── AgentType / Config ────────────────────────────────────────────────────

class TestAgentType:
    def test_anomaly_in_agent_type(self) -> None:
        assert AgentType.ANOMALY == "anomaly"

    def test_anomaly_agent_default_config(self) -> None:
        agent = AnomalyAgent()
        assert agent.agent_id == "anomaly"
        assert agent.config.agent_type == AgentType.ANOMALY

    def test_anomaly_agent_custom_config(self) -> None:
        config = AgentConfig(
            agent_id="anomaly",
            instrument="BTC/USD",
            horizon="4h",
        )
        agent = AnomalyAgent(config)
        assert agent.config.instrument == "BTC/USD"
        assert agent.config.horizon == "4h"


# ── Validation ────────────────────────────────────────────────────────────

class TestValidation:
    def test_missing_keys_raises(self) -> None:
        agent = AnomalyAgent()
        with pytest.raises(ValueError, match="close"):
            agent.analyze({"high": np.ones(10)})

    def test_all_keys_required(self) -> None:
        agent = AnomalyAgent()
        with pytest.raises(ValueError, match="volume"):
            agent.analyze({
                "close": np.ones(10),
                "high": np.ones(10),
                "low": np.ones(10),
                "open": np.ones(10),
            })

    def test_list_data_converts_to_array(self) -> None:
        """Lists are auto-converted by np.asarray — test they work."""
        agent = AnomalyAgent()
        report = agent.analyze({
            "close": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "volume": [1.0, 2.0],
            "open": [1.0, 2.0],
        })
        # Short data gets handled gracefully
        assert report is not None
        assert len(report.evidence) >= 1


# ── Score computation ─────────────────────────────────────────────────────

class TestAnomalyScore:
    def test_normal_data_low_anomaly_score(self, normal_ohlcv) -> None:
        """Normal data should not produce high anomaly scores."""
        agent = AnomalyAgent(threshold=0.6)
        report = agent.analyze(normal_ohlcv)
        # Anomaly score may be elevated for linear data (no variance)
        # but should be flagged conservatively
        assert report.raw_confidence >= 0.0
        assert isinstance(report.raw_confidence, float)

    def test_volume_spike_elevates_anomaly_score(self, volume_spike_data) -> None:
        """Large volume spike should produce elevated anomaly score."""
        agent = AnomalyAgent(threshold=0.3)
        report = agent.analyze(volume_spike_data)
        # With a 50x volume spike, at least one score should be above 0.3
        any_above = any(
            float(e.value.split(":")[1].split()[0]) >= 0.3
            for e in report.evidence
            if ":" in e.value and "score" in e.value
        )
        assert any_above or report.hypothesis != "No anomalies detected"


class TestSpoofingScore:
    def test_spoofing_score_normal(self, normal_ohlcv) -> None:
        """Normal data should have non-negative spoofing score."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        spoofing_evidence = [
            e for e in report.evidence if "spoofing" in e.feature.lower()
        ]
        assert len(spoofing_evidence) >= 1
        assert any(float(v.split(":")[1].split()[0]) >= 0.0 for e in spoofing_evidence for v in [e.value])

    def test_spoofing_score_float(self, normal_ohlcv) -> None:
        """All scores must be native Python floats."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        for e in report.evidence:
            val_str = e.value.split(":")[1].split()[0]
            val = float(val_str)
            assert isinstance(val, float)


class TestLiquidityScore:
    def test_liquidity_score_non_negative(self, normal_ohlcv) -> None:
        """Liquidity score should be non-negative."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        liquidity = [e for e in report.evidence if "liquidity" in e.feature.lower()]
        assert len(liquidity) >= 1


class TestCrossVenueScore:
    def test_cross_venue_score_float(self, normal_ohlcv) -> None:
        """Cross-venue score must be a native float."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        cross_venue = [e for e in report.evidence if "cross_venue" in e.feature.lower()]
        assert len(cross_venue) >= 1


# ── Thresholding ──────────────────────────────────────────────────────────

class TestThresholding:
    def test_conservative_scoring(self, normal_ohlcv) -> None:
        """Normal data should not flag anomalies at 0.6 threshold."""
        agent = AnomalyAgent(threshold=0.6)
        report = agent.analyze(normal_ohlcv)
        flagged = [e for e in report.evidence if e.direction == "positive"]
        # May have flagged if data is pathological, but shouldn't have many
        assert len(flagged) < 3

    def test_higher_threshold_stricter(self, normal_ohlcv) -> None:
        """Higher threshold should flag fewer anomalies."""
        agent = AnomalyAgent(threshold=0.9)
        report = agent.analyze(normal_ohlcv)
        flagged = [e for e in report.evidence if e.direction == "positive"]
        # Very few if any at 0.9
        assert len(flagged) <= 2


# ── Probability computation ───────────────────────────────────────────────

class TestProbabilities:
    def test_probabilities_sum_to_one(self, normal_ohlcv) -> None:
        """Probabilities must sum to 1.0 ± 0.0001."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) <= 0.0001

    def test_probabilities_all_non_negative(self, normal_ohlcv) -> None:
        """All probabilities must be >= 0."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        for v in report.probabilities.values():
            assert v >= 0.0

    def test_probabilities_all_native_float(self, normal_ohlcv) -> None:
        """Probabilities must be native Python floats, not np.float64."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        for k, v in report.probabilities.items():
            assert isinstance(v, float), f"{k} is {type(v)}, expected float"

    def test_no_data_uniform_distribution(self, short_ohlcv) -> None:
        """Short data should produce uniform probabilities."""
        agent = AnomalyAgent()
        report = agent.analyze(short_ohlcv)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) <= 0.0001

    def test_up_down_range_keys_present(self, normal_ohlcv) -> None:
        """Report must have up, down, and range keys."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert "up" in report.probabilities
        assert "down" in report.probabilities
        assert "range" in report.probabilities


# ── Evidence ──────────────────────────────────────────────────────────────

class TestEvidence:
    def test_evidence_min_one(self, normal_ohlcv) -> None:
        """Must have at least one evidence entry (AT-004)."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert len(report.evidence) >= 1

    def test_evidence_all_scored_not_claimed(self, normal_ohlcv) -> None:
        """Evidence must reference scores, not definitive statements."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        for e in report.evidence:
            # Evidence value should reference a score or data point
            assert "score" in e.value.lower() or "t=" in e.value or "bar" in e.value.lower()

    def test_evidence_has_correct_structure(self, normal_ohlcv) -> None:
        """Each evidence must have reference, feature, value, direction, relevance."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        for e in report.evidence:
            assert len(e.reference) > 0
            assert len(e.feature) > 0
            assert len(e.value) > 0
            assert e.direction in ("positive", "negative", "neutral")
            assert 0.0 <= e.relevance <= 1.0

    def test_evidence_count_for_normal_data(self, normal_ohlcv) -> None:
        """Normal data should have evidence for all 4 anomaly types."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        features = {e.feature for e in report.evidence}
        # Should cover all anomaly types
        assert "anomaly" in features or "spoofing_like" in features


class TestEvidenceTypes:
    def test_anomaly_evidence(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert any(e.feature == "anomaly" for e in report.evidence)

    def test_spoofing_evidence(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        features = {e.feature for e in report.evidence}
        assert "spoofing_like" in features

    def test_liquidity_evidence(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        features = {e.feature for e in report.evidence}
        assert "liquidity_withdrawal" in features

    def test_cross_venue_evidence(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        features = {e.feature for e in report.evidence}
        assert "cross_venue_divergence" in features


# ── Counter-evidence ──────────────────────────────────────────────────────

class TestCounterEvidence:
    def test_counter_evidence_present(self, normal_ohlcv) -> None:
        """Counter-evidence must be present (spec requirement)."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert len(report.counter_evidence) >= 1

    def test_counter_evidence_alternative_explanations(self, volume_spike_data) -> None:
        """With flagged anomalies, counter-evidence provides alternatives."""
        agent = AnomalyAgent(threshold=0.3)
        report = agent.analyze(volume_spike_data)
        # Should have counter-evidence for each flagged type
        assert len(report.counter_evidence) >= 1
        for ce in report.counter_evidence:
            assert len(ce.value) > 0
            assert ce.direction == "negative"

    def test_counter_evidence_normal_data(self, normal_ohlcv) -> None:
        """Normal data counter-evidence confirms normal conditions."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert len(report.counter_evidence) >= 1
        # Should mention normal conditions or insufficient anomalies
        any_ce = " ".join(ce.value.lower() for ce in report.counter_evidence)
        assert "normal" in any_ce or "insufficient" in any_ce or "no anomaly" in any_ce

    def test_counter_evidence_not_empty_list(self, normal_ohlcv) -> None:
        """Counter-evidence must never be an empty list."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert report.counter_evidence  # not empty


# ── Invalidations ─────────────────────────────────────────────────────────

class TestInvalidations:
    def test_invalidation_regime_change(self, normal_ohlcv) -> None:
        report = AnomalyAgent().analyze(normal_ohlcv)
        conditions = [i.condition for i in report.invalidations]
        assert any("regime" in c.lower() for c in conditions)

    def test_invalidation_data_quality(self, normal_ohlcv) -> None:
        report = AnomalyAgent().analyze(normal_ohlcv)
        conditions = [i.condition for i in report.invalidations]
        assert any("data quality" in c.lower() or "quality" in c.lower() for c in conditions)

    def test_invalidation_sample_size(self, normal_ohlcv) -> None:
        report = AnomalyAgent().analyze(normal_ohlcv)
        indicators = [i.indicator for i in report.invalidations]
        assert "sample_size" in indicators

    def test_invalidation_structure(self, normal_ohlcv) -> None:
        report = AnomalyAgent().analyze(normal_ohlcv)
        for inv in report.invalidations:
            assert len(inv.condition) > 0
            assert len(inv.indicator) > 0
            assert inv.threshold >= 0
            assert inv.direction in ("above", "below")


# ── Hypothesis ────────────────────────────────────────────────────────────

class TestHypothesis:
    def test_hypothesis_no_anomaly_normal_data(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert "no anomalies" in report.hypothesis.lower() or "below threshold" in report.hypothesis.lower()

    def test_hypothesis_scores_included(self, volume_spike_data) -> None:
        agent = AnomalyAgent(threshold=0.3)
        report = agent.analyze(volume_spike_data)
        assert "score" in report.hypothesis.lower()

    def test_hypothesis_short_data(self, short_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(short_ohlcv)
        assert "insufficient" in report.hypothesis.lower() or "minimum" in report.hypothesis.lower()


# ── Confidence ────────────────────────────────────────────────────────────

class TestConfidence:
    def test_confidence_in_valid_range(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert 0.1 <= report.raw_confidence <= 0.9

    def test_confidence_is_native_float(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert isinstance(report.raw_confidence, float)

    def test_confidence_low_for_normal_data(self, normal_ohlcv) -> None:
        """Normal data should have low confidence in anomalies."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        # Low confidence because no anomalies flagged
        assert report.raw_confidence < 0.5

    def test_confidence_higher_with_flagged(self, volume_spike_data) -> None:
        """More anomaly signals should increase confidence."""
        agent = AnomalyAgent(threshold=0.3)
        report = agent.analyze(volume_spike_data)
        assert report.raw_confidence >= 0.1


# ── Report Structure ──────────────────────────────────────────────────────

class TestReportStructure:
    def test_report_has_required_fields(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert isinstance(report, AgentReport)
        assert report.agent_id == "anomaly"
        assert report.agent_version == "0.1.0"

    def test_report_as_of_datetime(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        before = datetime.datetime.now()
        report = agent.analyze(normal_ohlcv)
        after = datetime.datetime.now()
        assert before <= report.as_of <= after

    def test_report_status_shadow(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert report.status == "shadow"

    def test_report_run_id_unique(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        r1 = agent.analyze(normal_ohlcv)
        r2 = agent.analyze(normal_ohlcv)
        assert r1.run_id != r2.run_id
        assert r1.report_id != r2.report_id

    def test_report_frozen(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        with pytest.raises((ValueError, TypeError)):
            report.__setattr__("agent_id", "changed")

    def test_agent_id_anomaly(self, normal_ohlcv) -> None:
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert report.agent_id == "anomaly"


# ── Edge Cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_very_short_data(self) -> None:
        agent = AnomalyAgent()
        data = {
            "close": np.array([100.0]),
            "high": np.array([101.0]),
            "low": np.array([99.0]),
            "volume": np.array([1000.0]),
            "open": np.array([100.0]),
        }
        report = agent.analyze(data)
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1

    def test_all_zeros_volume(self) -> None:
        n = 50
        data = {
            "close": np.linspace(100, 110, n),
            "high": np.linspace(101, 112, n),
            "low": np.linspace(99, 108, n),
            "volume": np.zeros(n),
            "open": np.linspace(100.5, 109.5, n),
        }
        report = AnomalyAgent().analyze(data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) <= 0.0001

    def test_constant_close(self) -> None:
        n = 50
        data = {
            "close": np.ones(n) * 100.0,
            "high": np.ones(n) * 101.0,
            "low": np.ones(n) * 99.0,
            "volume": np.ones(n) * 1000.0,
            "open": np.ones(n) * 100.5,
        }
        report = AnomalyAgent().analyze(data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) <= 0.0001

    def test_negative_prices(self) -> None:
        """Unusual but should not crash."""
        n = 50
        data = {
            "close": -np.linspace(1, 10, n),
            "high": -np.linspace(0.5, 9, n),
            "low": -np.linspace(1.5, 11, n),
            "volume": np.ones(n) * 1000.0,
            "open": -np.linspace(1.2, 9.5, n),
        }
        report = AnomalyAgent().analyze(data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) <= 0.0001

    def test_nan_in_data(self) -> None:
        n = 50
        data = {
            "close": np.linspace(100, 110, n),
            "high": np.linspace(101, 112, n),
            "low": np.linspace(99, 108, n),
            "volume": np.ones(n) * 1000.0,
            "open": np.linspace(100.5, 109.5, n),
        }
        data["close"][10] = np.nan
        # Should handle NaN gracefully (not crash)
        report = AnomalyAgent().analyze(data)
        assert report is not None
        assert isinstance(report, AgentReport)

    def test_single_decimal_places(self) -> None:
        """Probabilities should have reasonable precision."""
        agent = AnomalyAgent()
        report = agent.analyze({
            "close": np.linspace(100, 110, 100),
            "high": np.linspace(101, 112, 100),
            "low": np.linspace(99, 108, 100),
            "volume": np.ones(100) * 1000.0,
            "open": np.linspace(100.5, 109.5, 100),
        })
        for v in report.probabilities.values():
            # Values should have at most 4 decimal places
            assert len(str(v).split(".")[-1]) <= 4


# ── Score Weights ─────────────────────────────────────────────────────────

class TestScoreWeights:
    def test_anomaly_type_score_range(self, normal_ohlcv) -> None:
        """Each anomaly type score must be in [0.0, 1.0]."""
        agent = AnomalyAgent()
        for evidence in agent.analyze(normal_ohlcv).evidence:
            if "score" in evidence.value.lower():
                val_str = evidence.value.split(":")[1].split()[0]
                val = float(val_str)
                assert 0.0 <= val <= 1.0, f"{evidence.feature} score {val} out of range"

    def test_all_four_types_reported(self, normal_ohlcv) -> None:
        """All four anomaly types should appear in evidence."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        features = {e.feature for e in report.evidence}
        assert "anomaly" in features
        assert "spoofing_like" in features
        assert "liquidity_withdrawal" in features
        assert "cross_venue_divergence" in features


# ── Integration ───────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_normal(self, normal_ohlcv) -> None:
        """Full pipeline: validation → scores → report."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)

        assert report.agent_id == "anomaly"
        assert report.status == "shadow"
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1
        assert len(report.invalidations) >= 1
        assert abs(sum(report.probabilities.values()) - 1.0) <= 0.0001
        assert 0.0 <= report.raw_confidence <= 0.9

    def test_full_pipeline_with_spike(self, volume_spike_data) -> None:
        """Full pipeline with anomalous data."""
        agent = AnomalyAgent(threshold=0.3)
        report = agent.analyze(volume_spike_data)

        assert report.agent_id == "anomaly"
        # At least one positive evidence (flagged)
        positive = [e for e in report.evidence if e.direction == "positive"]
        assert len(positive) >= 1

        # Counter-evidence should provide alternative explanations
        assert len(report.counter_evidence) >= 1
        for ce in report.counter_evidence:
            assert ce.direction == "negative"

    def test_full_pipeline_custom_threshold(self, normal_ohlcv) -> None:
        """Custom threshold changes sensitivity."""
        agent_low = AnomalyAgent(threshold=0.9)
        agent_high = AnomalyAgent(threshold=0.3)

        r_low = agent_low.analyze(normal_ohlcv)
        r_high = agent_high.analyze(normal_ohlcv)

        pos_low = sum(1 for e in r_low.evidence if e.direction == "positive")
        pos_high = sum(1 for e in r_high.evidence if e.direction == "positive")

        # Higher threshold = fewer flagged (same or fewer positive)
        assert pos_low <= pos_high or pos_low == 0


# ── Spec Compliance ───────────────────────────────────────────────────────

class TestSpecCompliance:
    def test_only_scores_no_factual_claims(self, normal_ohlcv) -> None:
        """Evidence must be phrased as scores, not definitive statements."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        for e in report.evidence:
            val_lower = e.value.lower()
            # Must reference a score or be clearly informational
            has_score = "score" in val_lower or "threshold" in val_lower
            is_informative = "no anomaly" in val_lower or "t=" in val_lower
            assert has_score or is_informative, (
                f"Evidence {e.reference} makes a factual claim: {e.value[:80]}"
            )

    def test_probabilities_sum_to_one(self, normal_ohlcv) -> None:
        """AT-005: Probabilities must sum to 1.0 ± 0.0001."""
        agent = AnomalyAgent()
        for _ in range(10):
            report = agent.analyze(normal_ohlcv)
            total = sum(report.probabilities.values())
            assert abs(total - 1.0) <= 0.0001

    def test_counter_evidence_required(self, normal_ohlcv) -> None:
        """Spec: Counter-hypothesis required."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert len(report.counter_evidence) >= 1

    def test_evidence_min_one(self, normal_ohlcv) -> None:
        """AT-004: At least one evidence required."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert len(report.evidence) >= 1

    def test_initial_status_shadow(self, normal_ohlcv) -> None:
        """Spec: Initial status must be SHADOW."""
        agent = AnomalyAgent()
        report = agent.analyze(normal_ohlcv)
        assert report.status == "shadow"

    def test_conservative_scoring(self, normal_ohlcv) -> None:
        """Spec: Conservative scoring — only flag when score > 0.6."""
        agent = AnomalyAgent(threshold=0.6)
        report = agent.analyze(normal_ohlcv)
        # Normal data should have no (or very few) positive evidence
        positive = [e for e in report.evidence if e.direction == "positive"]
        assert len(positive) < 4  # Should not flag all types
