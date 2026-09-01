"""Tests for the dependency analysis and calibrated aggregation modules.

Covers DependencyAnalyzer correlation detection, weight reduction,
and CalibratedConsensusAggregator calibration and dissent detection.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from packages.consensus import (
    ConsensusDecision,
    DependencyAnalysisResult,
    DependencyAnalyzer,
    WeightConfig,
)
from packages.consensus.dependency_analysis import AgentDependency
from packages.schemas.agent_report import AgentReport, AgentStatus, EvidenceReference


def _make_report(
    agent_id: str = "agent-1",
    status: AgentStatus = AgentStatus.ACTIVE,
    probabilities: dict[str, float] | None = None,
    evidence: list[EvidenceReference] | None = None,
    feature_snapshot_id: str | None = None,
) -> AgentReport:
    """Create a minimal AgentReport for testing."""
    return AgentReport(
        report_id=f"rpt-{agent_id}",
        run_id="run-001",
        agent_id=agent_id,
        agent_version="0.1.0",
        instrument="EUR/USD",
        horizon="1h",
        as_of=datetime.now(),
        hypothesis="test-hypothesis",
        probabilities=probabilities or {"up": 0.7, "down": 0.1, "range": 0.2},
        expected_return=None,
        evidence=evidence
        or [
            EvidenceReference(
                reference=f"{agent_id}:rsi",
                feature="rsi",
                value="30",
                direction="positive",
                relevance=0.8,
            )
        ],
        raw_confidence=None,
        calibrated_confidence=None,
        status=status,
        feature_snapshot_id=feature_snapshot_id,
    )


# ── DependencyAnalyzer ──────────────────────────────────────────────────


class TestDependencyAnalyzer:
    """Tests for DependencyAnalyzer."""

    def test_empty_reports(self) -> None:
        """Empty input returns empty analysis."""
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze([])

        assert isinstance(result, DependencyAnalysisResult)
        assert result.dependencies == []
        assert result.dependency_matrix == {}
        assert result.adjusted_weights == {}

    def test_single_agent_no_dependency(self) -> None:
        """Single agent produces no dependencies."""
        analyzer = DependencyAnalyzer()
        report = _make_report(agent_id="a1")
        result = analyzer.analyze([report])

        assert result.dependencies == []
        assert "a1" in result.adjusted_weights
        assert result.adjusted_weights["a1"] == 1.0

    def test_non_correlated_agents(self) -> None:
        """Agents with different features get no correlation."""
        evidence_a = [
            EvidenceReference(
                reference="a:feature_a",
                feature="feature_a",
                value="1",
                direction="positive",
                relevance=0.8,
            ),
        ]
        evidence_b = [
            EvidenceReference(
                reference="b:feature_b",
                feature="feature_b",
                value="2",
                direction="negative",
                relevance=0.6,
            ),
        ]
        analyzer = DependencyAnalyzer()
        reports = [
            _make_report(agent_id="a1", evidence=evidence_a),
            _make_report(agent_id="b1", evidence=evidence_b),
        ]
        result = analyzer.analyze(reports)

        assert len(result.dependencies) == 0
        assert result.adjusted_weights["a1"] == 1.0
        assert result.adjusted_weights["b1"] == 1.0

    def test_highly_correlated_agents_reduce_weight(self) -> None:
        """Agents sharing >70% features get weight reduction."""
        shared_ev = [
            EvidenceReference(
                reference="shared:rsi",
                feature="rsi",
                value="30",
                direction="positive",
                relevance=0.8,
            ),
            EvidenceReference(
                reference="shared:macd",
                feature="macd",
                value="0.5",
                direction="positive",
                relevance=0.7,
            ),
        ]
        reports = [
            _make_report(agent_id="a1", evidence=shared_ev),
            _make_report(agent_id="a2", evidence=shared_ev),
        ]
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze(reports)

        assert len(result.dependencies) == 1
        dep = result.dependencies[0]
        assert dep.agent_a == "a1"
        assert dep.agent_b == "a2"
        assert dep.correlation == 1.0
        assert result.adjusted_weights["a1"] == 1.0
        assert result.adjusted_weights["a2"] == pytest.approx(0.0)

    def test_partial_correlation(self) -> None:
        """Partially overlapping features yield partial correlation."""
        evidence_a = [
            EvidenceReference(
                reference="a:f1",
                feature="f1",
                value="1",
                direction="positive",
                relevance=0.8,
            ),
            EvidenceReference(
                reference="a:f2",
                feature="f2",
                value="2",
                direction="positive",
                relevance=0.7,
            ),
            EvidenceReference(
                reference="a:f3",
                feature="f3",
                value="3",
                direction="positive",
                relevance=0.6,
            ),
        ]
        evidence_b = [
            EvidenceReference(
                reference="b:f1",
                feature="f1",
                value="1",
                direction="positive",
                relevance=0.8,
            ),
            EvidenceReference(
                reference="b:f2",
                feature="f2",
                value="2",
                direction="positive",
                relevance=0.7,
            ),
        ]
        reports = [
            _make_report(agent_id="a1", evidence=evidence_a),
            _make_report(agent_id="b1", evidence=evidence_b),
        ]
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze(reports)

        # 2 shared out of 4 unique = 0.5 correlation (below threshold)
        assert len(result.dependencies) == 0
        assert result.adjusted_weights["a1"] == 1.0
        assert result.adjusted_weights["b1"] == 1.0

    def test_feature_snapshot_includes_snapshot_feature(self) -> None:
        """feature_snapshot_id is included as a feature for correlation."""
        reports = [
            _make_report(agent_id="a1", feature_snapshot_id="snap-1"),
            _make_report(agent_id="a2", feature_snapshot_id="snap-1"),
        ]
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze(reports)

        assert len(result.dependency_matrix) == 1
        key = ("a1", "a2")
        assert key in result.dependency_matrix

    def test_dependency_matrix_populated(self) -> None:
        """All pairs have correlation scores in the matrix."""
        reports = [
            _make_report(agent_id="a1"),
            _make_report(agent_id="a2"),
            _make_report(agent_id="a3"),
        ]
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze(reports)

        assert len(result.dependency_matrix) == 3
        assert ("a1", "a2") in result.dependency_matrix
        assert ("a1", "a3") in result.dependency_matrix
        assert ("a2", "a3") in result.dependency_matrix

    def test_correlation_threshold_custom(self) -> None:
        """Custom threshold changes which pairs are flagged."""
        shared_ev = [
            EvidenceReference(
                reference="shared:feature",
                feature="feature",
                value="val",
                direction="positive",
                relevance=0.9,
            ),
        ]
        reports = [
            _make_report(agent_id="a1", evidence=shared_ev),
            _make_report(agent_id="a2", evidence=shared_ev),
        ]
        analyzer = DependencyAnalyzer(correlation_threshold=1.0)
        result = analyzer.analyze(reports)
        assert len(result.dependencies) == 1

    def test_reduce_weights_with_shadow(self) -> None:
        """Shadow agents always get weight 0.0."""
        reports = [
            _make_report(agent_id="active1", status=AgentStatus.ACTIVE),
            _make_report(agent_id="shadow1", status=AgentStatus.SHADOW),
        ]
        analyzer = DependencyAnalyzer()
        weights = analyzer.reduce_weights(reports)

        assert weights["active1"] == 1.0
        assert weights["shadow1"] == 0.0

    def test_reduce_weights_returns_dict(self) -> None:
        """reduce_weights returns a dict mapping agent_id to weight."""
        reports = [_make_report(agent_id="a1"), _make_report(agent_id="a2")]
        analyzer = DependencyAnalyzer()
        weights = analyzer.reduce_weights(reports)

        assert isinstance(weights, dict)
        assert "a1" in weights
        assert "a2" in weights
        assert all(isinstance(w, float) for w in weights.values())

    def test_highly_correlated_weight_reduction_factor(self) -> None:
        """Reduction factor is (1 - correlation) applied to the reduced agent."""
        shared_ev = [
            EvidenceReference(
                reference="shared:feature",
                feature="feature",
                value="val",
                direction="positive",
                relevance=0.9,
            ),
        ]
        reports = [
            _make_report(agent_id="a1", evidence=shared_ev),
            _make_report(agent_id="a2", evidence=shared_ev),
        ]
        analyzer = DependencyAnalyzer(correlation_threshold=0.0)
        result = analyzer.analyze(reports)

        dep = result.dependencies[0]
        assert dep.reduced_agent == "a2"
        assert result.adjusted_weights["a2"] == pytest.approx(0.0)


class TestAgentDependency:
    """Tests for the AgentDependency frozen dataclass."""

    def test_frozen(self) -> None:
        """AgentDependency is immutable."""
        dep = AgentDependency(
            agent_a="a1",
            agent_b="a2",
            correlation=0.5,
            shared_features=3,
            total_features=6,
            reduced_agent="a2",
        )
        with pytest.raises(Exception):
            dep.agent_a = "changed"


# ── CalibratedConsensusAggregator ──────────────────────────────────────


class TestCalibratedConsensusAggregator:
    """Tests for CalibratedConsensusAggregator."""

    def test_basic_long_consensus(self) -> None:
        """All-long reports produce LONG_BIAS."""
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        aggregator = CalibratedConsensusAggregator()
        reports = [
            _make_report(agent_id=f"a{i}", probabilities={"up": 0.7, "down": 0.1, "range": 0.2})
            for i in range(3)
        ]
        result = aggregator.aggregate(reports)

        assert result.decision == ConsensusDecision.LONG_BIAS
        assert not result.dissent_detected

    def test_basic_short_consensus(self) -> None:
        """All-short reports produce SHORT_BIAS."""
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        aggregator = CalibratedConsensusAggregator()
        reports = [
            _make_report(
                agent_id=f"a{i}", probabilities={"up": 0.1, "down": 0.7, "range": 0.2}
            )
            for i in range(3)
        ]
        result = aggregator.aggregate(reports)

        assert result.decision == ConsensusDecision.SHORT_BIAS

    def test_shadow_agent_zero_weight(self) -> None:
        """Shadow agents get weight 0.0 and do not influence consensus."""
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        aggregator = CalibratedConsensusAggregator()
        reports = [
            _make_report(agent_id="active1", status=AgentStatus.ACTIVE),
            _make_report(agent_id="shadow1", status=AgentStatus.SHADOW),
        ]
        result = aggregator.aggregate(reports)

        assert result.agent_weights["shadow1"] == 0.0
        assert result.agent_weights["active1"] == 1.0
        assert result.decision == ConsensusDecision.LONG_BIAS

    def test_empty_reports_raises(self) -> None:
        """Empty reports list raises ValueError."""
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        aggregator = CalibratedConsensusAggregator()
        with pytest.raises(ValueError, match="reports must not be empty"):
            aggregator.aggregate([])

    def test_empty_probabilities_raises(self) -> None:
        """Report with no probabilities raises ValueError."""
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        aggregator = CalibratedConsensusAggregator()
        report = _make_report(agent_id="a1")
        object.__setattr__(report, "probabilities", {})
        with pytest.raises(ValueError, match="empty probabilities"):
            aggregator.aggregate([report])

    def test_dissent_detection(self) -> None:
        """Close long/short ratios trigger dissent -> NO_TRADE.

        1 LONG agent (up=0.7) + 1 SHORT agent (down=0.7):
        long_ratio = 1/2 = 0.5, short_ratio = 1/2 = 0.5
        dissent_score = |0.5 - 0.5| = 0.0 < 0.1 -> high dissent
        """
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        aggregator = CalibratedConsensusAggregator(
            config=WeightConfig(min_consensus_threshold=0.3)
        )
        reports = [
            _make_report(agent_id="long1", probabilities={"up": 0.7, "down": 0.2, "range": 0.1}),
            _make_report(agent_id="short1", probabilities={"up": 0.2, "down": 0.7, "range": 0.1}),
        ]
        result = aggregator.aggregate(reports)

        assert result.decision == ConsensusDecision.NO_TRADE
        assert result.dissent_detected is True

    def test_no_dissent_clear_leader(self) -> None:
        """Clear long majority does not trigger dissent."""
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        aggregator = CalibratedConsensusAggregator(
            config=WeightConfig(min_consensus_threshold=0.3)
        )
        reports = [
            _make_report(
                agent_id=f"long{i}",
                probabilities={"up": 0.7, "down": 0.15, "range": 0.15},
            )
            for i in range(3)
        ]
        reports.append(
            _make_report(
                agent_id="short1",
                probabilities={"up": 0.15, "down": 0.7, "range": 0.15},
            )
        )
        result = aggregator.aggregate(reports)

        assert result.dissent_detected is False

    def test_calibrated_probabilities_present(self) -> None:
        """Result contains calibrated probabilities dict."""
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        aggregator = CalibratedConsensusAggregator()
        reports = [_make_report(agent_id="a1")]
        result = aggregator.aggregate(reports)

        assert "up_calibrated" in result.calibrated_probabilities
        assert "down_calibrated" in result.calibrated_probabilities
        assert "range_calibrated" in result.calibrated_probabilities

    def test_dependency_adjusted_weights_used(self) -> None:
        """Adjusted weights from DependencyAnalyzer are applied."""
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        correlation_ev = [
            EvidenceReference(
                reference="shared:feature",
                feature="feature",
                value="val",
                direction="positive",
                relevance=0.9,
            ),
        ]
        reports = [
            _make_report(agent_id="a1", evidence=correlation_ev),
            _make_report(agent_id="a2", evidence=correlation_ev),
        ]
        analyzer = DependencyAnalyzer(correlation_threshold=0.0)
        adjusted = analyzer.reduce_weights(reports)

        aggregator = CalibratedConsensusAggregator()
        result = aggregator.aggregate(reports, adjusted_weights=adjusted)

        assert result.agent_weights["a2"] == 0.0

    def test_vote_distribution_values(self) -> None:
        """Vote distribution sums to 1.0."""
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        aggregator = CalibratedConsensusAggregator()
        reports = [_make_report(agent_id="a1")]
        result = aggregator.aggregate(reports)

        total = sum(result.vote_distribution.values())
        assert abs(total - 1.0) < 1e-6

    def test_range_consensus(self) -> None:
        """All-range reports produce RANGE decision (low threshold)."""
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        aggregator = CalibratedConsensusAggregator(
            config=WeightConfig(min_consensus_threshold=0.3)
        )
        reports = [
            _make_report(
                agent_id=f"a{i}",
                probabilities={"up": 0.2, "down": 0.2, "range": 0.6},
            )
            for i in range(3)
        ]
        result = aggregator.aggregate(reports)

        assert result.decision == ConsensusDecision.RANGE

    def test_consensus_result_has_all_fields(self) -> None:
        """CalibratedConsensusResult has all expected fields."""
        from packages.consensus.aggregation import CalibratedConsensusAggregator

        aggregator = CalibratedConsensusAggregator()
        reports = [_make_report(agent_id="a1")]
        result = aggregator.aggregate(reports)

        assert isinstance(result.decision, ConsensusDecision)
        assert isinstance(result.vote_distribution, dict)
        assert isinstance(result.calibrated_probabilities, dict)
        assert isinstance(result.agent_weights, dict)
        assert isinstance(result.dissent_detected, bool)
        assert isinstance(result.dissent_score, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.reason, str)
