"""Tests for the consensus module.

Validates vote direction enum values, consensus decision enums,
weighted consensus engine behavior, and historical weight tracking.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from packages.consensus import (
    ConsensusDecision,
    HistoricalWeightTracker,
    VoteDirection,
    WeightConfig,
    WeightedConsensusEngine,
)
from packages.schemas.agent_report import AgentReport, AgentStatus, EvidenceReference


def _make_report(
    agent_id: str = "agent-1",
    status: AgentStatus = AgentStatus.ACTIVE,
    probabilities: dict[str, float] | None = None,
) -> AgentReport:
    """Erzeugt einen vollständigen AgentReport via Pydantic."""
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
        evidence=[
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
    )


class TestVoteDirection:
    """Testet VoteDirection-Enum."""

    def test_all_directions(self) -> None:
        assert VoteDirection.LONG == "long"
        assert VoteDirection.SHORT == "short"
        assert VoteDirection.RANGE == "range"
        assert VoteDirection.ABSTAIN == "abstain"


class TestConsensusDecision:
    """Testet ConsensusDecision-Enum."""

    def test_all_decisions(self) -> None:
        assert ConsensusDecision.LONG_BIAS == "LONG_BIAS"
        assert ConsensusDecision.SHORT_BIAS == "SHORT_BIAS"
        assert ConsensusDecision.RANGE == "RANGE"
        assert ConsensusDecision.NO_TRADE == "NO_TRADE"


class TestWeightConfig:
    """Testet WeightConfig-Standardwerte."""

    def test_defaults(self) -> None:
        config = WeightConfig()
        assert config.base_weight == 1.0
        assert config.status_multiplier == {
            "active": 1.0,
            "shadow": 0.5,
            "degraded": 0.3,
            "quarantined": 0.0,
            "disabled": 0.0,
        }
        assert config.min_consensus_threshold == 0.5
        assert config.max_agent_divergence == 0.7


class TestWeightedConsensusEngine:
    """Testet den gewichteten Konsens-Engine."""

    def test_single_long_agent(self) -> None:
        """Ein einziger Agent mit LONG-Vote ergibt LONG_BIAS."""
        engine = WeightedConsensusEngine()
        report = _make_report(agent_id="a1", probabilities={"up": 0.7, "down": 0.1, "range": 0.2})
        result = engine.compute_consensus([report])

        assert result.decision == ConsensusDecision.LONG_BIAS
        assert "a1" in result.agent_agreements
        assert result.confidence > 0.0

    def test_single_short_agent(self) -> None:
        """Ein einziger Agent mit SHORT-Vote ergibt SHORT_BIAS."""
        engine = WeightedConsensusEngine()
        report = _make_report(agent_id="a1", probabilities={"up": 0.1, "down": 0.7, "range": 0.2})
        result = engine.compute_consensus([report])

        assert result.decision == ConsensusDecision.SHORT_BIAS
        assert "a1" in result.agent_agreements

    def test_mixed_with_consensus(self) -> None:
        """3 LONG + 1 SHORT -> LONG_BIAS."""
        engine = WeightedConsensusEngine()
        reports = [
            _make_report(
                agent_id=f"a{i}",
                probabilities={"up": 0.7, "down": 0.1, "range": 0.2},
            )
            for i in range(3)
        ]
        reports.append(
            _make_report(
                agent_id="a3",
                probabilities={"up": 0.1, "down": 0.7, "range": 0.2},
            )
        )

        result = engine.compute_consensus(reports)

        assert result.decision == ConsensusDecision.LONG_BIAS
        assert len(result.agent_agreements) == 3
        assert len(result.agent_disagreements) == 1

    def test_no_consensus_reached(self) -> None:
        """Ausgewogene Verteilung -> NO_TRADE."""
        engine = WeightedConsensusEngine(config=WeightConfig(min_consensus_threshold=0.8))
        reports = [
            _make_report(agent_id="long1", probabilities={"up": 0.7, "down": 0.15, "range": 0.15}),
            _make_report(agent_id="short1", probabilities={"up": 0.15, "down": 0.7, "range": 0.15}),
        ]
        result = engine.compute_consensus(reports)

        assert result.decision == ConsensusDecision.NO_TRADE

    def test_status_multiplier(self) -> None:
        """Shadow-Agent hat halb so viel Gewicht wie aktiver Agent."""
        engine = WeightedConsensusEngine()
        active_report = _make_report(agent_id="active", status=AgentStatus.ACTIVE, probabilities={"up": 0.7, "down": 0.1, "range": 0.2})
        shadow_report = _make_report(agent_id="shadow", status=AgentStatus.SHADOW, probabilities={"up": 0.7, "down": 0.1, "range": 0.2})

        result = engine.compute_consensus([active_report, shadow_report])

        assert result.decision == ConsensusDecision.LONG_BIAS
        assert result.agent_weights["active"] == 1.0
        assert result.agent_weights["shadow"] == 0.5

    def test_empty_reports_raises(self) -> None:
        """Leere Reports-Liste lost ValueError aus."""
        engine = WeightedConsensusEngine()
        with pytest.raises(ValueError, match="reports must not be empty"):
            engine.compute_consensus([])

    def test_empty_probabilities_raises(self) -> None:
        """validate_input lost ValueError bei leerer Wahrscheinlichkeit."""
        engine = WeightedConsensusEngine()
        report = _make_report(agent_id="a1", probabilities={"up": 0.7, "down": 0.1, "range": 0.2})
        object.__setattr__(report, "probabilities", {})
        with pytest.raises(ValueError, match="empty probabilities"):
            engine.validate_input([report])

    def test_consensus_result_has_all_fields(self) -> None:
        """ConsensusResult enthaelt alle erwarteten Felder."""
        engine = WeightedConsensusEngine()
        report = _make_report(agent_id="a1", probabilities={"up": 0.7, "down": 0.1, "range": 0.2})
        result = engine.compute_consensus([report])

        assert isinstance(result.decision, ConsensusDecision)
        assert isinstance(result.vote_distribution, dict)
        assert isinstance(result.agent_weights, dict)
        assert isinstance(result.agent_agreements, list)
        assert isinstance(result.agent_disagreements, list)
        assert isinstance(result.confidence, float)
        assert isinstance(result.reason, str)
        assert 0.0 <= result.confidence <= 1.0


class TestHistoricalWeightTracker:
    """Testet den historischen Gewichtungs-Tracker."""

    def test_record_and_query_stats(self) -> None:
        """Vorhersagen aufnehmen und Statistiken abfragen."""
        tracker = HistoricalWeightTracker(lookback=10)
        tracker.record_prediction("a1", VoteDirection.LONG, VoteDirection.LONG)
        tracker.record_prediction("a1", VoteDirection.SHORT, VoteDirection.LONG)
        tracker.record_prediction("a1", VoteDirection.LONG, VoteDirection.LONG)

        stats = tracker.get_agent_stats("a1")
        assert stats["total_predictions"] == 3
        assert stats["correct"] == 2
        assert abs(stats["accuracy"] - 2 / 3) < 1e-6

    def test_accuracy_calculation(self) -> None:
        """Genauigkeitsberechnung mit exponentieller Decay.

        3 falsche (aelteste) + 7 richtige (neueste) mit decay=0.1:
        neueste (index 9): weight=1.0, correct -> 1.0
        index 8: weight=0.1, correct -> 0.1
        ... bis index 3: weight=0.1^6, correct -> 0.000001
        index 2: weight=0.1^7, wrong -> 0
        index 1: weight=0.1^8, wrong -> 0
        index 0: weight=0.1^9, wrong -> 0

        weighted_correct ~ 1.111111, total_weight ~ 1.111111111
        accuracy ~ 0.999999
        """
        tracker = HistoricalWeightTracker(lookback=10, accuracy_decay=0.1)
        for _ in range(3):
            tracker.record_prediction("a1", VoteDirection.SHORT, VoteDirection.LONG)
        for _ in range(7):
            tracker.record_prediction("a1", VoteDirection.LONG, VoteDirection.LONG)

        accuracy = tracker.get_accuracy("a1")
        assert accuracy > 0.99

    def test_no_history_returns_neutral(self) -> None:
        """Keine Historie -> accuracy 1.0, Gewichtsanpassung 1.0."""
        tracker = HistoricalWeightTracker()

        accuracy = tracker.get_accuracy("unknown-agent")
        assert accuracy == 1.0

        adjustment = tracker.get_weight_adjustment("unknown-agent")
        assert adjustment == 1.0

        stats = tracker.get_agent_stats("unknown-agent")
        assert stats["total_predictions"] == 0
        assert stats["accuracy"] == 0.0

    def test_decay_applied(self) -> None:
        """Aeltere Vorhersagen erhalten weniger Gewicht."""
        tracker = HistoricalWeightTracker(lookback=10, accuracy_decay=0.1)

        for _ in range(9):
            tracker.record_prediction("a1", VoteDirection.LONG, VoteDirection.SHORT)
        tracker.record_prediction("a1", VoteDirection.LONG, VoteDirection.LONG)

        accuracy = tracker.get_accuracy("a1")
        assert accuracy > 0.89
        assert accuracy < 1.0

    def test_weight_adjustment_range(self) -> None:
        """Gewichtsanpassung liegt immer im Bereich [0.0, 1.0]."""
        tracker = HistoricalWeightTracker(lookback=100)
        tracker.record_prediction("a1", VoteDirection.LONG, VoteDirection.SHORT)

        adjustment = tracker.get_weight_adjustment("a1")
        assert 0.0 <= adjustment <= 1.0
        assert abs(adjustment - 0.5) < 1e-6

    def test_per_agent_independence(self) -> None:
        """Agenten haben unabhaengige Historien."""
        tracker = HistoricalWeightTracker()
        tracker.record_prediction("a1", VoteDirection.LONG, VoteDirection.LONG)
        tracker.record_prediction("a2", VoteDirection.SHORT, VoteDirection.LONG)

        stats_a1 = tracker.get_agent_stats("a1")
        stats_a2 = tracker.get_agent_stats("a2")

        assert stats_a1["correct"] == 1
        assert stats_a2["correct"] == 0
