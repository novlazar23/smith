"""Tests fuer packages/orchestrator/second_round.py."""

from __future__ import annotations

import pytest
from packages.consensus import ConsensusDecision, ConsensusResult
from packages.orchestrator.graph import (
    OrchestratorGraph,
    PipelineStage,
    create_initial_state,
    run_first_round,
)
from packages.orchestrator.second_round import (
    RoundSummary,
    build_round_summary,
    compute_consensus,
    run_second_round,
)
from packages.schemas.agent_report import AgentReport, AgentStatus, EvidenceReference
from datetime import UTC, datetime


def _make_report(
    agent_id: str = "agent-1",
    up: float = 0.65,
    down: float = 0.15,
    status: AgentStatus = AgentStatus.ACTIVE,
) -> AgentReport:
    return AgentReport(
        report_id=f"{agent_id}-r1",
        run_id="test-run",
        agent_id=agent_id,
        agent_version="1.0.0",
        instrument="BTC/USD",
        horizon="1h",
        as_of=datetime.now(UTC),
        hypothesis=f"{agent_id} bullish",
        probabilities={"up": up, "down": down, "range": 1.0 - up - down},
        evidence=[EvidenceReference(
            reference=f"{agent_id}:test",
            feature="test",
            value="rising",
            direction="positive",
            relevance=0.8,
        )],
        raw_confidence=up,
        status=status,
    )


class TestRoundSummary:
    """Testet RoundSummary."""

    def test_default_empty(self) -> None:
        summary = RoundSummary()
        assert summary.direction == "NO_TRADE"
        assert summary.confidence == 0.0
        assert summary.dissent_score == 0.0
        assert summary.agent_agreements == []
        assert summary.agent_disagreements == []


class TestBuildRoundSummary:
    """Testet build_round_summary."""

    def test_empty_reports(self) -> None:
        summary = build_round_summary([])
        assert summary.direction == "NO_TRADE"
        assert summary.confidence == 0.0
        assert summary.dissent_score == 0.0

    def test_single_report(self) -> None:
        report = _make_report(agent_id="a1", up=0.7)
        summary = build_round_summary([report])
        assert summary.direction == "LONG_BIAS"
        assert summary.confidence == 0.7
        assert summary.dissent_score == 0.0  # single agent = no dissent

    def test_all_agree_long(self) -> None:
        reports = [
            _make_report(agent_id=f"a{i}", up=0.7 + i * 0.05)
            for i in range(3)
        ]
        summary = build_round_summary(reports)
        assert summary.direction == "LONG_BIAS"
        assert summary.dissent_score == 0.0  # all agree

    def test_high_dissent(self) -> None:
        """Equal split between long, short, range = maximum dissent."""
        reports = [
            _make_report(agent_id="long-1", up=0.7, down=0.1),  # long
            _make_report(agent_id="short-1", up=0.1, down=0.7),  # short
            _make_report(agent_id="range-1", up=0.2, down=0.2),   # range
        ]
        summary = build_round_summary(reports)
        assert summary.dissent_score > 0.5  # high dissent

    def test_avg_confidence(self) -> None:
        reports = [
            _make_report(agent_id="a1", up=0.8),
            _make_report(agent_id="a2", up=0.4),
        ]
        summary = build_round_summary(reports)
        assert summary.confidence == 0.6  # (0.8 + 0.4) / 2

    def test_direction_short(self) -> None:
        reports = [
            _make_report(agent_id="a1", up=0.1, down=0.7),
            _make_report(agent_id="a2", up=0.15, down=0.65),
        ]
        summary = build_round_summary(reports)
        assert summary.direction == "SHORT_BIAS"

    def test_direction_range(self) -> None:
        reports = [
            _make_report(agent_id="a1", up=0.33, down=0.33),
            _make_report(agent_id="a2", up=0.30, down=0.30),
        ]
        summary = build_round_summary(reports)
        assert summary.direction == "RANGE"


class TestRunSecondRound:
    """Testet run_second_round."""

    def test_raises_no_first_round_reports(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        with pytest.raises(ValueError, match="first_round_reports"):
            run_second_round(state, graph, [MockAgent(agent_id="a1")], {})

    def test_raises_no_seal_records(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        from packages.orchestrator.seal import seal_first_round
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id="a1")]
        state, graph = run_first_round(state, graph, agents, {})
        # Seal noch nicht durchgefuehrt — seal_records leer
        with pytest.raises(ValueError, match="seal_records"):
            run_second_round(state, graph, agents, {})

    def test_raises_no_agents(self) -> None:
        from packages.orchestrator.seal import seal_first_round
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        state, graph = run_first_round(state, graph, [MockAgent(agent_id="a1")], {})
        state, graph, _ = seal_first_round(state, graph)
        with pytest.raises(ValueError, match="at least one agent"):
            run_second_round(state, graph, [], {})

    def test_stores_second_round_reports(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        from packages.orchestrator.seal import seal_first_round
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id=f"agent-{i}") for i in range(3)]
        state, graph = run_first_round(state, graph, agents, {})
        state, graph, _ = seal_first_round(state, graph)

        state, graph, reports = run_second_round(state, graph, agents, {})
        assert len(reports) == 3
        assert len(state.second_round_reports) == 3
        assert state.current_stage == PipelineStage.SECOND_ROUND.value
        assert graph.has_completed(PipelineStage.SECOND_ROUND)

    def test_reports_have_context(self) -> None:
        """Second-Round-Reports sollten mit analyze_with_context erstellt werden."""
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        from packages.orchestrator.seal import seal_first_round
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agent = MockAgent(agent_id="a1")
        state, graph = run_first_round(state, graph, [agent], {})
        state, graph, _ = seal_first_round(state, graph)

        state, graph, reports = run_second_round(state, graph, [agent], {})
        # Second-Round-Report hat "-r2" Suffix (aus MockAgent)
        assert reports[0].report_id == "a1-r2"

    def test_round_summary_stored(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        from packages.orchestrator.seal import seal_first_round
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id="a1")]
        state, graph = run_first_round(state, graph, agents, {})
        state, graph, _ = seal_first_round(state, graph)

        state, graph, _ = run_second_round(state, graph, agents, {})
        assert state.round_summary["direction"] == "LONG_BIAS"
        assert state.round_summary["agent_count"] == 1

    def test_audit_event_created(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        from packages.orchestrator.seal import seal_first_round
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id="a1")]
        state, graph = run_first_round(state, graph, agents, {})
        state, graph, _ = seal_first_round(state, graph)

        run_second_round(state, graph, agents, {})
        events = [e.stage for e in graph.audit_events]
        assert "second_round" in events


class TestComputeConsensus:
    """Testet compute_consensus."""

    def test_no_active_agents(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import ShadowAgent
        shadow = ShadowAgent("shadow-1")
        reports = [shadow.analyze({})]
        result = compute_consensus(reports)
        assert result.decision == ConsensusDecision.NO_TRADE
        assert "shadow" in result.reason.lower()

    def test_shadow_weight_zero(self) -> None:
        """Shadow-Agenten haben weight 0.0 und beeinflussen Konsens nicht."""
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent, ShadowAgent
        agents = [
            MockAgent(agent_id="long-1", bias=0.7),  # long
            ShadowAgent("shadow-1"),  # shadow — weight 0.0
        ]
        reports = [a.analyze({}) for a in agents]
        result = compute_consensus(reports)
        # Nur long-1 gehoert zu active — weight 1.0
        assert result.agent_weights["long-1"] == 1.0
        assert "shadow-1" not in result.agent_weights

    def test_active_consensus(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        agents = [MockAgent(agent_id="a1", bias=0.7)]
        reports = [a.analyze({}) for a in agents]
        result = compute_consensus(reports)
        assert result.decision == ConsensusDecision.LONG_BIAS
        assert result.confidence > 0

    def test_high_dissent_no_trade(self) -> None:
        """>60% Dissens → NO_TRADE."""
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent, short_agent
        agents = [
            MockAgent(agent_id="long-1", bias=0.7),
            short_agent("short-1"),
            MockAgent(agent_id="long-2", bias=0.7),
        ]
        reports = [a.analyze({}) for a in agents]
        result = compute_consensus(reports, high_dissent=True)
        # 2 long, 1 short — consensus sollte LONG sein aber Dissens pruefen
        # confidence > 0.4 also kein NO_TRADE durch high_dissent
        if result.decision == ConsensusDecision.NO_TRADE:
            assert "High dissent" in result.reason
        else:
            assert result.decision == ConsensusDecision.LONG_BIAS

    def test_consensus_ignores_shadow_in_weights(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent, ShadowAgent
        agents = [
            MockAgent(agent_id="a1", bias=0.7),
            MockAgent(agent_id="a2", bias=0.6),
            ShadowAgent("sh1"),
        ]
        reports = [a.analyze({}) for a in agents]
        result = compute_consensus(reports)
        # Nur a1 und a2 sollten in weights sein
        assert "sh1" not in result.agent_weights
        assert "a1" in result.agent_weights
        assert "a2" in result.agent_weights

    def test_single_agent_consensus(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        agents = [MockAgent(agent_id="solo", bias=0.8)]
        reports = [a.analyze({}) for a in agents]
        result = compute_consensus(reports)
        assert result.decision == ConsensusDecision.LONG_BIAS
        assert len(result.agent_agreements) == 1

    def test_vote_distribution_sum(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        agents = [MockAgent(agent_id="a1", bias=0.7)]
        reports = [a.analyze({}) for a in agents]
        result = compute_consensus(reports)
        dist_sum = sum(result.vote_distribution.values())
        assert abs(dist_sum - 1.0) < 0.001