"""Tests fuer packages/orchestrator/graph.py."""

from __future__ import annotations

import pytest
from packages.orchestrator.graph import (
    AuditEvent,
    OrchestratorGraph,
    PipelineStage,
    create_initial_state,
    run_first_round,
)


class TestPipelineStage:
    """Testet PipelineStage Enum."""

    def test_all_stages_present(self) -> None:
        stages = [s for s in PipelineStage]
        assert len(stages) == 7

    def test_stage_values(self) -> None:
        expected = {
            "request", "first_round", "seal", "second_round",
            "consensus", "decision", "publish",
        }
        actual = {s.value for s in PipelineStage}
        assert actual == expected

    def test_first_round_folows_request(self) -> None:
        from packages.orchestrator.graph import VALID_TRANSITIONS
        allowed = VALID_TRANSITIONS[PipelineStage.REQUEST]
        assert PipelineStage.FIRST_ROUND in allowed

    def test_seal_folows_first_round(self) -> None:
        from packages.orchestrator.graph import VALID_TRANSITIONS
        allowed = VALID_TRANSITIONS[PipelineStage.FIRST_ROUND]
        assert PipelineStage.SEAL in allowed

    def test_second_round_folows_seal(self) -> None:
        from packages.orchestrator.graph import VALID_TRANSITIONS
        allowed = VALID_TRANSITIONS[PipelineStage.SEAL]
        assert PipelineStage.SECOND_ROUND in allowed


class TestAuditEvent:
    """Testet AuditEvent."""

    def test_creates_event_id(self) -> None:
        event = AuditEvent(stage="request")
        assert event.event_id != ""
        assert len(event.event_id) == 16

    def test_timestamp_has_tz(self) -> None:
        event = AuditEvent(stage="request")
        assert event.timestamp.tzinfo is not None

    def test_default_empty_inputs_outputs(self) -> None:
        event = AuditEvent(stage="test")
        assert event.inputs == {}
        assert event.outputs == {}
        assert event.duration_ms == 0.0

    def test_with_fields(self) -> None:
        event = AuditEvent(
            stage="first_round",
            inputs={"agent_count": 3},
            outputs={"report_count": 3},
            duration_ms=42.0,
        )
        assert event.inputs == {"agent_count": 3}
        assert event.outputs == {"report_count": 3}
        assert event.duration_ms == 42.0

    def test_immutability(self) -> None:
        event = AuditEvent(stage="test")
        with pytest.raises(Exception):
            event.stage = "changed"


class TestTradingGraphState:
    """Testet TradingGraphState."""

    def test_default_values(self) -> None:
        state = create_initial_state("run-1", "BTC/USD")
        assert state.run_id == "run-1"
        assert state.instrument == "BTC/USD"
        assert state.current_stage == PipelineStage.REQUEST
        assert state.first_round_reports == []
        assert state.errors == []
        assert state.warnings == []

    def test_frozen(self) -> None:
        state = create_initial_state("r1", "BTC/USD")
        with pytest.raises(Exception):
            state.run_id = "changed"

    def test_market_snapshot_default(self) -> None:
        state = create_initial_state("r1", "BTC/USD")
        assert state.market_snapshot == {}

    def test_with_market_snapshot(self) -> None:
        snapshot = {"price": 50000.0, "volume": 1000}
        state = create_initial_state("r1", "BTC/USD", market_data=snapshot)
        assert state.market_snapshot == snapshot

    def test_full_state_fields(self) -> None:
        state = create_initial_state("r1", "BTC/USD")
        assert state.second_round_reports == []
        assert state.seal_records == []
        assert state.round_summary == {}
        assert state.consensus_result == {}
        assert state.decision == ""


class TestOrchestratorGraph:
    """Testet OrchestratorGraph StageManager."""

    def test_initial_state(self) -> None:
        graph = OrchestratorGraph()
        assert graph.current_stage is None
        assert graph.stage_count() == 0
        assert graph.audit_events == ()
        assert graph.transition_log == []

    def test_can_transition_request(self) -> None:
        graph = OrchestratorGraph()
        assert graph.can_transition(PipelineStage.REQUEST) is True

    def test_cannot_transition_first_round_directly(self) -> None:
        graph = OrchestratorGraph()
        assert graph.can_transition(PipelineStage.FIRST_ROUND) is False

    def test_transition_to_request(self) -> None:
        graph = OrchestratorGraph()
        event = graph.transition(PipelineStage.REQUEST)
        assert graph.current_stage == PipelineStage.REQUEST
        assert graph.stage_count() == 1
        assert event.stage == "request"

    def test_sequential_transitions(self) -> None:
        graph = OrchestratorGraph()
        for stage in [
            PipelineStage.REQUEST,
            PipelineStage.FIRST_ROUND,
            PipelineStage.SEAL,
            PipelineStage.SECOND_ROUND,
            PipelineStage.CONSENSUS,
            PipelineStage.DECISION,
            PipelineStage.PUBLISH,
        ]:
            graph.transition(stage)
        assert graph.current_stage == PipelineStage.PUBLISH
        assert graph.stage_count() == 7

    def test_invalid_transition_raises(self) -> None:
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        with pytest.raises(ValueError, match="Ungueltiger Uebergang"):
            graph.transition(PipelineStage.SEAL)

    def test_has_completed(self) -> None:
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        assert graph.has_completed(PipelineStage.REQUEST) is True
        assert graph.has_completed(PipelineStage.FIRST_ROUND) is False

    def test_transition_log_records_all(self) -> None:
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        graph.transition(PipelineStage.FIRST_ROUND)
        assert len(graph.transition_log) == 2
        assert graph.transition_log[0] == (None, PipelineStage.REQUEST)
        assert graph.transition_log[1] == (PipelineStage.REQUEST, PipelineStage.FIRST_ROUND)

    def test_seal_hash(self) -> None:
        graph = OrchestratorGraph()
        h = graph.seal("test-data")
        assert len(h) == 64  # SHA-256 hex length
        assert h == graph.seal("test-data")  # deterministic

    def test_transition_with_inputs_outputs(self) -> None:
        graph = OrchestratorGraph()
        event = graph.transition(
            PipelineStage.REQUEST,
            inputs={"instrument": "BTC/USD"},
            outputs={"run_id": "r1"},
            duration_ms=100.0,
        )
        assert event.inputs == {"instrument": "BTC/USD"}
        assert event.outputs == {"run_id": "r1"}
        assert event.duration_ms == 100.0


class TestRunFirstRound:
    """Testet run_first_round."""

    def test_raises_on_no_agents(self) -> None:
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        with pytest.raises(ValueError, match="at least one agent"):
            run_first_round(state, graph, [], {})

    def test_stores_reports(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id="test-agent")]
        state, graph = run_first_round(state, graph, agents, {})
        assert len(state.first_round_reports) == 1
        assert state.first_round_reports[0].agent_id == "test-agent"
        assert state.current_stage == PipelineStage.FIRST_ROUND.value
        assert graph.has_completed(PipelineStage.FIRST_ROUND)

    def test_no_peer_contamination(self) -> None:
        """Jeder Agent bekommt nur market_data — keine Reports anderer."""
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id=f"agent-{i}") for i in range(3)]
        state, graph = run_first_round(state, graph, agents, {})
        assert len(state.first_round_reports) == 3
        # Kein Agent sollte einen anderen Report enthalten
        report_ids = {r.report_id for r in state.first_round_reports}
        assert len(report_ids) == 3  # alle unique

    def test_audit_event_created(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id="a1")]
        run_first_round(state, graph, agents, {})
        assert graph.stage_count() == 2  # request + first_round
        events = [e.stage for e in graph.audit_events]
        assert "request" in events
        assert "first_round" in events
