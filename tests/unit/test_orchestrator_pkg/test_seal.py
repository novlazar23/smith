"""Tests fuer packages/orchestrator/seal.py."""

from __future__ import annotations

import pytest
from packages.orchestrator.graph import (
    OrchestratorGraph,
    PipelineStage,
    create_initial_state,
    run_first_round,
)
from packages.orchestrator.seal import (
    SealRecord,
    _hash_report,
    seal_first_round,
    verify_seal,
)
from packages.schemas.agent_report import AgentReport, EvidenceReference
from datetime import UTC, datetime


def _make_report(report_id: str = "r1", agent_id: str = "agent-1") -> AgentReport:
    return AgentReport(
        report_id=report_id,
        run_id="test-run",
        agent_id=agent_id,
        agent_version="1.0.0",
        instrument="BTC/USD",
        horizon="1h",
        as_of=datetime.now(UTC),
        hypothesis="bullish",
        probabilities={"up": 0.65, "down": 0.15, "range": 0.20},
        evidence=[EvidenceReference(
            reference=f"{agent_id}:price",
            feature="price",
            value="rising",
            direction="positive",
            relevance=0.8,
        )],
        raw_confidence=0.7,
    )


class TestHashReport:
    """Testet _hash_report."""

    def test_hash_is_64_chars(self) -> None:
        report = _make_report()
        h = _hash_report(report)
        assert len(h) == 64  # SHA-256 hex

    def test_hash_deterministic(self) -> None:
        report = _make_report()
        h1 = _hash_report(report)
        h2 = _hash_report(report)
        assert h1 == h2

    def test_different_reports_different_hashes(self) -> None:
        r1 = _make_report(report_id="r1", agent_id="agent-1")
        r2 = _make_report(report_id="r2", agent_id="agent-2")
        h1 = _hash_report(r1)
        h2 = _hash_report(r2)
        assert h1 != h2

    def test_different_hypothesis_different_hash(self) -> None:
        report = _make_report()
        # Report ist frozen, use model_copy
        modified = report.model_copy(update={"hypothesis": "different"})
        assert _hash_report(report) != _hash_report(modified)


class TestVerifySeal:
    """Testet verify_seal."""

    def test_valid_seal(self) -> None:
        report = _make_report()
        h = _hash_report(report)
        assert verify_seal(report, h) is True

    def test_tampered_seal(self) -> None:
        report = _make_report()
        assert verify_seal(report, "00" * 32) is False

    def test_modified_report_fails(self) -> None:
        report = _make_report()
        h = _hash_report(report)
        # Report aendern — Hash passt nicht mehr
        modified = report.model_copy(update={"hypothesis": "tampered"})
        assert verify_seal(modified, h) is False


class TestSealRecord:
    """Testet SealRecord."""

    def test_default_timestamp(self) -> None:
        record = SealRecord(
            data_hash="abc123",
            report_id="r1",
        )
        assert record.timestamp.tzinfo is not None
        assert record.timestamp_iso != ""

    def test_frozen(self) -> None:
        record = SealRecord(data_hash="abc", report_id="r1")
        with pytest.raises(Exception):
            record.data_hash = "changed"


class TestSealFirstRound:
    """Testet seal_first_round."""

    def test_creates_seal_records(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id="agent-1")]
        state, graph = run_first_round(state, graph, agents, {})

        state, graph, records = seal_first_round(state, graph)
        assert len(records) == 1
        assert records[0].data_hash != ""
        assert records[0].report_id == "agent-1-r1"
        assert state.first_round_hash != ""

    def test_raises_on_empty_reports(self) -> None:
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        with pytest.raises(ValueError, match="must not be empty"):
            seal_first_round(state, graph)

    def test_multiple_reports_hashed(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id=f"agent-{i}") for i in range(3)]
        state, graph = run_first_round(state, graph, agents, {})

        state, graph, records = seal_first_round(state, graph)
        assert len(records) == 3
        # Alle Hashes unique
        hashes = [r.data_hash for r in records]
        assert len(set(hashes)) == 3

    def test_sealed_state_updated(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id="a1")]
        state, graph = run_first_round(state, graph, agents, {})

        state, graph, records = seal_first_round(state, graph)
        assert state.current_stage == PipelineStage.SEAL.value
        assert state.seal_records == [
            {
                "data_hash": records[0].data_hash,
                "report_id": records[0].report_id,
                "timestamp_iso": records[0].timestamp_iso,
            }
        ]

    def test_seal_is_immutable(self) -> None:
        """Seal kann nicht nachtraeglich geaendert werden — State ist frozen."""
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id="a1")]
        state, graph = run_first_round(state, graph, agents, {})

        state, graph, records = seal_first_round(state, graph)
        # State ist frozen — SealRecord ist frozen
        with pytest.raises(Exception):
            state.seal_records = []
        with pytest.raises(Exception):
            records[0].data_hash = "tampered"

    def test_verification_after_seal(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id="a1")]
        state, graph = run_first_round(state, graph, agents, {})

        state, graph, records = seal_first_round(state, graph)
        report = state.first_round_reports[0]
        assert verify_seal(report, records[0].data_hash) is True

    def test_audit_event_created(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        state = create_initial_state("r1", "BTC/USD")
        graph = OrchestratorGraph()
        graph.transition(PipelineStage.REQUEST)
        agents = [MockAgent(agent_id="a1")]
        state, graph = run_first_round(state, graph, agents, {})

        seal_first_round(state, graph)
        events = [e.stage for e in graph.audit_events]
        assert "seal" in events
        assert "request" in events
        assert "first_round" in events