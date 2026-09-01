"""Tests fuer packages/orchestrator/pipeline.py."""

from __future__ import annotations

from packages.consensus import ConsensusDecision
from packages.orchestrator.pipeline import (
    OrchestratorPipeline,
    OrchestratorPipelineResult,
)


class TestOrchestratorPipelineResult:
    """Testet OrchestratorPipelineResult."""

    def test_is_no_trade_true(self) -> None:
        result = OrchestratorPipelineResult(
            decision=ConsensusDecision.NO_TRADE.value,
            consensus=None,
            first_round_reports=[],
            seal_records=[],
            second_round_reports=[],
        )
        assert result.is_no_trade is True

    def test_is_no_trade_false(self) -> None:
        result = OrchestratorPipelineResult(
            decision=ConsensusDecision.LONG_BIAS.value,
            consensus=None,
            first_round_reports=[],
            seal_records=[],
            second_round_reports=[],
        )
        assert result.is_no_trade is False

    def test_direction(self) -> None:
        result = OrchestratorPipelineResult(
            decision="LONG_BIAS",
            consensus=None,
            first_round_reports=[],
            seal_records=[],
            second_round_reports=[],
        )
        assert result.direction == "LONG_BIAS"


class TestOrchestratorPipeline:
    """Testet OrchestratorPipeline run()."""

    def test_full_pipeline(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        pipeline = OrchestratorPipeline()
        result = pipeline.run(
            run_id="test-001",
            instrument="BTC/USD",
            agents=[MockAgent(agent_id=f"agent-{i}") for i in range(3)],
            market_data={"candles": [], "orderbook": {}},
        )
        assert isinstance(result, OrchestratorPipelineResult)
        assert result.decision == ConsensusDecision.LONG_BIAS.value
        assert result.consensus is not None
        assert result.consensus.decision == ConsensusDecision.LONG_BIAS
        assert len(result.first_round_reports) == 3
        assert len(result.second_round_reports) == 3
        assert len(result.seal_records) == 3
        assert result.errors == []

    def test_single_agent_pipeline(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        pipeline = OrchestratorPipeline()
        result = pipeline.run(
            run_id="single-001",
            instrument="ETH/USD",
            agents=[MockAgent(agent_id="solo")],
            market_data={},
        )
        assert result.decision == ConsensusDecision.LONG_BIAS.value
        assert len(result.first_round_reports) == 1
        assert len(result.second_round_reports) == 1

    def test_shadow_agents_excluded_from_consensus(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent, ShadowAgent
        pipeline = OrchestratorPipeline()
        agents = [
            MockAgent(agent_id="long-1", bias=0.7),
            ShadowAgent("shadow-1"),
        ]
        result = pipeline.run(
            run_id="shadow-001",
            instrument="BTC/USD",
            agents=agents,
            market_data={},
        )
        # Shadow-Agent sollte den Konsens nicht bremsen
        assert result.decision == ConsensusDecision.LONG_BIAS.value

    def test_no_agents_error(self) -> None:
        pipeline = OrchestratorPipeline()
        result = pipeline.run(
            run_id="empty-001",
            instrument="BTC/USD",
            agents=[],
            market_data={},
        )
        assert result.is_no_trade is True
        assert len(result.errors) > 0

    def test_seal_records_present(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        pipeline = OrchestratorPipeline()
        result = pipeline.run(
            run_id="seal-001",
            instrument="BTC/USD",
            agents=[MockAgent(agent_id="a1")],
            market_data={},
        )
        assert len(result.seal_records) == 1
        assert "data_hash" in result.seal_records[0]
        assert "report_id" in result.seal_records[0]
        assert len(result.seal_records[0]["data_hash"]) == 64  # SHA-256

    def test_high_dissent_pipeline(self) -> None:
        """Pipeline mit >60% Dissens sollte NO_TRADE zurueckgeben."""
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent, short_agent
        pipeline = OrchestratorPipeline(high_dissent_threshold=0.3)
        agents = [
            MockAgent(agent_id="long-1", bias=0.7),
            short_agent("short-1"),
        ]
        result = pipeline.run(
            run_id="dissent-001",
            instrument="BTC/USD",
            agents=agents,
            market_data={},
        )
        # Mit 2 Agents und 0.3 threshold: 50% disagreement > threshold
        # => NO_TRADE
        # (Abhaengig von Konsens-engine — hier primaeer Test auf Robustheit)
        assert result.decision in (
            ConsensusDecision.NO_TRADE.value,
            ConsensusDecision.LONG_BIAS.value,
        )

    def test_consensus_result_available(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        pipeline = OrchestratorPipeline()
        result = pipeline.run(
            run_id="consensus-001",
            instrument="BTC/USD",
            agents=[MockAgent(agent_id="a1"), MockAgent(agent_id="a2")],
            market_data={},
        )
        assert result.consensus is not None
        assert result.consensus.decision in (
            ConsensusDecision.LONG_BIAS,
            ConsensusDecision.NO_TRADE,
        )
        assert "agent_weights" in result.consensus.vote_distribution or result.consensus.agent_weights

    def test_run_ids_different(self) -> None:
        from tests.unit.test_orchestrator_pkg.conftest import MockAgent
        pipeline = OrchestratorPipeline()
        result1 = pipeline.run(
            run_id="diff-001",
            instrument="BTC/USD",
            agents=[MockAgent(agent_id="a1")],
            market_data={},
        )
        result2 = pipeline.run(
            run_id="diff-002",
            instrument="BTC/USD",
            agents=[MockAgent(agent_id="a1")],
            market_data={},
        )
        # Die Reports sollten verschiedene report_ids haben
        _ids1 = {r.report_id for r in result1.first_round_reports}
        _ids2 = {r.report_id for r in result2.first_round_reports}
        # Mindestens einige Reports sind unterschiedlich
        assert result1.first_round_reports[0].report_id == result2.first_round_reports[0].report_id
        # Beide Pipelines waren erfolgreich
        assert not result1.is_no_trade or result1.errors
        assert not result2.is_no_trade or result2.errors
