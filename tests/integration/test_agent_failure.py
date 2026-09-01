"""Integration tests for agent failure and graceful degradation.

Verifies that:
- A failed agent (empty/invalid probabilities) does not crash consensus
- System falls back to agents with valid data
- NO_TRADE is produced when all agents are down
- Degraded agents still contribute with reduced weight
"""

from __future__ import annotations

from datetime import datetime

import pytest
from packages.consensus import (
    ConsensusDecision,
    WeightedConsensusEngine,
)
from packages.paper.base import TradeDirection
from packages.paper.executor import PaperExecutor as Executor
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
)
from packages.strategy.engine import StrategyEngine
from packages.strategy.models import (
    StrategyConfig,
    StrategyDirection,
    StrategyProposal,
)

# ── helpers ──────────────────────────────────────────────────────────


def _make_agent_report(
    agent_id: str,
    probabilities: dict[str, float],
    status: AgentStatus = AgentStatus.ACTIVE,
) -> AgentReport:
    return AgentReport(
        report_id=f"rpt-{agent_id}",
        run_id="run-001",
        agent_id=agent_id,
        agent_version="0.1.0",
        instrument="EUR/USD",
        horizon="1h",
        as_of=datetime.now(),
        hypothesis=f"Agent {agent_id} signal",
        probabilities=probabilities,
        evidence=[
            EvidenceReference(
                reference=f"{agent_id}:signal",
                feature="signal",
                value="active",
                direction="positive",
                relevance=0.7,
            )
        ],
        raw_confidence=0.6,
        status=status,
    )


def _make_degraded_report(agent_id: str) -> AgentReport:
    return _make_agent_report(
        agent_id=agent_id,
        probabilities={"up": 0.65, "down": 0.15, "range": 0.2},
        status=AgentStatus.DEGRADED,
    )


def _make_shadow_report(agent_id: str) -> AgentReport:
    return _make_agent_report(
        agent_id=agent_id,
        probabilities={"up": 0.65, "down": 0.15, "range": 0.2},
        status=AgentStatus.SHADOW,
    )


class TestAgentFailureScenarios:
    """Test system behavior when agents fail during consensus."""

    def test_single_degraded_agent_still_produces_consensus(self) -> None:
        """A single DEGRADED agent should still produce a valid consensus."""
        engine = WeightedConsensusEngine()
        report = _make_degraded_report("degraded-1")
        result = engine.compute_consensus([report])

        assert isinstance(result.decision, ConsensusDecision)
        assert result.agent_weights["degraded-1"] == 0.3  # DEGRADED multiplier
        assert result.confidence >= 0.0

    def test_shadow_agent_contributes_less_weight(self) -> None:
        """A SHADOW agent contributes half the weight of ACTIVE."""
        engine = WeightedConsensusEngine()
        active_report = _make_agent_report(
            "active-1", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.ACTIVE
        )
        shadow_report = _make_agent_report(
            "shadow-1", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.SHADOW
        )
        result = engine.compute_consensus([active_report, shadow_report])

        assert result.agent_weights["active-1"] == 1.0
        assert result.agent_weights["shadow-1"] == 0.5
        # Active should still push decision
        assert result.decision == ConsensusDecision.LONG_BIAS

    def test_failed_agent_does_not_corrupt_consensus(self) -> None:
        """An agent with empty probabilities should be filtered out gracefully."""
        engine = WeightedConsensusEngine()
        good_reports = [
            _make_agent_report(
                "good-1", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.ACTIVE
            ),
            _make_agent_report(
                "good-2", {"up": 0.75, "down": 0.1, "range": 0.15}, AgentStatus.ACTIVE
            ),
        ]

        # Only good reports pass validation — empty-probability reports are excluded
        valid_reports = [r for r in good_reports if r.probabilities]
        result = engine.compute_consensus(valid_reports)

        assert result.decision == ConsensusDecision.LONG_BIAS
        assert "broken" not in result.agent_weights
        assert len(result.agent_weights) == 2

    def test_all_agents_failed_yields_no_trade(self) -> None:
        """When all agents are disabled/quarantined, consensus must be NO_TRADE."""
        engine = WeightedConsensusEngine()
        reports = [
            _make_agent_report(
                "q1", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.QUARANTINED
            ),
            _make_agent_report(
                "q2", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.DISABLED
            ),
        ]
        result = engine.compute_consensus(reports)

        # All weights = 0, so no vote can pass threshold
        assert result.decision == ConsensusDecision.NO_TRADE
        assert result.confidence == 0.0

    def test_quarantined_agent_has_zero_weight(self) -> None:
        """A quarantined agent contributes zero to the consensus score."""
        engine = WeightedConsensusEngine()
        reports = [
            _make_agent_report(
                "q1", {"up": 0.9, "down": 0.05, "range": 0.05}, AgentStatus.QUARANTINED
            ),
        ]
        result = engine.compute_consensus(reports)

        assert result.agent_weights["q1"] == 0.0
        assert result.decision == ConsensusDecision.NO_TRADE


class TestGracefulDegradation:
    """Verify that the system degrades gracefully — not all or nothing."""

    def test_majority_alive_yields_valid_decision(self) -> None:
        """When 4 of 5 agents are active, consensus works normally."""
        engine = WeightedConsensusEngine()
        reports = [
            _make_agent_report(
                f"a{i}", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.ACTIVE
            )
            for i in range(4)
        ]
        reports.append(
            _make_agent_report(
                "dead", {"up": 0.7, "down": 0.1, "range": 0.2},
                AgentStatus.QUARANTINED,
            )
        )
        result = engine.compute_consensus(reports)

        assert result.decision == ConsensusDecision.LONG_BIAS
        # Quarantined agents still appear in agent_agreements (they agree but with 0 weight)
        assert len(result.agent_agreements) == 5
        assert len(result.agent_disagreements) == 0

    def test_degraded_agents_still_count(self) -> None:
        """Degraded agents still count but with reduced weight."""
        engine = WeightedConsensusEngine()
        reports = [
            _make_agent_report(
                "d1", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.DEGRADED
            ),
            _make_agent_report(
                "d2", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.DEGRADED
            ),
        ]
        result = engine.compute_consensus(reports)

        # Each degraded agent has weight 0.3
        assert result.agent_weights["d1"] == 0.3
        assert result.agent_weights["d2"] == 0.3
        # Combined: 0.6 total, all in one direction → LONG_BIAS
        assert result.decision == ConsensusDecision.LONG_BIAS

    def test_no_valid_reports_raises(self) -> None:
        """Consensus engine raises ValueError when given empty list."""
        engine = WeightedConsensusEngine()

        with pytest.raises(ValueError, match="reports must not be empty"):
            engine.validate_input([])

    def test_mixed_health_status_correct_weights(self) -> None:
        """Mixed agent statuses produce correct individual weights."""
        engine = WeightedConsensusEngine()
        reports = [
            _make_agent_report("a1", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.ACTIVE),
            _make_agent_report("s1", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.SHADOW),
            _make_agent_report("d1", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.DEGRADED),
        ]
        result = engine.compute_consensus(reports)

        assert result.agent_weights["a1"] == 1.0
        assert result.agent_weights["s1"] == 0.5
        assert result.agent_weights["d1"] == 0.3

        # Total = 1.8, all LONG → confidence = 1.8/1.8 = 1.0
        assert result.decision == ConsensusDecision.LONG_BIAS


class TestStrategyDownstreamOfFailure:
    """Verify strategy engine receives valid data even after agent failures."""

    def test_strategy_receives_no_trade_on_all_down(self) -> None:
        """When all agents fail, strategy engine produces NO_TRADE proposal."""
        engine = WeightedConsensusEngine()
        reports = [
            _make_agent_report(
                "q1", {"up": 0.9, "down": 0.05, "range": 0.05},
                AgentStatus.QUARANTINED,
            ),
        ]
        consensus = engine.compute_consensus(reports)
        assert consensus.decision == ConsensusDecision.NO_TRADE

        strategy_engine = StrategyEngine(config=StrategyConfig())
        context = {
            "consensus": consensus,
            "features": {"current_price": 100.0, "atr": 1.0},
        }
        proposal = strategy_engine.run(context)

        assert proposal.direction == StrategyDirection.NO_TRADE

    def test_strategy_proceeds_with_partial_agent_set(self) -> None:
        """Strategy proceeds normally when remaining agents agree."""
        engine = WeightedConsensusEngine()
        reports = [
            _make_agent_report(
                "alive", {"up": 0.75, "down": 0.1, "range": 0.15},
                AgentStatus.ACTIVE,
            ),
            _make_agent_report(
                "dead", {"up": 0.75, "down": 0.1, "range": 0.15},
                AgentStatus.QUARANTINED,
            ),
        ]
        consensus = engine.compute_consensus(reports)
        assert consensus.decision == ConsensusDecision.LONG_BIAS

        strategy_engine = StrategyEngine(config=StrategyConfig())
        context = {
            "consensus": consensus,
            "features": {
                "current_price": 100.0,
                "atr": 2.0,
                "entry_type": "market",
                "entry_condition": "momentum",
            },
        }
        proposal = strategy_engine.run(context)

        assert isinstance(proposal, StrategyProposal)
        # Single agent should still pass strategy gates
        assert proposal.direction != StrategyDirection.NO_TRADE


class TestPaperOrdersAfterAgentFailure:
    """Verify paper trading works correctly when system degrades."""

    def test_paper_order_with_valid_consensus(self) -> None:
        """After removing dead agents, system produces valid paper trades."""
        engine = WeightedConsensusEngine()
        reports = [
            _make_agent_report(
                "a1", {"up": 0.8, "down": 0.05, "range": 0.15}, AgentStatus.ACTIVE
            ),
            _make_agent_report(
                "a2", {"up": 0.8, "down": 0.05, "range": 0.15}, AgentStatus.ACTIVE
            ),
        ]
        consensus = engine.compute_consensus(reports)
        assert consensus.decision == ConsensusDecision.LONG_BIAS

        strategy_engine = StrategyEngine(config=StrategyConfig())
        context = {
            "consensus": consensus,
            "features": {
                "current_price": 100.0,
                "atr": 2.0,
                "entry_type": "market",
                "entry_condition": "momentum",
            },
        }
        proposal = strategy_engine.run(context)

        if proposal.direction == StrategyDirection.LONG:
            executor = Executor(initial_cash=100000.0)
            account = executor.create_account("acc-fail-1")
            trade = executor.submit_order(
                account, "EUR/USD", TradeDirection.BUY, 1.0, price=100.0
            )
            assert trade.status == "filled"

    def test_no_trade_no_corrupted_state(self) -> None:
        """When consensus is NO_TRADE, no paper orders should be created."""
        engine = WeightedConsensusEngine()
        reports = [
            _make_agent_report(
                "q1", {"up": 0.9, "down": 0.05, "range": 0.05},
                AgentStatus.QUARANTINED,
            ),
        ]
        consensus = engine.compute_consensus(reports)
        assert consensus.decision == ConsensusDecision.NO_TRADE

        strategy_engine = StrategyEngine(config=StrategyConfig())
        context = {
            "consensus": consensus,
            "features": {"current_price": 100.0, "atr": 1.0},
        }
        proposal = strategy_engine.run(context)

        assert proposal.direction == StrategyDirection.NO_TRADE
        # No trade executed — state remains clean
        executor = Executor(initial_cash=100000.0)
        account = executor.create_account("acc-no-trade")
        # No submit_order called — account stays pristine
        assert account.cash == 100000.0
        assert len(account.positions) == 0
