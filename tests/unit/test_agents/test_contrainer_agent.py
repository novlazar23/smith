"""Tests for the contrarian agent."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from packages.agents.contrainer.agent import ContrarianAgent
from packages.agents.contrainer.models import (
    ContrarianConfig,
    ContrarianHypothesis,
)
from packages.consensus.base import VoteDirection
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
)


def _make_report(
    agent_id: str = "test_agent",
    probabilities: dict[str, float] | None = None,
    status: AgentStatus = AgentStatus.ACTIVE,
) -> AgentReport:
    """Helper to create test AgentReports."""
    return AgentReport(
        report_id=f"test-{agent_id}",
        run_id="test-run",
        agent_id=agent_id,
        agent_version="0.1.0",
        instrument="BTC/USDT",
        horizon="1h",
        as_of=datetime.now(UTC),
        hypothesis="Test hypothesis",
        probabilities=probabilities or {"up": 0.6, "down": 0.2, "range": 0.2},
        evidence=[
            EvidenceReference(
                reference="test:evid",
                feature="test_feature",
                value="test_value",
                direction="up",
                relevance=0.8,
            ),
        ],
        invalidations=[],
        status=status,
        raw_confidence=0.7,
        calibrated_confidence=0.7,
    )


class TestContrarianAgentBasic:
    """Tests for basic contrarian functionality."""

    def test_inverts_long_majority(self):
        agent = ContrarianAgent()
        reports = [_make_report("agent1", {"up": 0.7, "down": 0.15, "range": 0.15})]
        hypothesis, report = agent.analyze(reports)

        assert hypothesis.minority_direction == VoteDirection.SHORT
        assert report.probabilities["down"] >= 0.5

    def test_inverts_short_majority(self):
        agent = ContrarianAgent()
        reports = [_make_report("agent1", {"up": 0.15, "down": 0.7, "range": 0.15})]
        hypothesis, report = agent.analyze(reports)

        assert hypothesis.minority_direction == VoteDirection.LONG
        assert report.probabilities["up"] >= 0.5

    def test_empty_reports_raises(self):
        agent = ContrarianAgent()
        with pytest.raises(ValueError, match="must not be empty"):
            agent.analyze([])

    def test_all_shadow_agents_abstains(self):
        agent = ContrarianAgent()
        reports = [
            _make_report("s1", status=AgentStatus.SHADOW),
            _make_report("s2", status=AgentStatus.SHADOW),
        ]
        hypothesis, report = agent.analyze(reports)

        assert hypothesis.confidence == 0.0
        assert report.status == AgentStatus.SHADOW

    def test_agent_id_property(self):
        agent = ContrarianAgent(
            ContrarianConfig(agent_id="my-contrarian")
        )
        assert agent.agent_id == "my-contrarian"


class TestContrarianHypothesis:
    """Tests for hypothesis structure."""

    def test_hypothesis_fields(self):
        agent = ContrarianAgent()
        reports = [_make_report("agent1", {"up": 0.7, "down": 0.15, "range": 0.15})]
        hypothesis, _ = agent.analyze(reports)

        assert isinstance(hypothesis, ContrarianHypothesis)
        assert hypothesis.counter_argument
        assert 0.0 <= hypothesis.confidence <= 1.0
        assert hypothesis.evidence
        assert hypothesis.majority_direction
        assert hypothesis.minority_direction

    def test_counter_argument_for_long(self):
        agent = ContrarianAgent()
        reports = [_make_report("a1", {"up": 0.7, "down": 0.15, "range": 0.15})]
        hypothesis, _ = agent.analyze(reports)

        assert "bearish" in hypothesis.counter_argument.lower() or \
               "reversal" in hypothesis.counter_argument.lower()

    def test_confidence_based_on_minority(self):
        # 1 of 3 agents is minority → minority_ratio = 1/3 = 0.33
        reports = [
            _make_report("a1", {"up": 0.7, "down": 0.15, "range": 0.15}),
            _make_report("a2", {"up": 0.7, "down": 0.15, "range": 0.15}),
            _make_report("a3", {"up": 0.15, "down": 0.7, "range": 0.15}),
        ]
        agent = ContrarianAgent()
        hypothesis, _ = agent.analyze(reports)

        assert hypothesis.confidence > 0.0
        assert hypothesis.confidence <= 1.0


class TestContrarianReport:
    """Tests for generated AgentReport."""

    def test_report_has_shadow_status(self):
        agent = ContrarianAgent()
        reports = [_make_report("a1", {"up": 0.7, "down": 0.15, "range": 0.15})]
        _, report = agent.analyze(reports)

        assert report.status == AgentStatus.SHADOW
        assert report.agent_id == "contrarian"

    def test_report_probabilities_sum(self):
        agent = ContrarianAgent()
        reports = [_make_report("a1", {"up": 0.7, "down": 0.15, "range": 0.15})]
        _, report = agent.analyze(reports)

        total = sum(report.probabilities.values())
        assert abs(total - 1.0) < 0.01
