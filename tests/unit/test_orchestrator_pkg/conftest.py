"""Test fixtures: mock agent for orchestrator package tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
)


class MockAgent:
    """Minimal Agent for testing — supports analyze() and analyze_with_context()."""

    def __init__(
        self,
        agent_id: str,
        agent_version: str = "1.0.0",
        instrument: str = "BTC/USD",
        horizon: str = "1h",
        status: AgentStatus = AgentStatus.ACTIVE,
        bias: float = 0.65,
    ) -> None:
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.instrument = instrument
        self.horizon = horizon
        self.status = status
        self.bias = bias  # up-probability bias

    def analyze(self, market_data: dict[str, Any]) -> AgentReport:
        """First-Round analyze — pure market data, no context."""
        up = self.bias
        down = round(0.15 * (1 - self.bias) / 0.65, 4)
        range_v = round(1.0 - up - down, 4)
        return AgentReport(
            report_id=f"{self.agent_id}-r1",
            run_id="test-run",
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            instrument=self.instrument,
            horizon=self.horizon,
            as_of=datetime.now(UTC),
            hypothesis=f"{self.agent_id} bullish",
            probabilities={"up": up, "down": down, "range": range_v},
            evidence=[EvidenceReference(
                reference=f"{self.agent_id}:price",
                feature="price",
                value="rising",
                direction="positive",
                relevance=0.8,
            )],
            raw_confidence=self.bias,
            status=self.status,
        )

    def analyze_with_context(
        self,
        context: Any,
        market_data: dict[str, Any],
    ) -> AgentReport:
        """Second-Round analyze — with Round-1 summary context."""
        up = self.bias * 1.05  # slight context influence
        if up > 0.95:
            up = 0.95
        down = round(0.10 * (1 - up) / 0.7, 4)
        range_v = round(1.0 - up - down, 4)
        return AgentReport(
            report_id=f"{self.agent_id}-r2",
            run_id="test-run",
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            instrument=self.instrument,
            horizon=self.horizon,
            as_of=datetime.now(UTC),
            hypothesis=f"{self.agent_id} bullish v2",
            probabilities={"up": up, "down": down, "range": range_v},
            evidence=[EvidenceReference(
                reference=f"{self.agent_id}:price",
                feature="price",
                value="rising",
                direction="positive",
                relevance=0.85,
            )],
            raw_confidence=min(up, 0.95),
            status=self.status,
        )


class ShadowAgent:
    """Shadow agent for testing weight=0 behavior."""

    def __init__(self, agent_id: str = "shadow-agent") -> None:
        self.agent_id = agent_id

    def analyze(self, market_data: dict[str, Any]) -> AgentReport:
        return AgentReport(
            report_id=f"{self.agent_id}-r1",
            run_id="test-run",
            agent_id=self.agent_id,
            agent_version="1.0.0",
            instrument="BTC/USD",
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis="shadow analysis",
            probabilities={"up": 0.6, "down": 0.15, "range": 0.25},
            evidence=[EvidenceReference(
                reference=f"{self.agent_id}:test",
                feature="test",
                value="shadow",
                direction="neutral",
                relevance=0.5,
            )],
            raw_confidence=0.5,
            status=AgentStatus.SHADOW,
        )

    def analyze_with_context(
        self,
        context: Any,
        market_data: dict[str, Any],
    ) -> AgentReport:
        return AgentReport(
            report_id=f"{self.agent_id}-r2",
            run_id="test-run",
            agent_id=self.agent_id,
            agent_version="1.0.0",
            instrument="BTC/USD",
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis="shadow v2",
            probabilities={"up": 0.55, "down": 0.2, "range": 0.25},
            evidence=[EvidenceReference(
                reference=f"{self.agent_id}:test",
                feature="test",
                value="shadow",
                direction="neutral",
                relevance=0.5,
            )],
            raw_confidence=0.4,
            status=AgentStatus.SHADOW,
        )


def short_agent(agent_id: str = "short-agent") -> MockAgent:
    """Shortcut for a short-biased agent."""
    return MockAgent(
        agent_id=agent_id,
        bias=0.2,
    )