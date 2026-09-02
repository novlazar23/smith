"""Tests für das Agenten-Ensemble und den ContextualAgent-Adapter."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from apps.orchestrator_service.service import (
    ContextualAgent,
    build_ensemble,
)
from numpy.typing import NDArray
from packages.agents.base import AgentConfig, AgentType, BaseAgent
from packages.orchestrator.second_round import RoundContext
from packages.schemas.agent_report import AgentReport, AgentStatus, EvidenceReference


def make_report(agent_id: str = "dummy") -> AgentReport:
    """Baut einen minimal gültigen AgentReport."""
    return AgentReport(
        report_id="report-1",
        run_id="run-1",
        agent_id=agent_id,
        agent_version="0.1.0",
        instrument="BTC/USDT",
        horizon="15m",
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        hypothesis="test hypothesis",
        probabilities={"up": 0.4, "down": 0.3, "range": 0.3},
        evidence=[
            EvidenceReference(
                reference="dummy:feature",
                feature="feature",
                value="value",
                direction="neutral",
                relevance=0.5,
            )
        ],
        status=AgentStatus.SHADOW,
    )


class DummyAgent(BaseAgent):
    """Minimal-Agent, der analyze-Aufrufe zählt."""

    def __init__(self) -> None:
        super().__init__(AgentConfig(agent_id="dummy", agent_type=AgentType.INDICATOR))
        self.analyze_calls = 0

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        del data
        self.analyze_calls += 1
        return make_report(self.agent_id)


class TestContextualAgent:
    """Der Adapter stellt analyze und analyze_with_context bereit."""

    def test_analyze_delegates_to_wrapped_agent(self) -> None:
        """analyze() delegiert an den umschlossenen Agenten."""
        inner = DummyAgent()
        wrapper = ContextualAgent(inner)
        data: dict[str, NDArray[np.float64]] = {"close": np.array([1.0, 2.0])}

        report = wrapper.analyze(data)

        assert inner.analyze_calls == 1
        assert isinstance(report, AgentReport)
        assert wrapper.agent_id == "dummy"

    def test_analyze_with_context_delegates_to_analyze(self) -> None:
        """analyze_with_context() nutzt dieselbe deterministische OHLCV-Analyse."""
        inner = DummyAgent()
        wrapper = ContextualAgent(inner)
        data: dict[str, NDArray[np.float64]] = {"close": np.array([1.0, 2.0])}
        context = RoundContext(first_round_summary={"direction": "RANGE", "confidence": 0.2})

        report = wrapper.analyze_with_context(context, data)

        assert inner.analyze_calls == 1
        assert isinstance(report, AgentReport)

    def test_second_round_interface_available(self) -> None:
        """Jedes Ensemble-Mitglied besitzt die von run_second_round benötigte Methode."""
        agents = build_ensemble("BTC/USDT", "15m")
        for agent in agents:
            assert hasattr(agent, "analyze")
            assert hasattr(agent, "analyze_with_context")


class TestBuildEnsemble:
    """build_ensemble() erzeugt frische, korrekt konfigurierte Shadow-Agenten."""

    def test_returns_four_configured_shadow_agents(self) -> None:
        """Ensemble besteht aus den vier kanonischen Blickwinkel-Agenten."""
        agents = build_ensemble("BTC/USDT", "15m")

        assert len(agents) == 4
        assert {agent.agent_id for agent in agents} == {
            "trend",
            "mean_reversion",
            "volatility_regime",
            "volume_conviction",
        }
        inner_agents = [agent._agent for agent in agents]  # type: ignore[attr-defined]
        for inner in inner_agents:
            assert inner.config.status is AgentStatus.SHADOW
            assert inner.config.instrument == "BTC/USDT"
            assert inner.config.horizon == "15m"

    def test_build_ensemble_active_status(self) -> None:
        """Mit agent_status=ACTIVE erhalten alle Agenten den ACTIVE-Status."""
        agents = build_ensemble("BTC/USDT", "15m", AgentStatus.ACTIVE)

        inner_agents = [agent._agent for agent in agents]  # type: ignore[attr-defined]
        for inner in inner_agents:
            assert inner.config.status is AgentStatus.ACTIVE

    def test_agent_types_match(self) -> None:
        """Die AgentTypen stimmen mit den gewählten Klassen überein."""
        agents = build_ensemble("ETH/USDT", "15m")
        types = {
            agent.agent_id: agent._agent.config.agent_type  # type: ignore[attr-defined]
            for agent in agents
        }
        assert types == {
            "trend": AgentType.INDICATOR,
            "mean_reversion": AgentType.INDICATOR,
            "volatility_regime": AgentType.REGIME,
            "volume_conviction": AgentType.ORDERFLOW,
        }

    def test_fresh_instances_per_cycle(self) -> None:
        """Jeder Aufruf erzeugt neue Agent-Instanzen (keine Zustandsübernahme)."""
        first = build_ensemble("BTC/USDT", "15m")
        second = build_ensemble("BTC/USDT", "15m")

        first_inner = [agent._agent for agent in first]  # type: ignore[attr-defined]
        second_inner = [agent._agent for agent in second]  # type: ignore[attr-defined]
        for a, b in zip(first_inner, second_inner, strict=True):
            assert a is not b
