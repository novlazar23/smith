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
from packages.agents.mean_reversion_agent import MeanReversionAgent, MeanReversionParams
from packages.agents.trend_agent import TrendAgent, TrendParams
from packages.agents.volatility_regime_agent import VolatilityRegimeAgent, VolatilityRegimeParams
from packages.agents.volume_conviction_agent import VolumeConvictionAgent, VolumeConvictionParams
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

    def test_build_ensemble_respects_champion_status_overrides(self) -> None:
        """status_overrides überschreibt den Basis-Status nur für benannte Agenten."""
        agents = build_ensemble(
            "BTC/USDT",
            "15m",
            AgentStatus.ACTIVE,
            status_overrides={"trend": AgentStatus.SHADOW},
        )

        statuses = {
            agent.agent_id: agent._agent.config.status  # type: ignore[attr-defined]
            for agent in agents
        }

        assert statuses["trend"] is AgentStatus.SHADOW
        assert statuses["mean_reversion"] is AgentStatus.ACTIVE
        assert statuses["volatility_regime"] is AgentStatus.ACTIVE
        assert statuses["volume_conviction"] is AgentStatus.ACTIVE

    def test_injects_champion_params_for_named_agent(self) -> None:
        """champion_params injiziert evolvierte Parameter in benannte Agenten."""
        agents = build_ensemble(
            "BTC/USDT",
            "15m",
            champion_params={"trend": {"ema_fast": 7, "p_cap": 0.9}},
        )
        inner = {agent.agent_id: agent._agent for agent in agents}  # type: ignore[attr-defined]

        assert isinstance(inner["trend"], TrendAgent)
        assert inner["trend"]._params.ema_fast == 7
        assert inner["trend"]._params.p_cap == 0.9
        # Nicht benannte Agenten behalten die Defaults
        assert isinstance(inner["mean_reversion"], MeanReversionAgent)
        assert inner["mean_reversion"]._params == MeanReversionParams()
        assert isinstance(inner["volatility_regime"], VolatilityRegimeAgent)
        assert inner["volatility_regime"]._params == VolatilityRegimeParams()
        assert isinstance(inner["volume_conviction"], VolumeConvictionAgent)
        assert inner["volume_conviction"]._params == VolumeConvictionParams()

    def test_without_champion_params_uses_defaults(self) -> None:
        """Ohne champion_params bleibt das Verhalten unverändert (Defaults)."""
        agents = build_ensemble("BTC/USDT", "15m")
        inner = {agent.agent_id: agent._agent for agent in agents}  # type: ignore[attr-defined]

        assert isinstance(inner["trend"], TrendAgent)
        assert inner["trend"]._params == TrendParams()

    def test_unknown_agent_id_in_champion_params_ignored(self) -> None:
        """Unbekannte agent_ids im Champion-Satz werden stillschweigend ignoriert."""
        agents = build_ensemble(
            "BTC/USDT",
            "15m",
            champion_params={"kein_agent": {"foo": 1}, "trend": {"ema_fast": 9}},
        )
        inner = {agent.agent_id: agent._agent for agent in agents}  # type: ignore[attr-defined]

        assert isinstance(inner["trend"], TrendAgent)
        assert inner["trend"]._params.ema_fast == 9
        assert isinstance(inner["mean_reversion"], MeanReversionAgent)
        assert inner["mean_reversion"]._params == MeanReversionParams()

    def test_empty_params_mapping_uses_defaults(self) -> None:
        """Leerer Parameter-Mapping = kein Satz → Defaults."""
        agents = build_ensemble("BTC/USDT", "15m", champion_params={"trend": {}})
        inner = {agent.agent_id: agent._agent for agent in agents}  # type: ignore[attr-defined]

        assert isinstance(inner["trend"], TrendAgent)
        assert inner["trend"]._params == TrendParams()

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


EVOLVED_CODE = """import numpy as np

def predict(open, high, low, close, volume):
    m = float(close[-1] - close[-6])
    if m > 0:
        return (0.8, 0.1, 0.1)
    if m < 0:
        return (0.1, 0.8, 0.1)
    return (0.34, 0.33, 0.33)
"""


class TestBuildEnsembleEvolvedAgents:
    """evolved_agents hängt LLM-generierte Agenten als SHADOW-Mitglieder an."""

    def test_appends_evolved_agent_as_shadow(self) -> None:
        """Ein gültiger Evolved Agent wird 5. Mitglied mit SHADOW-Status."""
        agents = build_ensemble("BTC/USDT", "15m", AgentStatus.ACTIVE, evolved_agents={"momentum_test": EVOLVED_CODE})

        assert len(agents) == 5
        inner = {agent.agent_id: agent._agent for agent in agents}  # type: ignore[attr-defined]
        assert inner["momentum_test"].config.status is AgentStatus.SHADOW
        assert inner["momentum_test"].config.instrument == "BTC/USDT"
        assert inner["trend"].config.status is AgentStatus.ACTIVE

    def test_evolved_agent_produces_valid_report(self) -> None:
        """Der Evolved Agent liefert einen gültigen Report (Summe 1, Evidenz)."""
        agents = build_ensemble("BTC/USDT", "15m", AgentStatus.ACTIVE, evolved_agents={"momentum_test": EVOLVED_CODE})
        data = {key: np.ones(50) for key in ("open", "high", "low", "close", "volume")}
        data["close"] = np.linspace(100, 105, 50)

        report = agents[4].analyze(data)

        assert abs(sum(report.probabilities.values()) - 1.0) < 1e-6
        assert report.status is AgentStatus.SHADOW
        assert len(report.evidence) >= 1

    def test_invalid_evolved_code_rejected_without_breaking_ensemble(self) -> None:
        """Defekter Code wird verworfen, das Ensemble bleibt intakt."""
        agents = build_ensemble(
            "BTC/USDT",
            "15m",
            AgentStatus.ACTIVE,
            evolved_agents={"evil": "def predict(o,h,l,c,v):\n    open('x')\n    return (1,0,0)"},
        )

        assert len(agents) == 4
        assert all(agent.agent_id != "evil" for agent in agents)
