"""Tests fur Agentensystem — base agent und Analyse-Agenten."""

from __future__ import annotations

import uuid as uuid_mod

import numpy as np
import pytest
from packages.agents import (
    AgentConfig,
    AgentType,
    BaseAgent,
    ChartAgent,
    IndicatorAgent,
    OrderFlowAgent,
    RegimeAgent,
)
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
    InvalidationCondition,
)


def _make_sample_ohlcv(n: int = 100) -> dict[str, np.ndarray]:
    """Erstellt synthetische OHLCV-Daten fur Tests."""
    rng = np.random.RandomState(42)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + np.abs(rng.randn(n) * 0.3)
    low = close - np.abs(rng.randn(n) * 0.3)
    open_ = close - rng.randn(n) * 0.2
    volume = np.abs(rng.randn(n) * 1000) + 500
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


class TestAgentType:
    """Testet AgentType-Enum."""

    def test_agent_type_enum_values(self) -> None:
        assert AgentType.INDICATOR == "indicator"
        assert AgentType.REGIME == "regime"
        assert AgentType.CHART == "chart"
        assert AgentType.ORDERFLOW == "orderflow"


class TestAgentConfig:
    """Testet AgentConfig-Defaults."""

    def test_agent_config_defaults(self) -> None:
        config = AgentConfig(agent_id="test-agent", agent_type=AgentType.INDICATOR)
        assert config.agent_id == "test-agent"
        assert config.agent_version == "0.1.0"
        assert config.agent_type == AgentType.INDICATOR
        assert config.instrument == ""
        assert config.horizon == "1h"
        assert config.status == AgentStatus.SHADOW


class TestBaseAgent:
    """Testet BaseAgent-Hilfsfunktionen."""

    def test_base_agent_generate_report_id(self) -> None:
        config = AgentConfig(agent_id="test", agent_type=AgentType.INDICATOR)

        class DummyAgent(BaseAgent):
            def analyze(self, data: dict[str, np.ndarray]) -> AgentReport:
                return AgentReport(
                    report_id=self._generate_report_id(),
                    run_id=uuid_mod.uuid4().hex,
                    agent_id="dummy",
                    agent_version="0.1.0",
                    instrument="",
                    horizon="1h",
                    as_of=__import__("datetime").datetime.now(),
                    hypothesis="test",
                    probabilities={"up": 0.4, "down": 0.3, "range": 0.3},
                    evidence=[],
                )

        agent = DummyAgent(config)
        id1 = agent._generate_report_id()
        id2 = agent._generate_report_id()
        assert isinstance(id1, str)
        assert len(id1) > 0
        assert id1 != id2

    def test_base_agent_make_evidence(self) -> None:
        config = AgentConfig(agent_id="test", agent_type=AgentType.INDICATOR)

        class DummyAgent(BaseAgent):
            def analyze(self, data: dict[str, np.ndarray]) -> AgentReport:
                return AgentReport(
                    report_id=self._generate_report_id(),
                    run_id=uuid_mod.uuid4().hex,
                    agent_id="dummy",
                    agent_version="0.1.0",
                    instrument="",
                    horizon="1h",
                    as_of=__import__("datetime").datetime.now(),
                    hypothesis="test",
                    probabilities={"up": 0.4, "down": 0.3, "range": 0.3},
                    evidence=[],
                )

        agent = DummyAgent(config)
        ev = agent._make_evidence("RSI", "25.0", "positive", 0.7)
        assert isinstance(ev, EvidenceReference)
        assert ev.feature == "RSI"
        assert ev.value == "25.0"
        assert ev.direction == "positive"
        assert ev.relevance == 0.7
        assert ev.reference == "test:RSI"

    def test_base_agent_make_invalidations(self) -> None:
        config = AgentConfig(agent_id="test", agent_type=AgentType.INDICATOR)

        class DummyAgent(BaseAgent):
            def analyze(self, data: dict[str, np.ndarray]) -> AgentReport:
                return AgentReport(
                    report_id=self._generate_report_id(),
                    run_id=uuid_mod.uuid4().hex,
                    agent_id="dummy",
                    agent_version="0.1.0",
                    instrument="",
                    horizon="1h",
                    as_of=__import__("datetime").datetime.now(),
                    hypothesis="test",
                    probabilities={"up": 0.4, "down": 0.3, "range": 0.3},
                    evidence=[],
                )

        agent = DummyAgent(config)
        inv = agent._make_invalidations(
            condition="RSI andert sich",
            indicator="RSI",
            threshold=70.0,
            direction="above",
        )
        assert isinstance(inv, InvalidationCondition)
        assert inv.condition == "RSI andert sich"
        assert inv.indicator == "RSI"
        assert inv.threshold == 70.0
        assert inv.direction == "above"


class TestIndicatorAgent:
    """Testet IndicatorAgent."""

    def test_indicator_agent_basic_analysis(self) -> None:
        data = _make_sample_ohlcv(100)
        agent = IndicatorAgent()
        report = agent.analyze(data)
        assert report.agent_id == "indicator"
        assert "up" in report.probabilities
        assert "down" in report.probabilities
        assert "range" in report.probabilities

    def test_indicator_agent_probabilities_sum_to_one(self) -> None:
        data = _make_sample_ohlcv(100)
        agent = IndicatorAgent()
        report = agent.analyze(data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) <= 0.0001

    def test_indicator_agent_evidence_present(self) -> None:
        data = _make_sample_ohlcv(100)
        agent = IndicatorAgent()
        report = agent.analyze(data)
        assert len(report.evidence) >= 1

    def test_indicator_agent_invalid_data_raises(self) -> None:
        agent = IndicatorAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze({})  # type: ignore[arg-type]


class TestRegimeAgent:
    """Testet RegimeAgent."""

    def test_regime_agent_bull_regime(self) -> None:
        """Bull-Regime muss hohe 'up'-Wahrscheinlichkeit erzeugen."""
        close = np.array([100.0 + i * 2 for i in range(100)])
        high = close + np.abs(np.random.RandomState(42).randn(100) * 0.5)
        low = close - np.abs(np.random.RandomState(42).randn(100) * 0.5)
        data = {
            "open": close - np.random.RandomState(43).randn(100) * 0.2,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(100),
        }
        agent = RegimeAgent()
        report = agent.analyze(data)
        assert report.probabilities["up"] > report.probabilities["down"]

    def test_regime_agent_bear_regime(self) -> None:
        """Bear-Regime muss hohe 'down'-Wahrscheinlichkeit erzeugen."""
        close = np.array([100.0 - i * 2 for i in range(100)])
        high = close + np.abs(np.random.RandomState(42).randn(100) * 0.5)
        low = close - np.abs(np.random.RandomState(42).randn(100) * 0.5)
        data = {
            "open": close + np.random.RandomState(43).randn(100) * 0.2,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(100),
        }
        agent = RegimeAgent()
        report = agent.analyze(data)
        assert report.probabilities["down"] > report.probabilities["up"]

    def test_regime_agent_choppy_regime(self) -> None:
        """Choppy-Regime muss hohe 'range'-Wahrscheinlichkeit erzeugen."""
        rng = np.random.RandomState(123)
        close = np.cumsum(rng.randn(100) * 0.01) + 100
        high = close + 0.05
        low = close - 0.05
        data = {
            "open": close.copy(),
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(100),
        }
        agent = RegimeAgent()
        report = agent.analyze(data)
        assert report.probabilities["range"] > report.probabilities["up"]
        assert report.probabilities["range"] > report.probabilities["down"]

    def test_regime_agent_invalid_data_raises(self) -> None:
        agent = RegimeAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze({"close": np.array([1.0, 2.0])})  # type: ignore[arg-type]


class TestChartAgent:
    """Testet ChartAgent."""

    def test_chart_agent_produces_report(self) -> None:
        data = _make_sample_ohlcv(100)
        agent = ChartAgent()
        report = agent.analyze(data)
        assert report.agent_id == "chart"
        assert "up" in report.probabilities
        assert "down" in report.probabilities
        assert "range" in report.probabilities

    def test_chart_agent_bullish_signal(self) -> None:
        """Starker Uptrend mit BOS sollte 'up' erhoehen."""
        close = np.array([100.0 + i * 2 for i in range(100)])
        high = close + np.abs(np.random.RandomState(42).randn(100) * 0.5)
        low = close - np.abs(np.random.RandomState(42).randn(100) * 0.5)
        data = {
            "open": close - np.random.RandomState(43).randn(100) * 0.2,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(100),
        }
        agent = ChartAgent()
        report = agent.analyze(data)
        assert report.probabilities["up"] >= report.probabilities["down"]

    def test_chart_agent_invalid_data_raises(self) -> None:
        agent = ChartAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze({"close": np.array([1.0, 2.0])})  # type: ignore[arg-type]


class TestOrderFlowAgent:
    """Testet OrderFlowAgent."""

    def test_orderflow_agent_produces_report(self) -> None:
        data = _make_sample_ohlcv(100)
        agent = OrderFlowAgent()
        report = agent.analyze(data)
        assert report.agent_id == "orderflow"
        assert "up" in report.probabilities
        assert "down" in report.probabilities
        assert "range" in report.probabilities

    def test_orderflow_agent_positive_delta(self) -> None:
        """Positiver Cumulatives Delta sollte 'up' erhoehen."""
        # Erstelle Daten mit stark steigendem Trend (positives Delta)
        close = np.array([100.0 + i * 1.5 for i in range(100)])
        high = close + np.abs(np.random.RandomState(42).randn(100) * 0.3)
        low = close - np.abs(np.random.RandomState(42).randn(100) * 0.3)
        open_ = close - np.abs(np.random.RandomState(43).randn(100) * 0.1)
        volume = np.abs(np.random.RandomState(44).randn(100) * 1000) + 500
        data = {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        agent = OrderFlowAgent()
        report = agent.analyze(data)
        assert report.probabilities["up"] > report.probabilities["down"]

    def test_orderflow_agent_invalid_data_raises(self) -> None:
        agent = OrderFlowAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze({"close": np.array([1.0, 2.0])})  # type: ignore[arg-type]


class TestAgentImports:
    """Testet dass alle offentichen Symbole importierbar sind."""

    def test_agents_import_all_exports(self) -> None:
        from packages.agents import (
            AgentConfig,
            AgentType,
            BaseAgent,
            ChartAgent,
            IndicatorAgent,
            OrderFlowAgent,
            RegimeAgent,
        )

        assert BaseAgent is not None
        assert AgentConfig is not None
        assert AgentType is not None
        assert IndicatorAgent is not None
        assert RegimeAgent is not None
        assert ChartAgent is not None
        assert OrderFlowAgent is not None


class TestAgentReportDefaults:
    """Testet Default-Werte des AgentReports."""

    def test_agent_report_status_default(self) -> None:
        data = _make_sample_ohlcv(100)
        agent = IndicatorAgent()
        report = agent.analyze(data)
        assert report.status == AgentStatus.SHADOW
