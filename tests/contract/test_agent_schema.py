"""Contract tests for agent schema compliance.

Verifies:
- Agent outputs conform to AgentReport schema (direction/probabilities,
  confidence, evidence, invalidations).
- AgentReport has required fields: report_id, run_id, agent_id, instrument,
  horizon, as_of, hypothesis, probabilities, evidence.
- AgentConfig has required fields: agent_id, agent_type, etc.
- Agent registry (agents/__init__.py) exports all configured agents.
"""

from __future__ import annotations

import uuid as uuid_mod
from datetime import UTC, datetime
from typing import ClassVar

import numpy as np
import pytest
from packages.agents import (
    AgentConfig,
    AgentType,
    AnomalyAgent,
    BaseAgent,
    ChartAgent,
    ContrarianAgent,
    CrossMarketAgent,
    ElliottAgent,
    FibonacciAgent,
    HistoricalAnalogyAgent,
    IndicatorAgent,
    NewsAgent,
    OrderFlowAgent,
    PatternAgent,
    RegimeAgent,
)
from packages.agents.contrainer.models import ContrarianConfig, ContrarianHypothesis
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
    InvalidationCondition,
)

# ── AgentConfig contract ────────────────────────────────────────────────


class TestAgentConfig:
    """AgentConfig must have all expected fields with proper defaults."""

    def test_config_requires_agent_id(self) -> None:
        config = AgentConfig(agent_id="test", agent_type=AgentType.INDICATOR)
        assert config.agent_id == "test"

    def test_config_defaults(self) -> None:
        config = AgentConfig(agent_id="test", agent_type=AgentType.INDICATOR)
        assert config.agent_version == "0.1.0"
        assert config.agent_type == AgentType.INDICATOR
        assert config.instrument == ""
        assert config.horizon == "1h"
        assert config.status == AgentStatus.SHADOW

    def test_config_immutable(self) -> None:
        config = AgentConfig(agent_id="test", agent_type=AgentType.INDICATOR)
        # frozen=True means __setattr__ raises
        with pytest.raises(Exception):
            config.agent_version = "1.0.0"


# ── AgentType enum contract ─────────────────────────────────────────────


class TestAgentType:
    """AgentType enum must cover all defined agent categories."""

    def test_expected_types_present(self) -> None:
        expected = {
            "indicator", "regime", "chart", "orderflow", "pattern",
            "fibonacci", "elliott", "historical_analogy", "news",
            "cross_market", "anomaly", "contrarian",
        }
        actual = {at.value for at in AgentType}
        assert expected == actual


# ── AgentReport schema contract ─────────────────────────────────────────


def _make_sample_ohlcv(n: int = 40) -> dict[str, np.ndarray]:
    """Synthetic OHLCV data for agent analysis (≥35 bars for all agents)."""
    rng = np.random.RandomState(42)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    return {
        "open": close - rng.randn(n) * 0.2,
        "high": close + np.abs(rng.randn(n) * 0.3),
        "low": close - np.abs(rng.randn(n) * 0.3),
        "close": close,
        "volume": np.abs(rng.randn(n) * 1000) + 500,
    }


class TestAgentReportSchema:
    """AgentReport must conform to the defined Pydantic model."""

    def _sample_report(self) -> AgentReport:
        return AgentReport(
            report_id=uuid_mod.uuid4().hex,
            run_id=uuid_mod.uuid4().hex,
            agent_id="test-agent",
            agent_version="0.1.0",
            instrument="BTC/USDT",
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis="Bullish cross on MACD",
            probabilities={"up": 0.55, "down": 0.25, "range": 0.20},
            evidence=[
                EvidenceReference(
                    reference="test:macd",
                    feature="macd",
                    value="cross",
                    direction="positive",
                    relevance=0.7,
                ),
            ],
            counter_evidence=[
                EvidenceReference(
                    reference="test:volume",
                    feature="volume",
                    value="low",
                    direction="negative",
                    relevance=0.3,
                ),
            ],
            invalidations=[
                InvalidationCondition(
                    condition="MACD cross reversal",
                    indicator="macd",
                    threshold=0.0,
                    direction="below",
                ),
            ],
            raw_confidence=0.65,
            status=AgentStatus.SHADOW,
        )

    def test_report_has_all_required_fields(self) -> None:
        report = self._sample_report()
        required = {
            "report_id", "run_id", "agent_id", "agent_version",
            "instrument", "horizon", "as_of", "hypothesis",
            "probabilities", "evidence",
        }
        for field in required:
            assert hasattr(report, field), f"AgentReport missing: {field}"

    def test_probabilities_sum_to_one(self) -> None:
        report = self._sample_report()
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) <= 0.0001, (
            f"Probabilities sum to {total}, expected 1.0 ± 0.0001"
        )

    def test_probabilities_empty_raises(self) -> None:
        with pytest.raises(Exception):  # pydantic validation error
            AgentReport(
                report_id="r1",
                run_id="r1",
                agent_id="a1",
                agent_version="0.1.0",
                instrument="",
                horizon="1h",
                as_of=datetime.now(UTC),
                hypothesis="Test",
                probabilities={},  # empty should fail
                evidence=[],  # min_length=1 will also fail
            )

    def test_evidence_min_length_one(self) -> None:
        """Evidence must have at least one reference."""
        with pytest.raises(Exception):
            AgentReport(
                report_id="r1",
                run_id="r1",
                agent_id="a1",
                agent_version="0.1.0",
                instrument="",
                horizon="1h",
                as_of=datetime.now(UTC),
                hypothesis="Test",
                probabilities={"up": 0.33, "down": 0.33, "range": 0.34},
                evidence=[],
            )

    def test_evidence_reference_required_fields(self) -> None:
        ref = EvidenceReference(
            reference="test:ref",
            feature="macd",
            value="cross",
            direction="positive",
            relevance=0.7,
        )
        for field in ("reference", "feature", "value", "direction", "relevance"):
            assert hasattr(ref, field)

    def test_invalidations_required_fields(self) -> None:
        inv = InvalidationCondition(
            condition="MACD reversal",
            indicator="macd",
            threshold=0.0,
            direction="below",
        )
        for field in ("condition", "indicator", "threshold", "direction"):
            assert hasattr(inv, field)

    def test_raw_confidence_clamped(self) -> None:
        """raw_confidence must be in [0.0, 1.0]."""
        report = self._sample_report()
        assert 0.0 <= report.raw_confidence <= 1.0

    def test_report_frozen(self) -> None:
        report = self._sample_report()
        with pytest.raises(Exception):
            report.hypothesis = "modified"

    def test_optional_fields_defaults(self) -> None:
        report = self._sample_report()
        assert report.expected_return is None
        assert report.counter_evidence is not None  # default_factory returns []
        assert report.invalidations is not None
        assert report.sample_size is None
        assert report.calibrated_confidence is None
        assert report.data_quality == 1.0
        assert report.uncertainty is None


# ── Agent implementation contract ───────────────────────────────────────


class TestBaseAgentContract:
    """BaseAgent must define analyze(…) -> AgentReport."""

    def test_base_agent_has_analyze(self) -> None:
        assert hasattr(BaseAgent, "analyze")

    def test_base_agent_has_agent_id_property(self) -> None:
        class Dummy(BaseAgent):
            def analyze(self, data: dict) -> AgentReport:
                raise NotImplementedError

        dummy = Dummy(AgentConfig(agent_id="d", agent_type=AgentType.INDICATOR))
        assert dummy.agent_id == "d"

    def test_base_agent_has_config_property(self) -> None:
        class Dummy(BaseAgent):
            def analyze(self, data: dict) -> AgentReport:
                raise NotImplementedError

        dummy = Dummy(AgentConfig(agent_id="d", agent_type=AgentType.INDICATOR))
        assert dummy.config.agent_id == "d"

    def test_base_agent_implements_return_type(self) -> None:
        """Subclass analyze must return AgentReport (checked via actual call)."""
        class Dummy(BaseAgent):
            def analyze(self, data: dict) -> AgentReport:
                return AgentReport(
                    report_id=self._generate_report_id(),
                    run_id=uuid_mod.uuid4().hex,
                    agent_id="dummy",
                    agent_version="0.1.0",
                    instrument="",
                    horizon="1h",
                    as_of=datetime.now(UTC),
                    hypothesis="Test",
                    probabilities={"up": 0.33, "down": 0.33, "range": 0.34},
                    evidence=[
                        EvidenceReference(
                            reference="dummy:test",
                            feature="test",
                            value="ok",
                            direction="neutral",
                            relevance=0.0,
                        ),
                    ],
                    status=AgentStatus.SHADOW,
                )

        dummy = Dummy(AgentConfig(agent_id="d", agent_type=AgentType.INDICATOR))
        result = dummy.analyze({})
        assert isinstance(result, AgentReport)

    def test_evidence_reference_construction(self) -> None:
        class Dummy(BaseAgent):
            def analyze(self, data: dict) -> AgentReport:
                raise NotImplementedError

        dummy = Dummy(AgentConfig(agent_id="d", agent_type=AgentType.INDICATOR))
        ref = dummy._make_evidence("macd", "cross", "positive", 0.7)
        assert isinstance(ref, EvidenceReference)
        assert ref.feature == "macd"
        assert ref.direction == "positive"
        assert ref.relevance == 0.7

    def test_invalidations_construction(self) -> None:
        class Dummy(BaseAgent):
            def analyze(self, data: dict) -> AgentReport:
                raise NotImplementedError

        dummy = Dummy(AgentConfig(agent_id="d", agent_type=AgentType.INDICATOR))
        inv = dummy._make_invalidations("condition", "ind", 0.0, "below")
        assert isinstance(inv, InvalidationCondition)
        assert inv.condition == "condition"


# ── Concrete agent contract — each must produce a valid AgentReport ─────

# Agents whose analyze() returns a tuple instead of AgentReport.
# These must be unpacked: hypothesis, report = agent.analyze(data).
_TUPLE_RETURN_AGENTS = {"contrarian"}

# Agents needing special data sizes (handled via _make_sample_ohlcv(n)).
# IndicatorAgent needs ≥35 bars, Fibonacci needs ≥21, HistoricalAnalogy ≥21.


class TestConcreteAgentsReportContract:
    """All concrete agents must produce valid AgentReport from analyze()."""

    _AGENTS: ClassVar[tuple[tuple[str, type[BaseAgent]], ...]] = (
        ("indicator", IndicatorAgent),
        ("chart", ChartAgent),
        ("regime", RegimeAgent),
        ("orderflow", OrderFlowAgent),
        ("pattern", PatternAgent),
        ("fibonacci", FibonacciAgent),
        ("elliott", ElliottAgent),
        ("news", NewsAgent),
        ("cross_market", CrossMarketAgent),
        ("anomaly", AnomalyAgent),
        ("historical_analogy", HistoricalAnalogyAgent),
        ("contrarian", ContrarianAgent),
    )

    def _make_ohlcv_for_agent(self, agent_name: str) -> dict:
        """Return OHLCV data sized appropriately for each agent."""
        n = 55  # default: IndicatorAgent needs ≥50 (SMA-50)
        if agent_name in ("fibonacci", "historical_analogy"):
            n = 60  # need ≥21 bars + extra for historical
        return _make_sample_ohlcv(n)

    def _build_news_data(self) -> dict:
        return {"news": [
            {"title": "Test", "body": "Content",
             "source_name": "Test", "source_type": "blog"},
        ]}

    def _build_cross_market_data(self) -> dict:
        return {"btc_dominance": 0.55}

    def _build_contrarian_data(self) -> list[AgentReport]:
        """Build sample AgentReport list for ContrarianAgent.analyze()."""
        base = {
            "report_id": "test",
            "run_id": "run",
            "agent_id": "test_agent",
            "agent_version": "1.0",
            "instrument": "BTC/USDT",
            "horizon": "1h",
            "as_of": "2025-01-01T00:00:00Z",
            "hypothesis": "Test",
            "probabilities": {"up": 0.6, "down": 0.2, "range": 0.2},
            "evidence": [{"reference": "ref1", "feature": "test", "value": "high", "direction": "up", "relevance": 0.8}],
            "invalidations": [{"condition": "none", "indicator": "test", "threshold": 0.0, "direction": "above"}],
            "status": "active",
            "raw_confidence": 0.7,
            "calibrated_confidence": 0.7,
        }
        reports = [
            AgentReport(**{**base, "agent_id": "bull1", "probabilities": {"up": 0.7, "down": 0.15, "range": 0.15}}),
            AgentReport(**{**base, "agent_id": "bull2", "probabilities": {"up": 0.65, "down": 0.2, "range": 0.15}}),
            AgentReport(**{**base, "agent_id": "bear1", "probabilities": {"up": 0.2, "down": 0.7, "range": 0.1}}),
        ]
        return reports

    def _extract_report(
        self,
        agent_name: str,
        result: AgentReport | tuple[ContrarianHypothesis, AgentReport],
    ) -> AgentReport:
        """Extract AgentReport from analyze result (handles tuple returns)."""
        if agent_name in _TUPLE_RETURN_AGENTS:
            assert isinstance(result, tuple), (
                f"{agent_name}.analyze() returned {type(result).__name__}, "
                f"expected tuple[ContrarianHypothesis, AgentReport]"
            )
            assert len(result) == 2, (
                f"{agent_name}.analyze() returned tuple of length {len(result)}"
            )
            report = result[1]
        else:
            report = result
        assert isinstance(report, AgentReport), (
            f"{agent_name}.analyze() returned {type(report).__name__}, "
            f"expected AgentReport"
        )
        return report

    @pytest.mark.parametrize(("agent_name", "agent_cls"), _AGENTS)
    def test_agent_analyze_returns_agent_report(self, agent_name: str,
                                                  agent_cls: type) -> None:
        """Every concrete agent must return AgentReport from analyze()."""
        if agent_name == "contrarian":
            config = ContrarianConfig(agent_id=agent_name)
        else:
            config = AgentConfig(agent_id=agent_name, agent_type=AgentType.INDICATOR)
        agent = agent_cls(config)

        if agent_name == "news":
            data = self._build_news_data()
        elif agent_name == "cross_market":
            data = self._build_cross_market_data()
        elif agent_name == "contrarian":
            data = self._build_contrarian_data()
        else:
            data = self._make_ohlcv_for_agent(agent_name)

        result = agent.analyze(data)
        report = self._extract_report(agent_name, result)
        assert isinstance(report, AgentReport), (
            f"{agent_name}.analyze() returned {type(report).__name__}, "
            f"expected AgentReport"
        )

    @pytest.mark.parametrize(("agent_name", "agent_cls"), _AGENTS)
    def test_agent_report_has_direction_data(self, agent_name: str,
                                               agent_cls: type) -> None:
        """AgentReport probabilities encode direction (up/down/range)."""
        if agent_name == "contrarian":
            config = ContrarianConfig(agent_id=agent_name)
        else:
            config = AgentConfig(agent_id=agent_name, agent_type=AgentType.INDICATOR)
        agent = agent_cls(config)

        if agent_name == "news":
            data = self._build_news_data()
        elif agent_name == "cross_market":
            data = self._build_cross_market_data()
        elif agent_name == "contrarian":
            data = self._build_contrarian_data()
        else:
            data = self._make_ohlcv_for_agent(agent_name)

        result = agent.analyze(data)
        report = self._extract_report(agent_name, result)
        for key in ("up", "down", "range"):
            assert key in report.probabilities, (
                f"{agent_name}: probabilities missing '{key}'"
            )

    @pytest.mark.parametrize(("agent_name", "agent_cls"), _AGENTS)
    def test_agent_report_has_evidence(self, agent_name: str,
                                         agent_cls: type) -> None:
        """AgentReport must include at least one evidence reference."""
        if agent_name == "contrarian":
            config = ContrarianConfig(agent_id=agent_name)
        else:
            config = AgentConfig(agent_id=agent_name, agent_type=AgentType.INDICATOR)
        agent = agent_cls(config)

        if agent_name == "news":
            data = self._build_news_data()
        elif agent_name == "cross_market":
            data = self._build_cross_market_data()
        elif agent_name == "contrarian":
            data = self._build_contrarian_data()
        else:
            data = self._make_ohlcv_for_agent(agent_name)

        result = agent.analyze(data)
        report = self._extract_report(agent_name, result)
        assert len(report.evidence) >= 1, (
            f"{agent_name}: evidence must have at least 1 entry"
        )

    @pytest.mark.parametrize(("agent_name", "agent_cls"), _AGENTS)
    def test_agent_report_has_hypothesis(self, agent_name: str,
                                          agent_cls: type) -> None:
        """AgentReport must include a hypothesis string."""
        if agent_name == "contrarian":
            config = ContrarianConfig(agent_id=agent_name)
        else:
            config = AgentConfig(agent_id=agent_name, agent_type=AgentType.INDICATOR)
        agent = agent_cls(config)

        if agent_name == "news":
            data = self._build_news_data()
        elif agent_name == "cross_market":
            data = self._build_cross_market_data()
        elif agent_name == "contrarian":
            data = self._build_contrarian_data()
        else:
            data = self._make_ohlcv_for_agent(agent_name)

        result = agent.analyze(data)
        report = self._extract_report(agent_name, result)
        assert isinstance(report.hypothesis, str)
        assert len(report.hypothesis) > 0


# ── Agent registry contract ─────────────────────────────────────────────


class TestAgentRegistry:
    """The agents package must export all configured agents."""

    def test_all_agents_exported(self) -> None:
        """Agents __all__ must contain every agent class."""
        from packages import agents as agent_pkg

        expected = {
            "AgentConfig", "AgentType", "BaseAgent",
            "AnomalyAgent", "ChartAgent", "ContrarianAgent",
            "CrossMarketAgent", "ElliottAgent", "FibonacciAgent",
            "HistoricalAnalogyAgent", "IndicatorAgent", "NewsAgent",
            "OrderFlowAgent", "PatternAgent", "RegimeAgent",
        }
        actual = set(agent_pkg.__all__)
        assert expected == actual, f"Missing: {expected - actual}; Extra: {actual - expected}"

    def test_all_agents_instantiable(self) -> None:
        """Every exported agent class must be instantiable."""
        from packages import agents as agent_pkg

        agent_classes = {
            "AnomalyAgent", "ChartAgent", "ContrarianAgent",
            "CrossMarketAgent", "ElliottAgent", "FibonacciAgent",
            "HistoricalAnalogyAgent", "IndicatorAgent", "NewsAgent",
            "OrderFlowAgent", "PatternAgent", "RegimeAgent",
        }

        for name in agent_classes:
            cls = getattr(agent_pkg, name, None)
            assert cls is not None, f"Missing class: {name}"

    def test_base_agent_is_abc(self) -> None:
        """BaseAgent must be abstract (not instantiable directly)."""
        with pytest.raises(TypeError):
            BaseAgent(agent_id="nope", agent_type=AgentType.INDICATOR)  # type: ignore
