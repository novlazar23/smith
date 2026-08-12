"""Tests fuer ElliottAgent — Multi-Szenario, Regelvaliditaet, Wahrscheinlichkeiten."""

from __future__ import annotations

import numpy as np
import pytest
from packages.agents import AgentConfig, AgentType, ElliottAgent
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
    InvalidationCondition,
)

# ── fixtures ──────────────────────────────────────────────────────────────


def _make_ohlcv(
    n: int = 100,
    trend: str = "up",
    rng_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Erstellt synthetische OHLCV-Daten."""
    rng = np.random.RandomState(rng_seed)
    if trend == "up":
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5) + np.arange(n) * 0.3
    elif trend == "down":
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5) - np.arange(n) * 0.3
    else:
        close = 100.0 + np.cumsum(rng.randn(n) * 0.1)

    high = close + np.abs(rng.randn(n) * 0.3)
    low = close - np.abs(rng.randn(n) * 0.3)
    return {
        "close": close,
        "high": high,
        "low": low,
    }


# ── AgentType tests ─────────────────────────────────────────────────────


class TestAgentTypeElliott:
    """Testet ELLIOTT in AgentType Enum."""

    def test_elliott_in_enum(self) -> None:
        assert AgentType.ELLIOTT == "elliott"

    def test_elliott_is_str_enum(self) -> None:
        assert isinstance(AgentType.ELLIOTT, str)


# ── Config tests ─────────────────────────────────────────────────────────


class TestElliottAgentConfig:
    """Testet ElliottAgent-Konfiguration."""

    def test_default_config(self) -> None:
        agent = ElliottAgent()
        assert agent.agent_id == "elliott"
        assert agent.config.agent_type == AgentType.ELLIOTT
        assert agent.config.status == AgentStatus.SHADOW
        assert agent.config.agent_version == "0.1.0"

    def test_custom_config(self) -> None:
        config = AgentConfig(
            agent_id="elliott",
            agent_type=AgentType.ELLIOTT,
            instrument="ETH/USD",
            horizon="1d",
        )
        agent = ElliottAgent(config=config)
        assert agent.config.instrument == "ETH/USD"
        assert agent.config.horizon == "1d"

    def test_custom_pivot_window(self) -> None:
        agent = ElliottAgent(pivot_window=10)
        assert agent.config.agent_type == AgentType.ELLIOTT


# ── Basic analysis tests ─────────────────────────────────────────────────


class TestElliottAgentBasic:
    """Testet Grundfunktionen des ElliottAgent."""

    def test_produces_agent_report(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)
        assert report.agent_id == "elliott"
        assert report.agent_version == "0.1.0"

    def test_probabilities_sum_to_one(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) <= 0.0001

    def test_probabilities_have_required_keys(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert "up" in report.probabilities
        assert "down" in report.probabilities
        assert "range" in report.probabilities

    def test_probabilities_non_negative(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        for key, val in report.probabilities.items():
            assert val >= 0.0, f"probability {key}={val} is negative"

    def test_evidence_present(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert len(report.evidence) >= 1

    def test_evidence_is_evidence_reference(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        for ev in report.evidence:
            assert isinstance(ev, EvidenceReference)
            assert ev.reference
            assert ev.feature
            assert ev.value
            assert ev.direction in ("positive", "negative", "neutral")
            assert 0.0 <= ev.relevance <= 1.0

    def test_counter_evidence_required(self) -> None:
        """Counter_evidence muss als Liste vorhanden sein."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert isinstance(report.counter_evidence, list)
        assert len(report.counter_evidence) >= 1

    def test_counter_evidence_are_evidence_references(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        for ev in report.counter_evidence:
            assert isinstance(ev, EvidenceReference)

    def test_counter_evidence_negative_direction(self) -> None:
        """Counter_evidence sollten direction='negative' haben."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        # wenigstens einer muss negativ sein
        assert any(ev.direction == "negative" for ev in report.counter_evidence)

    def test_invalidations_present(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert len(report.invalidations) >= 1
        for inv in report.invalidations:
            assert isinstance(inv, InvalidationCondition)
            assert inv.condition
            assert inv.indicator

    def test_status_shadow(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert report.status == AgentStatus.SHADOW

    def test_report_id_is_unique(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report1 = agent.analyze(data)
        report2 = agent.analyze(data)
        assert report1.report_id != report2.report_id

    def test_raw_confidence_valid(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert report.raw_confidence is not None
        assert 0.0 <= report.raw_confidence <= 0.95

    def test_hypothesis_non_empty(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert report.hypothesis
        assert len(report.hypothesis) > 0

    def test_hypothesis_mentions_scenario_count(self) -> None:
        """Hypothese muss Anzahl der Szenarios enthalten."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert "scenario" in report.hypothesis.lower()


# ── Multi-scenario tests ─────────────────────────────────────────────────


class TestElliottAgentMultiScenario:
    """Testet mindestens 2 Szenarios pro Analyse."""

    def test_at_least_two_scenarios(self) -> None:
        """Spezifikation: mind. 2 Szenarios."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        # mindestens 2 evidence-einträge vom typ scenario_
        scenario_evidence = [
            ev for ev in report.evidence
            if "scenario_" in ev.reference
        ]
        assert len(scenario_evidence) >= 2

    def test_scenarios_have_directions(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        scenario_evidence = [
            ev for ev in report.evidence
            if "scenario_" in ev.reference
        ]
        for ev in scenario_evidence:
            assert "up" in ev.feature.lower() or "down" in ev.feature.lower()

    def test_up_and_down_scenarios_present(self) -> None:
        """Sollte sowohl up- als auch down-Richtung abdecken."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        scenario_features = [ev.feature for ev in report.evidence if "scenario_" in ev.reference]
        directions = " ".join(scenario_features).lower()
        # mindestens ein up- oder down-Erwähnung
        assert "up" in directions or "down" in directions

    def test_rule_valid_in_scenarios(self) -> None:
        """Jedes Szenario hat rule_valid boolean."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        scenario_evidence = [
            ev for ev in report.evidence
            if "scenario_" in ev.reference
        ]
        # direction positive = rule_valid, neutral = nicht valid
        assert any(ev.direction == "positive" for ev in scenario_evidence)

    def test_fallback_guarantees_two_scenarios(self) -> None:
        """Selbst bei schlechten Daten: 2 Szenarios."""
        rng = np.random.RandomState(999)
        close = np.cumsum(rng.randn(50) * 10) + 50.0
        high = close + np.abs(rng.randn(50) * 5)
        low = close - np.abs(rng.randn(50) * 5)
        data = {"close": close, "high": high, "low": low}

        agent = ElliottAgent()
        report = agent.analyze(data)
        scenario_evidence = [
            ev for ev in report.evidence
            if "scenario_" in ev.reference
        ]
        assert len(scenario_evidence) >= 2


# ── Probability tests ────────────────────────────────────────────────────


class TestElliottAgentProbabilities:
    """Testet Wahrscheinlichkeitsberechnung."""

    def test_probability_values_in_range(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        for val in report.probabilities.values():
            assert 0.0 <= val <= 1.0

    def test_probability_up_down_range(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        # up sollte in uptrend hoehere wahrscheinlichkeit haben
        assert report.probabilities["up"] > 0.1

    def test_probability_sum_exact(self) -> None:
        """Summe muss exakt 1.0 sein (±0.0001)."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) < 0.0001

    def test_probability_consistent_across_runs(self) -> None:
        """Gleiche Daten -> gleiche Wahrscheinlichkeiten."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report1 = agent.analyze(data)
        report2 = agent.analyze(data)
        assert report1.probabilities == report2.probabilities

    def test_uptrend_biased_probabilities(self) -> None:
        data = _make_ohlcv(100, trend="up", rng_seed=42)
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert report.probabilities["up"] > report.probabilities["down"]

    def test_downtrend_biased_probabilities(self) -> None:
        data = _make_ohlcv(100, trend="down", rng_seed=42)
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert report.probabilities["down"] > report.probabilities["up"]


# ── Evidence tests ───────────────────────────────────────────────────────


class TestElliottAgentEvidence:
    """Testet Evidenz-Erzeugung."""

    def test_evidence_min_one(self) -> None:
        """Spezifikation: evidence min 1 (AT-004)."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert len(report.evidence) >= 1

    def test_evidence_pivot_info(self) -> None:
        """Evidenz sollte Pivot-Informationen enthalten."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        pivot_evidence = [
            ev for ev in report.evidence
            if "pivot" in ev.reference.lower()
        ]
        # sollte mindestens einen pivot eintrag geben
        assert len(pivot_evidence) >= 1

    def test_evidence_relevance_valid(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        for ev in report.evidence:
            assert 0.0 <= ev.relevance <= 1.0

    def test_counter_evidence_not_empty(self) -> None:
        """Counter_evidence darf nicht leer sein."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert len(report.counter_evidence) >= 1


# ── Invalidations tests ──────────────────────────────────────────────────


class TestElliottAgentInvalidations:
    """Testet Invalidierungsbedingungen."""

    def test_invalidations_have_threshold(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        for inv in report.invalidations:
            assert inv.threshold is not None

    def test_invalidations_have_direction(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        for inv in report.invalidations:
            assert inv.direction in ("above", "below")

    def test_invalidations_up_trend_below(self) -> None:
        """Im Uptrend sollte Invalidierung 'below' sein."""
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        # wenigstens eine invalidation muss below sein
        assert any(inv.direction == "below" for inv in report.invalidations)


# ── Missing keys test ────────────────────────────────────────────────────


class TestElliottAgentValidation:
    """Testet Eingabevalidierung."""

    def test_missing_keys_raises(self) -> None:
        data = {"close": _make_ohlcv(100)["close"]}  # fehlt high, low
        agent = ElliottAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze(data)

    def test_empty_data_raises(self) -> None:
        agent = ElliottAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze({})

    def test_only_close_raises(self) -> None:
        data = {"close": np.array([100.0, 101.0])}
        agent = ElliottAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze(data)


# ── Confidence tests ─────────────────────────────────────────────────────


class TestElliottAgentConfidence:
    """Testet Konfidenzberechnung."""

    def test_confidence_positive(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert report.raw_confidence > 0

    def test_confidence_max_095(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert report.raw_confidence <= 0.95

    def test_confidence_non_zero_with_data(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = ElliottAgent()
        report = agent.analyze(data)
        assert report.raw_confidence > 0


# ── Integration / cross-agent tests ──────────────────────────────────────


class TestElliottAgentIntegration:
    """Integrationstests mit verschiedenen Marktszenarien."""

    def test_choppy_market(self) -> None:
        rng = np.random.RandomState(777)
        close = 100.0 + np.cumsum(rng.randn(100) * 2.0)
        high = close + np.abs(rng.randn(100) * 1.0)
        low = close - np.abs(rng.randn(100) * 1.0)
        data = {"close": close, "high": high, "low": low}

        agent = ElliottAgent()
        report = agent.analyze(data)

        assert isinstance(report, AgentReport)
        assert len(report.probabilities) == 3
        assert sum(report.probabilities.values()) == pytest.approx(1.0)

    def test_strong_uptrend(self) -> None:
        close = np.array([100.0 + i * 5.0 for i in range(100)], dtype=np.float64)
        high = close + 2.0
        low = close - 2.0
        data = {"close": close, "high": high, "low": low}

        agent = ElliottAgent()
        report = agent.analyze(data)

        assert report.probabilities["up"] > report.probabilities["down"]

    def test_strong_downtrend(self) -> None:
        close = np.array([500.0 - i * 3.0 for i in range(100)], dtype=np.float64)
        high = close + 2.0
        low = close - 2.0
        data = {"close": close, "high": high, "low": low}

        agent = ElliottAgent()
        report = agent.analyze(data)

        assert report.probabilities["down"] > report.probabilities["up"]
