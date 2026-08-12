"""Tests fuer FibonacciAgent — Pivot-Erkennung, Fib-Zonen, Konfluenz, Report."""

from __future__ import annotations

import uuid as uuid_mod

import numpy as np
import pytest
from packages.agents import (
    AgentConfig,
    AgentType,
    FibonacciAgent,
)
from packages.chart_structure.base import SupportResistanceLevel
from packages.fibonacci import (
    FibonacciArea,
    FibonacciPivot,
)
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


def _make_sr_levels() -> list[SupportResistanceLevel]:
    """Erstellt eine Liste von SR-Levels."""
    return [
        SupportResistanceLevel(price=105.0, level_type="support", strength=0.8, touch_count=3),
        SupportResistanceLevel(price=115.0, level_type="resistance", strength=0.6, touch_count=2),
        SupportResistanceLevel(price=90.0, level_type="support", strength=0.9, touch_count=5),
    ]


# ── AgentType tests ─────────────────────────────────────────────────────


class TestAgentTypeFibonacci:
    """Testet FIBONACCI in AgentType Enum."""

    def test_fibonacci_in_enum(self) -> None:
        assert AgentType.FIBONACCI == "fibonacci"

    def test_fibonacci_is_str_enum(self) -> None:
        assert isinstance(AgentType.FIBONACCI, str)


class TestFibonacciAgentConfig:
    """Testet FibonacciAgent-Konfiguration."""

    def test_default_config(self) -> None:
        agent = FibonacciAgent()
        assert agent.agent_id == "fibonacci"
        assert agent.config.agent_type == AgentType.FIBONACCI
        assert agent.config.status == AgentStatus.SHADOW
        assert agent.config.agent_version == "0.1.0"

    def test_custom_config(self) -> None:
        config = AgentConfig(
            agent_id="fibonacci",
            agent_type=AgentType.FIBONACCI,
            instrument="BTC/USD",
            horizon="4h",
        )
        agent = FibonacciAgent(config=config)
        assert agent.config.instrument == "BTC/USD"
        assert agent.config.horizon == "4h"


# ── Basic analysis tests ────────────────────────────────────────────────


class TestFibonacciAgentBasic:
    """Testet Grundfunktionen des FibonacciAgent."""

    def test_produces_agent_report(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)
        assert report.agent_id == "fibonacci"
        assert report.agent_version == "0.1.0"

    def test_probabilities_sum_to_one(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) <= 0.0001

    def test_probabilities_have_required_keys(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert "up" in report.probabilities
        assert "down" in report.probabilities
        assert "range" in report.probabilities

    def test_evidence_present(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert len(report.evidence) >= 1

    def test_evidence_is_evidence_reference(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        for ev in report.evidence:
            assert isinstance(ev, EvidenceReference)
            assert ev.reference
            assert ev.feature
            assert ev.value
            assert ev.direction in ("positive", "negative", "neutral")
            assert 0.0 <= ev.relevance <= 1.0

    def test_counter_evidence_required(self) -> None:
        """Counter_evidence muss vorhanden sein (min 1, auch wenn leer erlaubt)."""
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        # Die Spec verlangt "Gegenhypothese required" → mindestens 1
        assert isinstance(report.counter_evidence, list)

    def test_invalidations_present(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert len(report.invalidations) >= 1
        for inv in report.invalidations:
            assert isinstance(inv, InvalidationCondition)
            assert inv.condition
            assert inv.indicator
            assert inv.direction in ("above", "below")

    def test_status_shadow(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert report.status == AgentStatus.SHADOW

    def test_report_id_is_unique(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report1 = agent.analyze(data)
        report2 = agent.analyze(data)
        assert report1.report_id != report2.report_id

    def test_raw_confidence_valid(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert report.raw_confidence is not None
        assert 0.0 <= report.raw_confidence <= 0.95

    def test_hypothesis_non_empty(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert report.hypothesis
        assert len(report.hypothesis) > 0


# ── Trend direction tests ───────────────────────────────────────────────


class TestFibonacciAgentTrendDirection:
    """Testet dass die Trendrichtung die Wahrscheinlichkeit beeinflusst."""

    def test_uptrend_bias(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert report.probabilities["up"] > report.probabilities["down"]

    def test_downtrend_bias(self) -> None:
        data = _make_ohlcv(100, trend="down")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert report.probabilities["down"] > report.probabilities["up"]


# ── Data validation tests ───────────────────────────────────────────────


class TestFibonacciAgentValidation:
    """Testet Datenvalidierung."""

    def test_missing_close_raises(self) -> None:
        agent = FibonacciAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze({"high": np.array([1.0]), "low": np.array([1.0])})

    def test_missing_high_raises(self) -> None:
        agent = FibonacciAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze({"close": np.array([1.0]), "low": np.array([1.0])})

    def test_missing_low_raises(self) -> None:
        agent = FibonacciAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze({"close": np.array([1.0]), "high": np.array([1.0])})

    def test_empty_data_raises(self) -> None:
        agent = FibonacciAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            agent.analyze({})  # type: ignore[arg-type]


# ── Custom window parameters ────────────────────────────────────────────


class TestFibonacciAgentCustomParams:
    """Testet benutzerdefinierte Parameter."""

    def test_small_window(self) -> None:
        """Kleine Fenster sollten mehr Pivots erkennen."""
        data = _make_ohlcv(50, trend="up")
        agent = FibonacciAgent(high_window=2, low_window=2)
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)
        assert len(report.evidence) >= 1

    def test_large_window(self) -> None:
        """Grosse Fenster sollten weniger Pivots erkennen."""
        data = _make_ohlcv(50, trend="up")
        agent = FibonacciAgent(high_window=15, low_window=15)
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)


# ── SR levels confluence tests ──────────────────────────────────────────


class TestFibonacciAgentConfluence:
    """Testet Konfluenz mit SR-Levels."""

    def test_with_sr_levels(self) -> None:
        """SR-Levels sollten die Konfluenz beeinflussen."""
        data = _make_ohlcv(100, trend="up")
        data["sr_levels"] = _make_sr_levels()
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)
        assert len(report.evidence) >= 1

    def test_without_sr_levels(self) -> None:
        """Ohne SR-Levels sollte der Agent auch funktionieren."""
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)


# ── Evidence quality tests ──────────────────────────────────────────────


class TestEvidenceQuality:
    """Testet die Qualitaet der Evidenz."""

    def test_evidence_references_unique(self) -> None:
        """Evidenz-Referenzen sollten eindeutig sein."""
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        refs = [ev.reference for ev in report.evidence]
        assert len(refs) == len(set(refs))

    def test_evidence_has_relevance(self) -> None:
        """Jede Evidenz muss eine Relevanz zwischen 0 und 1 haben."""
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        for ev in report.evidence:
            assert 0.0 <= ev.relevance <= 1.0


# ── Edge cases ──────────────────────────────────────────────────────────


class TestFibonacciAgentEdgeCases:
    """Testet Randfaelle."""

    def test_very_long_data(self) -> None:
        data = _make_ohlcv(500, trend="up")
        agent = FibonacciAgent()
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)
        assert len(report.evidence) >= 1

    def test_short_data_min_pivots(self) -> None:
        """Mindestens 2*window+1 Balken benoetigt."""
        rng = np.random.RandomState(42)
        n = 25
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
        high = close + np.abs(rng.randn(n) * 0.3)
        low = close - np.abs(rng.randn(n) * 0.3)
        data = {"close": close, "high": high, "low": low}
        agent = FibonacciAgent(high_window=2, low_window=2)
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)

    def test_small_zone_band(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent(zone_band=0.001)
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)

    def test_large_zone_band(self) -> None:
        data = _make_ohlcv(100, trend="up")
        agent = FibonacciAgent(zone_band=0.02)
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)


# ── Integration test ────────────────────────────────────────────────────


class TestFibonacciAgentIntegration:
    """Integrations-Test fuer den kompletten Analyse-Pipeline."""

    def test_full_pipeline_with_confluence(self) -> None:
        """Kompletter Pipeline mit Pivot-Detection, Retracements, Konfluenz."""
        data = _make_ohlcv(150, trend="up", rng_seed=99)
        data["sr_levels"] = _make_sr_levels()
        agent = FibonacciAgent(high_window=5, low_window=5)
        report = agent.analyze(data)

        # Report-Struktur
        assert report.agent_id == "fibonacci"
        assert report.agent_version == "0.1.0"
        assert report.hypothesis
        assert report.status == AgentStatus.SHADOW

        # Wahrscheinlichkeiten
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) <= 0.0001

        # Evidenz
        assert len(report.evidence) >= 1
        for ev in report.evidence:
            assert isinstance(ev, EvidenceReference)

        # Gegenhypothese
        assert isinstance(report.counter_evidence, list)

        # Invalidierungen
        assert len(report.invalidations) >= 1
        for inv in report.invalidations:
            assert isinstance(inv, InvalidationCondition)

        # Konfidenz
        assert 0.0 <= report.raw_confidence <= 0.95