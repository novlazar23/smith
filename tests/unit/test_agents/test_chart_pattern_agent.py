"""Tests fuer ChartPatternAgent — 4 Evidenz-Familien, Kombination, Report, Ensemble."""

from __future__ import annotations

import math

import numpy as np
import pytest
from packages.agents import (
    AgentConfig,
    AgentType,
    ChartPatternAgent,
)
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
    InvalidationCondition,
)

# ── fixtures ──────────────────────────────────────────────────────────────


def _make_ohlcv(
    n: int = 200,
    trend: str = "up",
    rng_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Erstellt synthetische OHLCV-Daten (open = vorheriges close)."""
    rng = np.random.RandomState(rng_seed)
    if trend == "up":
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5) + np.arange(n) * 0.3
    elif trend == "down":
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5) - np.arange(n) * 0.3
    else:
        close = 100.0 + np.cumsum(rng.randn(n) * 0.1)
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + np.abs(rng.randn(n) * 0.3)
    low = np.minimum(open_, close) - np.abs(rng.randn(n) * 0.3)
    volume = np.abs(rng.randn(n)) * 100.0 + 100.0
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _make_flat_ohlcv(n: int = 100, price: float = 100.0) -> dict[str, np.ndarray]:
    """Erstellt flache OHLCV-Daten (alle Preise konstant, Zero-Range-Kerzen)."""
    const = np.full(n, price)
    return {
        "open": const.copy(),
        "high": const.copy(),
        "low": const.copy(),
        "close": const.copy(),
        "volume": const.copy(),
    }


# ── AgentType tests ─────────────────────────────────────────────────────


class TestAgentTypeChartPattern:
    """Testet PATTERN in AgentType Enum."""

    def test_pattern_in_enum(self) -> None:
        assert AgentType.PATTERN == "pattern"

    def test_pattern_is_str_enum(self) -> None:
        assert isinstance(AgentType.PATTERN, str)


# ── Config tests ─────────────────────────────────────────────────────────


class TestChartPatternAgentConfig:
    """Testet ChartPatternAgent-Konfiguration."""

    def test_default_config(self) -> None:
        agent = ChartPatternAgent()
        assert agent.agent_id == "chart_pattern"
        assert agent.config.agent_type == AgentType.PATTERN
        assert agent.config.status == AgentStatus.SHADOW
        assert agent.config.agent_version == "0.1.0"

    def test_custom_config(self) -> None:
        config = AgentConfig(
            agent_id="chart_pattern",
            agent_type=AgentType.PATTERN,
            instrument="BTC/USD",
            horizon="15m",
        )
        agent = ChartPatternAgent(config=config)
        assert agent.config.instrument == "BTC/USD"
        assert agent.config.horizon == "15m"

    def test_params_kwarg_accepted(self) -> None:
        """params=None wird der Ensemble-Signatur wegen akzeptiert."""
        agent = ChartPatternAgent(params=None)
        assert agent.agent_id == "chart_pattern"


# ── Basic analysis tests ─────────────────────────────────────────────────


class TestChartPatternAgentBasic:
    """Testet Grundfunktionen des ChartPatternAgent."""

    def test_produces_agent_report(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)
        assert report.agent_id == "chart_pattern"
        assert report.agent_version == "0.1.0"

    def test_probabilities_sum_to_one(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) <= 0.001

    def test_probabilities_finite_and_non_negative(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        for key, val in report.probabilities.items():
            assert math.isfinite(val), f"probability {key}={val} ist nicht endlich"
            assert val >= 0.0, f"probability {key}={val} ist negativ"

    def test_probabilities_have_required_keys(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert "up" in report.probabilities
        assert "down" in report.probabilities
        assert "range" in report.probabilities

    def test_evidence_present(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert len(report.evidence) >= 1

    def test_evidence_is_evidence_reference(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        for ev in report.evidence:
            assert isinstance(ev, EvidenceReference)
            assert ev.reference
            assert ev.feature
            assert ev.value
            assert ev.direction in ("positive", "negative", "neutral")
            assert 0.0 <= ev.relevance <= 1.0

    def test_counter_evidence_present(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert isinstance(report.counter_evidence, list)
        assert len(report.counter_evidence) >= 1
        assert any(ev.direction == "negative" for ev in report.counter_evidence)

    def test_invalidations_present(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert len(report.invalidations) >= 1
        for inv in report.invalidations:
            assert isinstance(inv, InvalidationCondition)
            assert inv.condition
            assert inv.indicator
            assert inv.direction in ("above", "below")

    def test_status_shadow(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert report.status == AgentStatus.SHADOW

    def test_hypothesis_non_empty(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert report.hypothesis
        assert len(report.hypothesis) > 0

    def test_raw_confidence_valid(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert report.raw_confidence is not None
        assert 0.0 <= report.raw_confidence <= 1.0

    def test_report_id_is_unique(self) -> None:
        data = _make_ohlcv(200, trend="up")
        agent = ChartPatternAgent()
        report1 = agent.analyze(data)
        report2 = agent.analyze(data)
        assert report1.report_id != report2.report_id


# ── Determinism tests ────────────────────────────────────────────────────


class TestChartPatternAgentDeterminism:
    """Testet Determinismus: gleiche Daten → gleiche Wahrscheinlichkeiten."""

    @pytest.mark.parametrize("trend", ["up", "down", "range"])
    def test_deterministic_across_runs(self, trend: str) -> None:
        data = _make_ohlcv(200, trend=trend)
        agent = ChartPatternAgent()
        report1 = agent.analyze(data)
        report2 = agent.analyze(data)
        assert report1.probabilities == report2.probabilities
        assert report1.raw_confidence == report2.raw_confidence
        assert report1.hypothesis == report2.hypothesis

    def test_deterministic_on_flat_data(self) -> None:
        data = _make_flat_ohlcv(100)
        agent = ChartPatternAgent()
        report1 = agent.analyze(data)
        report2 = agent.analyze(data)
        assert report1.probabilities == report2.probabilities


# ── Edge case tests ──────────────────────────────────────────────────────


class TestChartPatternAgentEdgeCases:
    """Testet Randfaelle: kurze Daten, flache Preise, fehlende Keys."""

    def test_short_data_neutral_report(self) -> None:
        """n < 31 → neutraler Short-Data-Report (kein Signal möglich)."""
        data = _make_ohlcv(30, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)
        assert "Insufficient data" in report.hypothesis
        assert report.probabilities == {"up": 0.34, "down": 0.33, "range": 0.33}
        assert abs(sum(report.probabilities.values()) - 1.0) <= 0.001
        assert len(report.evidence) >= 1
        assert report.raw_confidence == 0.08
        assert report.status == AgentStatus.SHADOW

    def test_min_bars_boundary(self) -> None:
        """n = 31 (MIN_BARS) analysiert regulär statt Short-Data."""
        data = _make_ohlcv(31, trend="up")
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert "Insufficient data" not in report.hypothesis
        assert abs(sum(report.probabilities.values()) - 1.0) <= 0.001

    def test_flat_prices_finite_output(self) -> None:
        """Flache/zero-range Kerzen: keine NaN/Inf, Summe 1.0, Range-Prior."""
        data = _make_flat_ohlcv(100)
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)
        for val in report.probabilities.values():
            assert math.isfinite(val)
            assert val >= 0.0
        assert abs(sum(report.probabilities.values()) - 1.0) <= 0.001
        # ohne feuende Familie: neutrale Verteilung (renormiert), Range dominant
        assert report.probabilities["up"] == report.probabilities["down"]
        assert report.probabilities["range"] > report.probabilities["up"]
        assert abs(report.probabilities["range"] - 0.3636) <= 0.001

    def test_flat_prices_short_data(self) -> None:
        """Flache Kerzen mit n < 31 bleiben neutral."""
        data = _make_flat_ohlcv(20)
        agent = ChartPatternAgent()
        report = agent.analyze(data)
        assert report.probabilities == {"up": 0.34, "down": 0.33, "range": 0.33}

    def test_missing_open_raises(self) -> None:
        agent = ChartPatternAgent()
        with pytest.raises(ValueError, match="Missing required OHLCV keys"):
            agent.analyze({"high": np.ones(50), "low": np.ones(50), "close": np.ones(50)})

    def test_missing_close_raises(self) -> None:
        agent = ChartPatternAgent()
        with pytest.raises(ValueError, match="Missing required OHLCV keys"):
            agent.analyze({"open": np.ones(50), "high": np.ones(50), "low": np.ones(50)})

    def test_empty_data_raises(self) -> None:
        agent = ChartPatternAgent()
        with pytest.raises(ValueError, match="Missing required OHLCV keys"):
            agent.analyze({})  # type: ignore[arg-type]


# ── Ensemble integration tests ───────────────────────────────────────────


class TestChartPatternAgentEnsemble:
    """Integrationstest: ChartPatternAgent ist 5. Ensemble-Mitglied."""

    def test_build_active_ensemble_has_five_agents(self) -> None:
        from apps.demo_trader.service import build_active_ensemble

        agents = build_active_ensemble("BTC/USDT", "1h")
        ids = [a.agent_id for a in agents]
        assert len(agents) == 5
        assert "chart_pattern" in ids
        assert ids == [
            "trend",
            "mean_reversion",
            "volatility_regime",
            "volume_conviction",
            "chart_pattern",
        ]

    def test_ensemble_agent_status_pinned_shadow(self) -> None:
        """chart_pattern bleibt im ACTIVE-Ensemble vorerst fest SHADOW (pin)."""
        from apps.demo_trader.service import build_active_ensemble

        agents = build_active_ensemble("BTC/USDT", "1h")
        cp = next(a for a in agents if a.agent_id == "chart_pattern")
        assert cp._agent.config.status == AgentStatus.SHADOW

    def test_ensemble_agent_produces_valid_report(self) -> None:
        """Der Ensemble-Agent liefert einen gültigen AgentReport."""
        from apps.demo_trader.service import build_active_ensemble

        agents = build_active_ensemble("BTC/USDT", "1h")
        cp = next(a for a in agents if a.agent_id == "chart_pattern")
        report = cp.analyze(_make_ohlcv(200, trend="up"))
        assert isinstance(report, AgentReport)
        assert abs(sum(report.probabilities.values()) - 1.0) <= 0.001
        assert len(report.evidence) >= 1
        assert report.status == AgentStatus.SHADOW
