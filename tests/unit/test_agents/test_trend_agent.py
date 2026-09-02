"""Unit tests for the Trend Agent (packages/agents/trend_agent.py)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from packages.agents.base import AgentConfig, AgentType
from packages.agents.trend_agent import TrendAgent
from packages.schemas.agent_report import AgentReport, AgentStatus

# ── Deterministische synthetische OHLCV-Daten ───────────────────────────────


def _ramp_ohlcv(
    n: int, step: float, start: float = 100.0
) -> dict[str, NDArray[np.float64]]:
    """Lineare Kursrampe mit konstanter Bar-Spanne und leicht steigendem Volume."""
    close = start + step * np.arange(n, dtype=np.float64)
    return {
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "open": close - step / 2.0,
        "volume": 1000.0 * (1.0 + 0.001 * np.arange(n, dtype=np.float64)),
    }


def _flat_ohlcv(n: int = 100) -> dict[str, NDArray[np.float64]]:
    """Konstanter Kurs — kein Trend."""
    return {
        "close": np.full(n, 100.0),
        "high": np.full(n, 101.0),
        "low": np.full(n, 99.0),
        "open": np.full(n, 100.0),
        "volume": np.full(n, 1000.0),
    }


def _all_ohlcv_datasets() -> list[dict]:
    """Alle vier deterministischen Szenarien (Auf-, Ab-, Seitwärts, kurz)."""
    return [
        _ramp_ohlcv(100, 0.5),
        _ramp_ohlcv(100, -0.5),
        _flat_ohlcv(),
        _ramp_ohlcv(20, 0.5),
    ]


@pytest.fixture
def uptrend_ohlcv() -> dict:
    """100 Bars, +0,5/Bar — starker, nachhaltiger Aufwärtstrend."""
    return _ramp_ohlcv(100, 0.5)


@pytest.fixture
def downtrend_ohlcv() -> dict:
    """100 Bars, -0,5/Bar — starker, nachhaltiger Abwärtstrend."""
    return _ramp_ohlcv(100, -0.5)


@pytest.fixture
def flat_ohlcv() -> dict:
    """100 Bars, konstanter Kurs — kein Trend."""
    return _flat_ohlcv()


@pytest.fixture
def short_ohlcv() -> dict:
    """20 Bars — unter dem Minimum von 50."""
    return _ramp_ohlcv(20, 0.5)


# ── Trendrichtung ───────────────────────────────────────────────────────────


class TestTrendDirection:
    def test_uptrend_votes_up(self, uptrend_ohlcv: dict) -> None:
        """Echte Rampe nach oben → 'up' >= 0.65 (Konsens-Schwelle 0.6)."""
        report = TrendAgent().analyze(uptrend_ohlcv)
        assert report.probabilities["up"] >= 0.65

    def test_uptrend_dominates_other_sides(self, uptrend_ohlcv: dict) -> None:
        report = TrendAgent().analyze(uptrend_ohlcv)
        assert report.probabilities["up"] > report.probabilities["down"]
        assert report.probabilities["up"] > report.probabilities["range"]

    def test_downtrend_votes_down(self, downtrend_ohlcv: dict) -> None:
        """Echte Rampe nach unten → 'down' >= 0.65 (Spiegelbild)."""
        report = TrendAgent().analyze(downtrend_ohlcv)
        assert report.probabilities["down"] >= 0.65

    def test_flat_votes_range(self, flat_ohlcv: dict) -> None:
        """Konstanter Kurs → 'range' dominant (>= 0.5)."""
        report = TrendAgent().analyze(flat_ohlcv)
        assert report.probabilities["range"] >= 0.5

    def test_flat_with_micro_noise_stays_range(self) -> None:
        """Flache Serie mit winziger deterministischer Welle bleibt Range."""
        n = 100
        close = 100.0 + 0.05 * np.sin(np.arange(n))
        data = {
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "open": close,
            "volume": np.full(n, 1000.0),
        }
        report = TrendAgent().analyze(data)
        assert report.probabilities["range"] >= 0.5

    def test_steeper_ramp_scores_higher(self) -> None:
        """Monotonie: stärkere Rampe → höhere 'up'-Wahrscheinlichkeit."""
        gentle = TrendAgent().analyze(_ramp_ohlcv(100, 0.1))
        steep = TrendAgent().analyze(_ramp_ohlcv(100, 1.0))
        assert gentle.probabilities["up"] >= 0.65
        assert steep.probabilities["up"] >= 0.65
        assert steep.probabilities["up"] >= gentle.probabilities["up"]


# ── Validierung ─────────────────────────────────────────────────────────────


class TestValidation:
    def test_missing_volume_raises(self, uptrend_ohlcv: dict) -> None:
        data = dict(uptrend_ohlcv)
        del data["volume"]
        with pytest.raises(ValueError, match="volume"):
            TrendAgent().analyze(data)

    def test_missing_multiple_keys_reports_sorted(self, uptrend_ohlcv: dict) -> None:
        data = {"close": uptrend_ohlcv["close"]}
        with pytest.raises(ValueError, match="high"):
            TrendAgent().analyze(data)


# ── Kurze Daten ─────────────────────────────────────────────────────────────


class TestShortData:
    def test_short_data_neutral_without_exception(self, short_ohlcv: dict) -> None:
        """n < 50 → neutraler Bericht, keine Exception."""
        report = TrendAgent().analyze(short_ohlcv)
        assert report.probabilities == {"up": 0.33, "down": 0.33, "range": 0.34}
        confidence = report.raw_confidence
        assert confidence is not None
        assert 0.05 <= confidence <= 0.1
        assert "insufficient" in report.hypothesis.lower()

    def test_short_data_sections_present(self, short_ohlcv: dict) -> None:
        report = TrendAgent().analyze(short_ohlcv)
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1
        assert len(report.invalidations) >= 1


# ── Wahrscheinlichkeitsintegrität ───────────────────────────────────────────


class TestProbabilityIntegrity:
    def test_probabilities_sum_to_one(self) -> None:
        """AT-005: Summe exakt 1.0 (± 1e-6) in allen vier Szenarien."""
        for data in _all_ohlcv_datasets():
            report = TrendAgent().analyze(data)
            total = sum(report.probabilities.values())
            assert abs(total - 1.0) <= 1e-6

    def test_probabilities_native_floats_in_unit_interval(self) -> None:
        """Werte sind native Floats in [0, 1] — kein np.float64, nichts Negatives."""
        for data in _all_ohlcv_datasets():
            report = TrendAgent().analyze(data)
            for key, value in report.probabilities.items():
                assert isinstance(value, float), f"{key} is {type(value)}, expected float"
                assert 0.0 <= value <= 1.0

    def test_deterministic_same_input_same_output(self, uptrend_ohlcv: dict) -> None:
        """Gleiche Eingabe zwei Mal → identische Wahrscheinlichkeiten."""
        agent = TrendAgent()
        first = agent.analyze(uptrend_ohlcv)
        second = agent.analyze(uptrend_ohlcv)
        assert first.probabilities == second.probabilities
        assert first.hypothesis == second.hypothesis
        assert first.raw_confidence == second.raw_confidence


# ── Konfigurations-Propagation ──────────────────────────────────────────────


class TestConfigPropagation:
    def test_default_config(self) -> None:
        agent = TrendAgent()
        assert agent.agent_id == "trend"
        assert agent.config.agent_type == AgentType.INDICATOR

    def test_status_and_metadata_propagated_to_report(self, uptrend_ohlcv: dict) -> None:
        config = AgentConfig(
            agent_id="trend",
            agent_type=AgentType.INDICATOR,
            instrument="BTC/USD",
            horizon="1h",
            status=AgentStatus.ACTIVE,
        )
        report = TrendAgent(config).analyze(uptrend_ohlcv)
        assert report.agent_id == "trend"
        assert report.status == AgentStatus.ACTIVE
        assert report.instrument == "BTC/USD"
        assert report.horizon == "1h"
        assert report.agent_version == "0.1.0"


# ── Berichtstruktur ─────────────────────────────────────────────────────────


class TestReportStructure:
    def test_report_structure_and_boundaries(self, uptrend_ohlcv: dict) -> None:
        report = TrendAgent().analyze(uptrend_ohlcv)
        assert isinstance(report, AgentReport)
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1
        assert 2 <= len(report.invalidations) <= 3
        confidence = report.raw_confidence
        assert confidence is not None
        assert 0.0 <= confidence <= 1.0
        assert report.calibrated_confidence == 0.0
        assert report.expected_return is None

    def test_evidence_covers_trend_features(self, uptrend_ohlcv: dict) -> None:
        report = TrendAgent().analyze(uptrend_ohlcv)
        features = {e.feature for e in report.evidence}
        assert {"ema_separation", "ema_slope", "roc10", "atr14"} <= features
        for e in report.evidence:
            assert e.direction in ("positive", "negative", "neutral")
            assert 0.0 <= e.relevance <= 1.0

    def test_confidence_rises_with_trend_strength(
        self, uptrend_ohlcv: dict, flat_ohlcv: dict
    ) -> None:
        strong_conf = TrendAgent().analyze(uptrend_ohlcv).raw_confidence
        flat_conf = TrendAgent().analyze(flat_ohlcv).raw_confidence
        assert strong_conf is not None
        assert flat_conf is not None
        assert strong_conf >= 0.3
        assert 0.1 <= flat_conf <= 0.3

    def test_hypothesis_reflects_direction(
        self, uptrend_ohlcv: dict, downtrend_ohlcv: dict, flat_ohlcv: dict
    ) -> None:
        up = TrendAgent().analyze(uptrend_ohlcv)
        down = TrendAgent().analyze(downtrend_ohlcv)
        flat = TrendAgent().analyze(flat_ohlcv)
        assert "uptrend" in up.hypothesis.lower()
        assert "downtrend" in down.hypothesis.lower()
        assert "no sustained trend" in flat.hypothesis.lower()
