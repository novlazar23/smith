"""Unit tests for the Mean-Reversion Agent (packages/agents/mean_reversion_agent.py)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from packages.agents.base import AgentConfig, AgentType
from packages.agents.mean_reversion_agent import (
    MeanReversionAgent,
    ReversionState,
)
from packages.schemas.agent_report import AgentReport, AgentStatus

# ── Deterministische synthetische OHLCV-Daten ───────────────────────────────


def _flat_ohlcv(n: int = 150) -> dict[str, NDArray[np.float64]]:
    """Flache Serie mit winziger deterministischer Welle — keine Ausprägung."""
    close = 100.0 + 0.1 * np.sin(np.arange(n, dtype=np.float64))
    return {
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "open": close,
        "volume": np.full(n, 1000.0),
    }


def _crash_ohlcv() -> dict[str, NDArray[np.float64]]:
    """140 flache Bars, dann 10 Bars Absturz 100 → 55 — oversold."""
    n_flat, n_crash = 140, 10
    close_flat = 100.0 + 0.1 * np.sin(np.arange(n_flat, dtype=np.float64))
    close_crash = np.linspace(100.0, 55.0, n_crash)
    close = np.concatenate([close_flat, close_crash])
    return {
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "open": close,
        "volume": np.full(len(close), 1000.0),
    }


def _pump_ohlcv() -> dict[str, NDArray[np.float64]]:
    """140 flache Bars, dann 10 Bars Anstieg 100 → 145 — overbought."""
    n_flat, n_pump = 140, 10
    close_flat = 100.0 + 0.1 * np.sin(np.arange(n_flat, dtype=np.float64))
    close_pump = np.linspace(100.0, 145.0, n_pump)
    close = np.concatenate([close_flat, close_pump])
    return {
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "open": close,
        "volume": np.full(len(close), 1000.0),
    }


def _pump_short_ohlcv(n: int = 60) -> dict[str, NDArray[np.float64]]:
    """Kleine Aufwärtsrampe unterhalb des Minimums (100)."""
    close = np.linspace(100.0, 130.0, n)
    return {
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "open": close,
        "volume": np.full(n, 1000.0),
    }


def _all_ohlcv_datasets() -> list[dict]:
    """Alle fünf deterministischen Szenarien (Crash, Pump, flat, kurz, kurz-pump)."""
    return [
        _crash_ohlcv(),
        _pump_ohlcv(),
        _flat_ohlcv(),
        _flat_ohlcv(50),
        _pump_short_ohlcv(),
    ]


@pytest.fixture
def crash_ohlcv() -> dict:
    """150 Bars: 140 flat, 10er-Absturz — oversold, Reversion nach oben."""
    return _crash_ohlcv()


@pytest.fixture
def pump_ohlcv() -> dict:
    """150 Bars: 140 flat, 10er-Anstieg — overbought, Reversion nach unten."""
    return _pump_ohlcv()


@pytest.fixture
def flat_ohlcv() -> dict:
    """150 Bars flach — nahe dem Mittelwert, keine Reversion."""
    return _flat_ohlcv()


@pytest.fixture
def short_ohlcv() -> dict:
    """50 Bars — unter dem Minimum von 100."""
    return _flat_ohlcv(50)


# ── Reversionsrichtung ──────────────────────────────────────────────────────


class TestReversionDirection:
    def test_crash_votes_up(self, crash_ohlcv: dict) -> None:
        """Absturz weit unter SMA50 → 'up' >= 0.65 (Konsens-Schwelle 0.6)."""
        report = MeanReversionAgent().analyze(crash_ohlcv)
        assert report.probabilities["up"] >= 0.65

    def test_crash_dominates_other_sides(self, crash_ohlcv: dict) -> None:
        report = MeanReversionAgent().analyze(crash_ohlcv)
        assert report.probabilities["up"] > report.probabilities["down"]
        assert report.probabilities["up"] > report.probabilities["range"]

    def test_pump_votes_down(self, pump_ohlcv: dict) -> None:
        """Anstieg weit über SMA50 → 'down' >= 0.65 (Spiegelbild)."""
        report = MeanReversionAgent().analyze(pump_ohlcv)
        assert report.probabilities["down"] >= 0.65

    def test_flat_votes_range(self, flat_ohlcv: dict) -> None:
        """Flache Serie nahe Mittelwert → 'range' dominant (>= 0.5)."""
        report = MeanReversionAgent().analyze(flat_ohlcv)
        assert report.probabilities["range"] >= 0.5


class TestClassifyState:
    """Klassifikationslogik auf State-Ebene (Entscheidungsregeln isoliert)."""

    @staticmethod
    def _state(z: float, rsi: float) -> ReversionState:
        return ReversionState(z=z, rsi=rsi, pct_distance=z * 0.1, sma50=100.0)

    def test_oversold_with_rsi_confirmation_is_reversion_up(self) -> None:
        assert (
            MeanReversionAgent()._classify(self._state(-2.5, 30.0)) == "reversion_up"
        )

    def test_overbought_with_rsi_confirmation_is_reversion_down(self) -> None:
        assert (
            MeanReversionAgent()._classify(self._state(2.5, 70.0)) == "reversion_down"
        )

    def test_extreme_z_against_rsi_is_range(self) -> None:
        """z extrem negativ, aber RSI überkauft → laufender Trend, keine Reversion."""
        assert MeanReversionAgent()._classify(self._state(-2.5, 70.0)) == "range"

    def test_extreme_positive_z_with_low_rsi_is_range(self) -> None:
        assert MeanReversionAgent()._classify(self._state(2.5, 30.0)) == "range"

    def test_middle_zone_is_range(self) -> None:
        assert MeanReversionAgent()._classify(self._state(0.5, 50.0)) == "range"

    def test_z_below_threshold_with_confirmation_is_range(self) -> None:
        """|z| < 2 reicht nicht, selbst mit RSI-Bestätigung."""
        assert MeanReversionAgent()._classify(self._state(-1.5, 20.0)) == "range"


# ── Validierung ─────────────────────────────────────────────────────────────


class TestValidation:
    def test_missing_volume_raises(self, crash_ohlcv: dict) -> None:
        data = dict(crash_ohlcv)
        del data["volume"]
        with pytest.raises(ValueError, match="volume"):
            MeanReversionAgent().analyze(data)

    def test_missing_multiple_keys_reports_sorted(self, crash_ohlcv: dict) -> None:
        data = {"close": crash_ohlcv["close"]}
        with pytest.raises(ValueError, match="high"):
            MeanReversionAgent().analyze(data)


# ── Kurze Daten ─────────────────────────────────────────────────────────────


class TestShortData:
    def test_short_data_neutral_without_exception(self, short_ohlcv: dict) -> None:
        """n < 100 → neutraler Bericht, keine Exception."""
        report = MeanReversionAgent().analyze(short_ohlcv)
        assert report.probabilities == {"up": 0.33, "down": 0.33, "range": 0.34}
        confidence = report.raw_confidence
        assert confidence is not None
        assert 0.05 <= confidence <= 0.1
        assert "insufficient" in report.hypothesis.lower()

    def test_short_data_sections_present(self, short_ohlcv: dict) -> None:
        report = MeanReversionAgent().analyze(short_ohlcv)
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1
        assert len(report.invalidations) >= 1


# ── Wahrscheinlichkeitsintegrität ───────────────────────────────────────────


class TestProbabilityIntegrity:
    def test_probabilities_sum_to_one(self) -> None:
        """Summe exakt 1.0 (± 1e-6) in allen Szenarien."""
        for data in _all_ohlcv_datasets():
            report = MeanReversionAgent().analyze(data)
            total = sum(report.probabilities.values())
            assert abs(total - 1.0) <= 1e-6

    def test_probabilities_native_floats_in_unit_interval(self) -> None:
        """Werte sind native Floats in [0, 1] — kein np.float64, nichts Negatives."""
        for data in _all_ohlcv_datasets():
            report = MeanReversionAgent().analyze(data)
            for key, value in report.probabilities.items():
                assert isinstance(value, float), f"{key} is {type(value)}, expected float"
                assert 0.0 <= value <= 1.0

    def test_deterministic_same_input_same_output(self, crash_ohlcv: dict) -> None:
        """Gleiche Eingabe zwei Mal → identische Wahrscheinlichkeiten."""
        agent = MeanReversionAgent()
        first = agent.analyze(crash_ohlcv)
        second = agent.analyze(crash_ohlcv)
        assert first.probabilities == second.probabilities
        assert first.hypothesis == second.hypothesis
        assert first.raw_confidence == second.raw_confidence


# ── Konfigurations-Propagation ──────────────────────────────────────────────


class TestConfigPropagation:
    def test_default_config(self) -> None:
        agent = MeanReversionAgent()
        assert agent.agent_id == "mean_reversion"
        assert agent.config.agent_type == AgentType.INDICATOR

    def test_status_and_metadata_propagated_to_report(self, crash_ohlcv: dict) -> None:
        config = AgentConfig(
            agent_id="mean_reversion",
            agent_type=AgentType.INDICATOR,
            instrument="BTC/USD",
            horizon="1h",
            status=AgentStatus.ACTIVE,
        )
        report = MeanReversionAgent(config).analyze(crash_ohlcv)
        assert report.agent_id == "mean_reversion"
        assert report.status == AgentStatus.ACTIVE
        assert report.instrument == "BTC/USD"
        assert report.horizon == "1h"
        assert report.agent_version == "0.1.0"


# ── Berichtstruktur ─────────────────────────────────────────────────────────


class TestReportStructure:
    def test_report_structure_and_boundaries(self, crash_ohlcv: dict) -> None:
        report = MeanReversionAgent().analyze(crash_ohlcv)
        assert isinstance(report, AgentReport)
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1
        assert 2 <= len(report.invalidations) <= 3
        confidence = report.raw_confidence
        assert confidence is not None
        assert 0.0 <= confidence <= 1.0
        assert report.calibrated_confidence == 0.0
        assert report.expected_return is None

    def test_evidence_covers_reversion_features(self, crash_ohlcv: dict) -> None:
        report = MeanReversionAgent().analyze(crash_ohlcv)
        features = {e.feature for e in report.evidence}
        assert {"z_score", "rsi14", "pct_distance"} <= features
        for e in report.evidence:
            assert e.direction in ("positive", "negative", "neutral")
            assert 0.0 <= e.relevance <= 1.0

    def test_confidence_rises_with_reversion_strength(
        self, crash_ohlcv: dict, flat_ohlcv: dict
    ) -> None:
        strong_conf = MeanReversionAgent().analyze(crash_ohlcv).raw_confidence
        flat_conf = MeanReversionAgent().analyze(flat_ohlcv).raw_confidence
        assert strong_conf is not None
        assert flat_conf is not None
        assert strong_conf >= 0.3
        assert 0.1 <= flat_conf <= 0.3

    def test_hypothesis_reflects_direction(
        self, crash_ohlcv: dict, pump_ohlcv: dict, flat_ohlcv: dict
    ) -> None:
        crash = MeanReversionAgent().analyze(crash_ohlcv)
        pump = MeanReversionAgent().analyze(pump_ohlcv)
        flat = MeanReversionAgent().analyze(flat_ohlcv)
        assert "oversold" in crash.hypothesis.lower()
        assert "overbought" in pump.hypothesis.lower()
        assert "no reversion" in flat.hypothesis.lower()
