"""Unit tests for the Volatility-Regime Agent (packages/agents/volatility_regime_agent.py)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from packages.agents.base import AgentConfig, AgentType
from packages.agents.volatility_regime_agent import (
    RegimeState,
    VolatilityRegimeAgent,
)
from packages.schemas.agent_report import AgentReport, AgentStatus

# ── Deterministische synthetische OHLCV-Daten ───────────────────────────────


def _normal_ohlcv(n: int = 150) -> dict[str, NDArray[np.float64]]:
    """Normale Volatilität: Sinuswelle mit Amplitude 1 um 100."""
    close = 100.0 + 1.0 * np.sin(np.arange(n, dtype=np.float64))
    return {
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "open": close,
        "volume": np.full(n, 1000.0),
    }


def _breakout_up_ohlcv() -> dict[str, NDArray[np.float64]]:
    """100 normal, 40 Squeeze (Amplitude 0,05), dann 10er-Ausbruch nach oben."""
    n_normal, n_squeeze, n_break = 100, 40, 10
    close_normal = 100.0 + 1.0 * np.sin(np.arange(n_normal, dtype=np.float64))
    close_squeeze = 100.0 + 0.05 * np.sin(np.arange(n_squeeze, dtype=np.float64))
    close_break = 100.5 + 0.5 * np.arange(n_break, dtype=np.float64)
    close = np.concatenate([close_normal, close_squeeze, close_break])
    return {
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "open": close,
        "volume": np.full(len(close), 1000.0),
    }


def _breakout_down_ohlcv() -> dict[str, NDArray[np.float64]]:
    """100 normal, 40 Squeeze, dann 10er-Ausbruch nach unten (Spiegelbild)."""
    n_normal, n_squeeze, n_break = 100, 40, 10
    close_normal = 100.0 + 1.0 * np.sin(np.arange(n_normal, dtype=np.float64))
    close_squeeze = 100.0 + 0.05 * np.sin(np.arange(n_squeeze, dtype=np.float64))
    close_break = 99.5 - 0.5 * np.arange(n_break, dtype=np.float64)
    close = np.concatenate([close_normal, close_squeeze, close_break])
    return {
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "open": close,
        "volume": np.full(len(close), 1000.0),
    }


def _expansion_ohlcv() -> dict[str, NDArray[np.float64]]:
    """100 normal, dann 50 Bars mit stark vergrößerter Amplitude (3,0)."""
    n_normal, n_exp = 100, 50
    close_normal = 100.0 + 1.0 * np.sin(np.arange(n_normal, dtype=np.float64))
    close_exp = 100.0 + 3.0 * np.sin(2.0 * np.arange(n_exp, dtype=np.float64))
    close = np.concatenate([close_normal, close_exp])
    return {
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "open": close,
        "volume": np.full(len(close), 1000.0),
    }


def _all_ohlcv_datasets() -> list[dict]:
    """Alle fünf deterministischen Szenarien (Breakout x2, Expansion, normal, kurz)."""
    return [
        _breakout_up_ohlcv(),
        _breakout_down_ohlcv(),
        _expansion_ohlcv(),
        _normal_ohlcv(),
        _normal_ohlcv(100),
    ]


@pytest.fixture
def breakout_up_ohlcv() -> dict:
    """Squeeze + Ausbruch an der 20-Bar-Oben-Grenze."""
    return _breakout_up_ohlcv()


@pytest.fixture
def breakout_down_ohlcv() -> dict:
    """Squeeze + Ausbruch an der 20-Bar-Unten-Grenze."""
    return _breakout_down_ohlcv()


@pytest.fixture
def expansion_ohlcv() -> dict:
    """Vol-Expansion ohne richtungsbestimmten Ausbruch."""
    return _expansion_ohlcv()


@pytest.fixture
def normal_ohlcv() -> dict:
    """Gleichmäßige normale Volatilität — kein Regime-Wechsel."""
    return _normal_ohlcv()


@pytest.fixture
def short_ohlcv() -> dict:
    """100 Bars — unter dem Minimum von 150."""
    return _normal_ohlcv(100)


# ── Regime-Richtung ─────────────────────────────────────────────────────────


class TestRegimeDirection:
    def test_squeeze_breakout_up_votes_up(self, breakout_up_ohlcv: dict) -> None:
        """Squeeze + Ausbruch nach oben → 'up' >= 0.65 (Konsens-Schwelle 0.6)."""
        report = VolatilityRegimeAgent().analyze(breakout_up_ohlcv)
        assert report.probabilities["up"] >= 0.65

    def test_squeeze_breakout_up_dominates(self, breakout_up_ohlcv: dict) -> None:
        report = VolatilityRegimeAgent().analyze(breakout_up_ohlcv)
        assert report.probabilities["up"] > report.probabilities["down"]
        assert report.probabilities["up"] > report.probabilities["range"]

    def test_squeeze_breakdown_votes_down(self, breakout_down_ohlcv: dict) -> None:
        """Squeeze + Ausbruch nach unten → 'down' >= 0.65 (Spiegelbild)."""
        report = VolatilityRegimeAgent().analyze(breakout_down_ohlcv)
        assert report.probabilities["down"] >= 0.65

    def test_expansion_votes_range(self, expansion_ohlcv: dict) -> None:
        """Vol-Expansion ohne Ausbruch → 'range' dominant (>= 0.5)."""
        report = VolatilityRegimeAgent().analyze(expansion_ohlcv)
        assert report.probabilities["range"] >= 0.5

    def test_normal_volatility_votes_range(self, normal_ohlcv: dict) -> None:
        """Gleichmäßige Volatilität → 'range' dominant (>= 0.5)."""
        report = VolatilityRegimeAgent().analyze(normal_ohlcv)
        assert report.probabilities["range"] >= 0.5


class TestClassifyState:
    """Klassifikationslogik auf State-Ebene (Entscheidungsregeln isoliert)."""

    @staticmethod
    def _state(
        squeeze: float, position: float, expansion: float = 1.0
    ) -> RegimeState:
        return RegimeState(
            squeeze_ratio=squeeze,
            expansion_ratio=expansion,
            position20=position,
            atr14=1.0,
        )

    def test_squeeze_at_upper_edge_is_up(self) -> None:
        assert VolatilityRegimeAgent()._classify(self._state(0.3, 0.9)) == "up"

    def test_squeeze_at_lower_edge_is_down(self) -> None:
        assert VolatilityRegimeAgent()._classify(self._state(0.3, 0.1)) == "down"

    def test_squeeze_mid_band_is_range(self) -> None:
        assert VolatilityRegimeAgent()._classify(self._state(0.3, 0.5)) == "range"

    def test_no_squeeze_at_edge_is_range(self) -> None:
        """An der Kante, aber ohne Squeeze → kein Ausbruchssignal."""
        assert VolatilityRegimeAgent()._classify(self._state(1.0, 0.9)) == "range"

    def test_expansion_without_squeeze_mid_band_is_range(self) -> None:
        """Expansion ohne Squeeze in der Bandmitte → 'range' (unsicheres Regime).

        Ein gültiger Squeeze-Ausbruch (squeeze <= 0,5 an der Kante) wird
        dagegen durch die Expansion nicht blockiert — die Bandverbreiterung
        ist dessen Folge, kein Gegenindikator.
        """
        assert (
            VolatilityRegimeAgent()._classify(self._state(1.0, 0.5, expansion=2.5))
            == "range"
        )


# ── Validierung ─────────────────────────────────────────────────────────────


class TestValidation:
    def test_missing_volume_raises(self, breakout_up_ohlcv: dict) -> None:
        data = dict(breakout_up_ohlcv)
        del data["volume"]
        with pytest.raises(ValueError, match="volume"):
            VolatilityRegimeAgent().analyze(data)

    def test_missing_multiple_keys_reports_sorted(self, breakout_up_ohlcv: dict) -> None:
        data = {"close": breakout_up_ohlcv["close"]}
        with pytest.raises(ValueError, match="high"):
            VolatilityRegimeAgent().analyze(data)


# ── Kurze Daten ─────────────────────────────────────────────────────────────


class TestShortData:
    def test_short_data_neutral_without_exception(self, short_ohlcv: dict) -> None:
        """n < 150 → neutraler Bericht, keine Exception."""
        report = VolatilityRegimeAgent().analyze(short_ohlcv)
        assert report.probabilities == {"up": 0.33, "down": 0.33, "range": 0.34}
        confidence = report.raw_confidence
        assert confidence is not None
        assert 0.05 <= confidence <= 0.1
        assert "insufficient" in report.hypothesis.lower()

    def test_short_data_sections_present(self, short_ohlcv: dict) -> None:
        report = VolatilityRegimeAgent().analyze(short_ohlcv)
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1
        assert len(report.invalidations) >= 1


# ── Wahrscheinlichkeitsintegrität ───────────────────────────────────────────


class TestProbabilityIntegrity:
    def test_probabilities_sum_to_one(self) -> None:
        """Summe exakt 1.0 (± 1e-6) in allen Szenarien."""
        for data in _all_ohlcv_datasets():
            report = VolatilityRegimeAgent().analyze(data)
            total = sum(report.probabilities.values())
            assert abs(total - 1.0) <= 1e-6

    def test_probabilities_native_floats_in_unit_interval(self) -> None:
        """Werte sind native Floats in [0, 1] — kein np.float64, nichts Negatives."""
        for data in _all_ohlcv_datasets():
            report = VolatilityRegimeAgent().analyze(data)
            for key, value in report.probabilities.items():
                assert isinstance(value, float), f"{key} is {type(value)}, expected float"
                assert 0.0 <= value <= 1.0

    def test_deterministic_same_input_same_output(self, breakout_up_ohlcv: dict) -> None:
        """Gleiche Eingabe zwei Mal → identische Wahrscheinlichkeiten."""
        agent = VolatilityRegimeAgent()
        first = agent.analyze(breakout_up_ohlcv)
        second = agent.analyze(breakout_up_ohlcv)
        assert first.probabilities == second.probabilities
        assert first.hypothesis == second.hypothesis
        assert first.raw_confidence == second.raw_confidence


# ── Konfigurations-Propagation ──────────────────────────────────────────────


class TestConfigPropagation:
    def test_default_config(self) -> None:
        agent = VolatilityRegimeAgent()
        assert agent.agent_id == "volatility_regime"
        assert agent.config.agent_type == AgentType.REGIME

    def test_status_and_metadata_propagated_to_report(self, breakout_up_ohlcv: dict) -> None:
        config = AgentConfig(
            agent_id="volatility_regime",
            agent_type=AgentType.REGIME,
            instrument="BTC/USD",
            horizon="1h",
            status=AgentStatus.ACTIVE,
        )
        report = VolatilityRegimeAgent(config).analyze(breakout_up_ohlcv)
        assert report.agent_id == "volatility_regime"
        assert report.status == AgentStatus.ACTIVE
        assert report.instrument == "BTC/USD"
        assert report.horizon == "1h"
        assert report.agent_version == "0.1.0"


# ── Berichtstruktur ─────────────────────────────────────────────────────────


class TestReportStructure:
    def test_report_structure_and_boundaries(self, breakout_up_ohlcv: dict) -> None:
        report = VolatilityRegimeAgent().analyze(breakout_up_ohlcv)
        assert isinstance(report, AgentReport)
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1
        assert 2 <= len(report.invalidations) <= 3
        confidence = report.raw_confidence
        assert confidence is not None
        assert 0.0 <= confidence <= 1.0
        assert report.calibrated_confidence == 0.0
        assert report.expected_return is None

    def test_evidence_covers_regime_features(self, breakout_up_ohlcv: dict) -> None:
        report = VolatilityRegimeAgent().analyze(breakout_up_ohlcv)
        features = {e.feature for e in report.evidence}
        assert {"squeeze_ratio", "expansion_ratio", "position20", "atr14"} <= features
        for e in report.evidence:
            assert e.direction in ("positive", "negative", "neutral")
            assert 0.0 <= e.relevance <= 1.0

    def test_confidence_rises_with_breakout_strength(
        self, breakout_up_ohlcv: dict, normal_ohlcv: dict
    ) -> None:
        strong_conf = VolatilityRegimeAgent().analyze(breakout_up_ohlcv).raw_confidence
        normal_conf = VolatilityRegimeAgent().analyze(normal_ohlcv).raw_confidence
        assert strong_conf is not None
        assert normal_conf is not None
        assert strong_conf >= 0.3
        assert 0.1 <= normal_conf <= 0.3

    def test_hypothesis_reflects_direction(
        self,
        breakout_up_ohlcv: dict,
        breakout_down_ohlcv: dict,
        expansion_ohlcv: dict,
    ) -> None:
        up = VolatilityRegimeAgent().analyze(breakout_up_ohlcv)
        down = VolatilityRegimeAgent().analyze(breakout_down_ohlcv)
        expansion = VolatilityRegimeAgent().analyze(expansion_ohlcv)
        assert "upside" in up.hypothesis.lower()
        assert "downside" in down.hypothesis.lower()
        assert "no directional breakout" in expansion.hypothesis.lower()
