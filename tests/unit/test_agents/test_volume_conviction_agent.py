"""Unit tests für den Volumen-Konviktions-Agenten.

Deterministische synthetische OHLCV-Daten: open[i] = close[i-1], damit die
Up/Down-Klassifikation pro Bar kontrolliert ist (close[i] vs. open[i]).
"""

from __future__ import annotations

import numpy as np
import pytest
from packages.agents.base import AgentConfig, AgentType
from packages.agents.volume_conviction_agent import REQUIRED_KEYS, VolumeConvictionAgent
from packages.schemas.agent_report import AgentReport

# ── Synthetische Datenbuilder ─────────────────────────────────────────────


def _finish(close: np.ndarray, volume: np.ndarray) -> dict[str, np.ndarray]:
    """open[i] = close[i-1]; high/low umschließen open/close."""
    n = len(close)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _accumulation_data(n: int = 60) -> dict[str, np.ndarray]:
    """Akkumulation: +0,2/Bar bei Kaufvolumen 1000, jede 5. Bar -0,1 bei 200."""
    close = np.empty(n)
    volume = np.empty(n)
    close[0] = 100.0
    volume[0] = 1000.0
    for i in range(1, n):
        if i % 5 == 4:
            close[i] = close[i - 1] - 0.1
            volume[i] = 200.0
        else:
            close[i] = close[i - 1] + 0.2
            volume[i] = 1000.0
    return _finish(close, volume)


def _distribution_data(n: int = 60) -> dict[str, np.ndarray]:
    """Distribution: -0,2/Bar bei Verkaufsvolumen 1000, jede 5. Bar +0,1 bei 200."""
    close = np.empty(n)
    volume = np.empty(n)
    close[0] = 100.0
    volume[0] = 1000.0
    for i in range(1, n):
        if i % 5 == 4:
            close[i] = close[i - 1] + 0.1
            volume[i] = 200.0
        else:
            close[i] = close[i - 1] - 0.2
            volume[i] = 1000.0
    return _finish(close, volume)


def _flat_data(n: int = 60) -> dict[str, np.ndarray]:
    """Flat: ±0,05 um 100 oszillierend, gleiches niedriges Volumen 100 pro Bar."""
    close = np.array([100.05 if i % 2 == 1 else 99.95 for i in range(n)])
    volume = np.full(n, 100.0)
    return _finish(close, volume)


def _divergence_data(n: int = 60) -> dict[str, np.ndarray]:
    """Divergenz: Kurs steigt netto (+0,3/Pair), Volumen liegt aber auf Downbars."""
    close = np.empty(n)
    volume = np.empty(n)
    close[0] = 100.0
    volume[0] = 200.0
    for i in range(1, n):
        if i % 2 == 1:
            close[i] = close[i - 1] + 0.5
            volume[i] = 200.0
        else:
            close[i] = close[i - 1] - 0.2
            volume[i] = 1000.0
    return _finish(close, volume)


def _short_data(n: int = 30) -> dict[str, np.ndarray]:
    """Zu kurz für die Konviktions-Indikatoren (< 50 Kerzen)."""
    close = np.linspace(100.0, 106.0, n)
    volume = np.full(n, 500.0)
    return _finish(close, volume)


def _all_scenarios() -> list[dict[str, np.ndarray]]:
    return [
        _accumulation_data(),
        _distribution_data(),
        _flat_data(),
        _divergence_data(),
        _short_data(),
    ]


# ── Konfiguration ──────────────────────────────────────────────────────────


class TestConfig:
    def test_default_agent_id_and_type(self) -> None:
        agent = VolumeConvictionAgent()
        assert agent.agent_id == "volume_conviction"
        assert agent.config.agent_type == AgentType.ORDERFLOW

    def test_custom_config(self) -> None:
        config = AgentConfig(
            agent_id="volume_conviction",
            instrument="BTC/USDT",
            horizon="4h",
        )
        agent = VolumeConvictionAgent(config)
        assert agent.config.instrument == "BTC/USDT"
        assert agent.config.horizon == "4h"

    def test_required_keys_contract(self) -> None:
        assert frozenset({"open", "high", "low", "close", "volume"}) == REQUIRED_KEYS


# ── Votings / Kalibrierung ─────────────────────────────────────────────────


class TestConviction:
    def test_accumulation_votes_up_above_threshold(self) -> None:
        """Kaufvolumen-Dominanz muss 'up' > 0,6 liefern (Konsens-Gate)."""
        report = VolumeConvictionAgent().analyze(_accumulation_data())
        assert report.probabilities["up"] >= 0.65
        assert report.probabilities["up"] > report.probabilities["down"]
        assert report.probabilities["up"] > report.probabilities["range"]

    def test_distribution_votes_down_above_threshold(self) -> None:
        report = VolumeConvictionAgent().analyze(_distribution_data())
        assert report.probabilities["down"] >= 0.65
        assert report.probabilities["down"] > report.probabilities["up"]

    def test_flat_price_no_direction(self) -> None:
        """Flacher Kurs bei gleichem dünnen Volumen → range oder neutral."""
        report = VolumeConvictionAgent().analyze(_flat_data())
        p = report.probabilities
        assert p["range"] >= 0.5 or (p["up"], p["down"], p["range"]) == (
            0.33,
            0.33,
            0.34,
        )
        assert p["up"] <= 0.33
        assert p["down"] <= 0.33

    def test_divergence_does_not_vote_up(self) -> None:
        """Preis hoch, Volumen auf Downbars → keine 'up'-Überzeugung."""
        report = VolumeConvictionAgent().analyze(_divergence_data())
        p = report.probabilities
        assert p["up"] <= 0.6
        assert p["range"] >= 0.5

    def test_conviction_probability_capped_at_0_85(self) -> None:
        for data in (_accumulation_data(), _distribution_data()):
            report = VolumeConvictionAgent().analyze(data)
            for value in report.probabilities.values():
                assert value <= 0.85


# ── Validierung ────────────────────────────────────────────────────────────


class TestValidation:
    def test_missing_volume_raises_value_error(self) -> None:
        data = _accumulation_data()
        del data["volume"]
        with pytest.raises(ValueError, match="volume"):
            VolumeConvictionAgent().analyze(data)

    def test_missing_multiple_keys_sorted_in_message(self) -> None:
        with pytest.raises(ValueError, match=r"close.*open"):
            VolumeConvictionAgent().analyze({"high": np.ones(60)})


# ── Wahrscheinlichkeiten ───────────────────────────────────────────────────


class TestProbabilities:
    def test_probabilities_sum_to_one(self) -> None:
        for data in _all_scenarios():
            report = VolumeConvictionAgent().analyze(data)
            assert abs(sum(report.probabilities.values()) - 1.0) <= 1e-6

    def test_probabilities_bounded_native_floats(self) -> None:
        for data in _all_scenarios():
            report = VolumeConvictionAgent().analyze(data)
            assert set(report.probabilities) == {"up", "down", "range"}
            for value in report.probabilities.values():
                assert isinstance(value, float)
                assert 0.0 <= value <= 1.0

    def test_determinism_same_input_same_output(self) -> None:
        agent = VolumeConvictionAgent()
        data = _accumulation_data()
        first = agent.analyze(data)
        second = agent.analyze(data)
        assert first.probabilities == second.probabilities
        assert first.raw_confidence == second.raw_confidence
        assert first.hypothesis == second.hypothesis


# ── Konfidenz ──────────────────────────────────────────────────────────────


class TestConfidence:
    def test_confidence_neutral_low(self) -> None:
        report = VolumeConvictionAgent().analyze(_flat_data())
        assert report.raw_confidence is not None
        assert 0.1 <= report.raw_confidence <= 0.3

    def test_confidence_short_data_very_low(self) -> None:
        report = VolumeConvictionAgent().analyze(_short_data())
        assert report.raw_confidence is not None
        assert report.raw_confidence <= 0.1

    def test_confidence_conviction_high(self) -> None:
        report = VolumeConvictionAgent().analyze(_accumulation_data())
        assert report.raw_confidence is not None
        assert 0.4 <= report.raw_confidence <= 0.9


# ── Kurzdaten ──────────────────────────────────────────────────────────────


class TestShortData:
    def test_short_data_neutral_no_exception(self) -> None:
        report = VolumeConvictionAgent().analyze(_short_data())
        assert report.probabilities == {"up": 0.33, "down": 0.33, "range": 0.34}
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1
        assert "unzureichend" in report.hypothesis.lower()


# ── Report-Struktur ────────────────────────────────────────────────────────


class TestReportStructure:
    def test_report_type_and_agent_id(self) -> None:
        report = VolumeConvictionAgent().analyze(_accumulation_data())
        assert isinstance(report, AgentReport)
        assert report.agent_id == "volume_conviction"
        assert report.agent_version == "0.1.0"
        assert report.status == "shadow"
        assert report.expected_return is None
        assert report.calibrated_confidence == 0.0

    def test_evidence_features_present_and_scored(self) -> None:
        report = VolumeConvictionAgent().analyze(_accumulation_data())
        features = {e.feature for e in report.evidence}
        assert {"up_down_volume_ratio", "obv_slope", "participation"} <= features
        for e in report.evidence:
            assert e.direction in ("positive", "negative", "neutral")
            assert 0.0 <= e.relevance <= 1.0

    def test_evidence_direction_up_accumulation(self) -> None:
        report = VolumeConvictionAgent().analyze(_accumulation_data())
        by_feature = {e.feature: e for e in report.evidence}
        assert by_feature["up_down_volume_ratio"].direction == "positive"
        assert by_feature["obv_slope"].direction == "positive"

    def test_counter_evidence_present_and_negative(self) -> None:
        for data in _all_scenarios():
            report = VolumeConvictionAgent().analyze(data)
            assert len(report.counter_evidence) >= 1
            assert all(e.direction == "negative" for e in report.counter_evidence)

    def test_invalidations_between_two_and_three(self) -> None:
        report = VolumeConvictionAgent().analyze(_accumulation_data())
        assert 2 <= len(report.invalidations) <= 3
        for inv in report.invalidations:
            assert inv.direction in ("above", "below")
