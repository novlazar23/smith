"""Tests für den Champion-Evaluations-Kern (Brier/Stabilität/LOO + Replay)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from apps.champion_evals.score import (
    EvalSample,
    build_artifact,
    normalize_probs,
    replay_ensemble,
    score_window,
    write_artifact,
)
from apps.orchestrator_service.champion_feed import REQUALIFICATION_CONFIG, load_status_overrides
from packages.backtesting.core import Candle
from packages.schemas.agent_report import AgentStatus


def _candles(n: int) -> list[Candle]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Candle(
            timestamp=base + timedelta(minutes=5 * i),
            symbol="BTC/USDT",
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.0 + i,
            volume=1.0,
        )
        for i in range(n)
    ]


def _four_samples() -> list[EvalSample]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    a = {"UP": 0.9, "DOWN": 0.05, "RANGE": 0.05}  # votet stark UP (Outcome ist UP)
    b = {"UP": 0.05, "DOWN": 0.9, "RANGE": 0.05}  # votet stark DOWN (Outcome ist UP)
    return [
        EvalSample(as_of=base + timedelta(hours=k), per_agent_probs={"a": a, "b": b}, actual="UP", realized_return=0.0)
        for k in range(4)
    ]


class TestNormalizeProbs:
    def test_maps_lowercase_to_uppercase(self) -> None:
        assert normalize_probs({"up": 0.7, "down": 0.2, "range": 0.1}) == {"UP": 0.7, "DOWN": 0.2, "RANGE": 0.1}

    def test_ignores_unknown_keys(self) -> None:
        assert normalize_probs({"up": 0.5, "foo": 0.1}) == {"UP": 0.5}


class TestScoreWindow:
    def test_per_agent_brier_and_stability(self) -> None:
        m = score_window(_four_samples(), calibration_ratio=0.5)
        assert set(m) == {"a", "b"}
        assert m["a"].oos_brier == pytest.approx(0.015)  # (0.9-1)^2 + 0.05^2 + 0.05^2
        assert m["b"].oos_brier == pytest.approx(1.715)  # (0.05-1)^2 + 0.9^2 + 0.05^2
        assert m["a"].cal_samples == 2
        assert m["a"].oos_samples == 2
        assert m["a"].oos_stability == pytest.approx(1.0)  # argmax UP == Outcome UP
        assert m["b"].oos_stability == pytest.approx(0.0)  # argmax DOWN != Outcome UP

    def test_loo_marginal_sign(self) -> None:
        m = score_window(_four_samples(), calibration_ratio=0.5)
        assert m["a"].oos_marginal > 0  # a senkt den Ensemble-Brier
        assert m["b"].oos_marginal < 0  # b erhöht den Ensemble-Brier

    def test_pools_multi_instrument_samples_on_common_timeline(self) -> None:
        # Quell-1 (Instrument A, Stunden 0-2) und Quell-2 (Instrument B,
        # Stunden 3-5) werden in Quell-Reihenfolge concat'd, aber auf der
        # gepoolten Zeitachse gesplittet: früherer Quell = Kalibrierung,
        # späterer = OOS.
        base = datetime(2024, 1, 1, tzinfo=UTC)
        good = {"UP": 0.9, "DOWN": 0.05, "RANGE": 0.05}
        bad = {"UP": 0.05, "DOWN": 0.9, "RANGE": 0.05}
        source_a = [EvalSample(base + timedelta(hours=k), {"a": good}, "UP", 0.0) for k in range(3)]
        source_b = [EvalSample(base + timedelta(hours=3 + k), {"a": bad}, "UP", 0.0) for k in range(3)]
        m = score_window(source_a + source_b, calibration_ratio=0.5)
        assert m["a"].cal_samples == 3
        assert m["a"].oos_samples == 3
        assert m["a"].cal_brier == pytest.approx(0.015)  # Kalibrierung = guter Quell A
        assert m["a"].oos_brier == pytest.approx(1.715)  # OOS = schlechter Quell B

    def test_drops_incomplete_agents(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        a = {"UP": 0.9, "DOWN": 0.05, "RANGE": 0.05}
        c = {"UP": 0.5, "DOWN": 0.3, "RANGE": 0.2}
        samples = [
            EvalSample(base, {"a": a, "c": c}, "UP", 0.0),
            EvalSample(base + timedelta(hours=1), {"a": a}, "UP", 0.0),  # c fehlt
            EvalSample(base + timedelta(hours=2), {"a": a, "c": c}, "UP", 0.0),
            EvalSample(base + timedelta(hours=3), {"a": a, "c": c}, "UP", 0.0),
        ]
        assert set(score_window(samples, calibration_ratio=0.5)) == {"a"}


class TestArtifact:
    def test_roundtrip_requalification(self, tmp_path: Path) -> None:
        metrics = score_window(_four_samples(), calibration_ratio=0.5)
        path = write_artifact(tmp_path / "evals.json", metrics)
        overrides = load_status_overrides(path, config=REQUALIFICATION_CONFIG)
        assert overrides["a"] == AgentStatus.ACTIVE  # stabiles OOS + positiver LOO
        assert overrides["b"] == AgentStatus.SHADOW  # negativer LOO-Marginal

    def test_build_artifact_scores(self) -> None:
        m = score_window(_four_samples(), calibration_ratio=0.5)
        a = build_artifact(m)["a"]
        assert a["champion"]["oos_score"] == pytest.approx(1.0 - m["a"].cal_brier)
        assert a["challenger"]["oos_score"] == pytest.approx(1.0 - m["a"].oos_brier)
        assert a["challenger"]["marginal_contribution"] == pytest.approx(m["a"].oos_marginal)
        assert a["challenger"]["stability_score"] == pytest.approx(m["a"].oos_stability)
        assert a["new_risks"] == []
        assert a["shadow_success"] is True


class _FakeReport:
    def __init__(self, agent_id: str, probs: dict[str, float]) -> None:
        self.agent_id = agent_id
        self.probabilities = probs


class _FakeResult:
    def __init__(self, reports: list[_FakeReport]) -> None:
        self.first_round_reports = reports


class _FakePipeline:
    def __init__(self, seen: list[float]) -> None:
        self._seen = seen

    def run(self, run_id: str, instrument: str, agents: object, market_data: object) -> _FakeResult:
        self._seen.append(float(market_data["close"][-1]))
        return _FakeResult(
            [
                _FakeReport("trend", {"up": 0.7, "down": 0.2, "range": 0.1}),
                _FakeReport("vol", {"up": 0.1, "down": 0.7, "range": 0.2}),
            ]
        )


class TestReplayEnsemble:
    def test_structure_no_lookahead_and_actual(self) -> None:
        candles = _candles(50)
        seen: list[float] = []
        samples = replay_ensemble(
            candles,
            "BTC/USDT",
            "15m",
            candle_limit=200,
            min_candles=10,
            evaluate_every=5,
            horizon_bars=2,
            pipeline_factory=lambda: _FakePipeline(seen),
            ensemble_factory=lambda instrument, horizon: ["trend", "vol"],
        )
        assert len(samples) == 8
        by_ts = {c.timestamp: c.close for c in candles}
        for k, s in enumerate(samples):
            assert set(s.per_agent_probs) == {"trend", "vol"}
            assert s.per_agent_probs["trend"]["UP"] == pytest.approx(0.7)
            assert seen[k] == pytest.approx(by_ts[s.as_of])  # nur Kerzen bis as_of
        assert all(s.actual == "UP" for s in samples)
