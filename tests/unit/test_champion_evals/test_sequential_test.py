"""Tests für den Shadow-Sequenztest: Statistik-Kern, Score-Deltas, beide Wirings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from apps.champion_evals import __main__ as cm
from apps.champion_evals.score import EvalSample, oos_score_deltas, oos_score_vs_random_base
from apps.champion_evals.sequential_test import (
    SHADOW_LOG_FILENAME,
    append_jsonl,
    bh_reject,
    daily_lag,
    holm_reject,
    newey_west_factor,
    one_sided_z_pvalue,
)


class TestNeweyWestFactor:
    def test_lag_zero_is_naive(self) -> None:
        assert newey_west_factor([1.0, -1.0, 2.0, -2.0], lag=0) == 1.0

    def test_small_sample_is_naive(self) -> None:
        assert newey_west_factor([1.0, -1.0], lag=3) == 1.0

    def test_constant_series_is_naive(self) -> None:
        assert newey_west_factor([0.5] * 10, lag=2) == 1.0

    def test_positive_autocorrelation_inflates(self) -> None:
        assert newey_west_factor(list(range(1, 30)), lag=1) > 1.0

    def test_negative_autocorrelation_capped_at_one(self) -> None:
        assert newey_west_factor([1.0, -1.0] * 10, lag=1) == 1.0


class TestOneSidedZPvalue:
    def test_empty_sample_is_one(self) -> None:
        assert one_sided_z_pvalue([], 0.005) == 1.0

    def test_single_sample_above_effect(self) -> None:
        assert one_sided_z_pvalue([0.01], 0.005) == 0.0

    def test_single_sample_below_effect(self) -> None:
        assert one_sided_z_pvalue([0.001], 0.005) == 1.0

    def test_constant_above_effect_is_degenerate_reject(self) -> None:
        assert one_sided_z_pvalue([0.01] * 100, 0.005) == 0.0

    def test_constant_below_effect_is_degenerate_keep(self) -> None:
        assert one_sided_z_pvalue([0.001] * 100, 0.005) == 1.0

    def test_clear_effect_is_significant(self) -> None:
        diffs = [0.02 + (0.01 if i % 2 else -0.01) for i in range(100)]
        assert one_sided_z_pvalue(diffs, 0.005, nw_lag=1) < 1e-6

    def test_null_effect_is_about_half(self) -> None:
        diffs = [0.005 + (0.01 if i % 2 else -0.01) for i in range(100)]
        assert 0.3 < one_sided_z_pvalue(diffs, 0.005, nw_lag=1) < 0.7


class TestBhHolm:
    def test_bh_rejects_top_rank_prefix(self) -> None:
        assert bh_reject([0.001, 0.02, 0.5], q=0.05) == frozenset({0, 1})

    def test_bh_rejects_nothing(self) -> None:
        assert bh_reject([0.1, 0.2], q=0.05) == frozenset()

    def test_bh_empty(self) -> None:
        assert bh_reject([], q=0.05) == frozenset()

    def test_holm_stops_at_first_failure(self) -> None:
        assert holm_reject([0.001, 0.04, 0.5], alpha=0.05) == frozenset({0})

    def test_holm_rejects_all(self) -> None:
        assert holm_reject([0.001, 0.002], alpha=0.05) == frozenset({0, 1})

    def test_holm_empty(self) -> None:
        assert holm_reject([], alpha=0.05) == frozenset()


class TestDailyLag:
    def test_single_point_is_one(self) -> None:
        assert daily_lag([datetime(2026, 1, 1, tzinfo=UTC)]) == 1

    def test_samples_per_day(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        as_ofs = [base + timedelta(hours=48 * i / 99) for i in range(100)]
        assert daily_lag(as_ofs) == 50


class TestAppendJsonl:
    def test_appends_json_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "shadow.jsonl"
        append_jsonl(path, {"a": 1})
        append_jsonl(path, {"b": "x"})
        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": "x"}]


class TestScoreDeltas:
    @staticmethod
    def _samples() -> list[EvalSample]:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        uniform = {"UP": 1 / 3, "DOWN": 1 / 3, "RANGE": 1 / 3}
        perfect = {"UP": 1.0, "DOWN": 0.0, "RANGE": 0.0}
        return [
            EvalSample(base, {"cand": perfect, "champ": uniform}, "UP", 0.01),
            EvalSample(base + timedelta(hours=1), {"cand": perfect, "champ": uniform}, "UP", 0.01),
            EvalSample(base + timedelta(hours=2), {"cand": perfect, "champ": uniform}, "UP", 0.01),
            EvalSample(base + timedelta(hours=3), {"champ": uniform}, "UP", 0.01),
        ]

    def test_oos_only_and_paired(self) -> None:
        samples = self._samples()
        deltas = oos_score_deltas(samples, "cand", "champ")
        assert deltas == [(samples[2].as_of, pytest.approx(2 / 3))]


class TestVsRandomBase:
    def test_uniform_agent_is_zero(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        uniform = {"UP": 1 / 3, "DOWN": 1 / 3, "RANGE": 1 / 3}
        samples = [EvalSample(base + timedelta(hours=i), {"agent": uniform}, "UP", 0.01) for i in range(4)]
        deltas = oos_score_vs_random_base(samples, "agent")
        assert len(deltas) == 2
        assert all(delta == pytest.approx(0.0) for _, delta in deltas)

    def test_perfect_agent_is_two_thirds(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        perfect = {"UP": 1.0, "DOWN": 0.0, "RANGE": 0.0}
        samples = [EvalSample(base + timedelta(hours=i), {"agent": perfect}, "UP", 0.01) for i in range(4)]
        deltas = oos_score_vs_random_base(samples, "agent")
        assert all(delta == pytest.approx(2 / 3) for _, delta in deltas)


class TestStage1Wiring:
    def test_run_evolve_writes_shadow_log_and_keeps_selection(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        uniform = {"UP": 0.4, "DOWN": 0.3, "RANGE": 0.3}

        def _fake_replay(candles: object, instances: Mapping[str, object], **_kwargs: Any) -> list[EvalSample]:
            return [
                EvalSample(
                    base + timedelta(hours=i),
                    {name: dict(uniform) for name in instances},
                    "UP",
                    0.01,
                )
                for i in range(40)
            ]

        monkeypatch.setattr("apps.champion_evals.score.replay_instances", _fake_replay)
        args = SimpleNamespace(
            output=str(tmp_path / "champion_evals.json"),
            configs_output=str(tmp_path / "champion_configs.json"),
            variants=2,
            promotion_margin=0.99,
            seed=42,
            horizon="15m",
            resample="5m",
            instrument="BTC/USDT",
            up_threshold=0.01,
            down_threshold=-0.01,
            calibration_ratio=0.5,
            min_samples=10,
            candle_limit=200,
            min_candles=30,
            evaluate_every=12,
            horizon_bars=3,
            version="current",
        )
        assert cm._run_evolve(args, [("BTC/USDT", [])]) == 0

        shadow_path = tmp_path / SHADOW_LOG_FILENAME
        assert shadow_path.exists()
        entry = json.loads(shadow_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert entry["n_candidates"] == 8
        assert entry["n_bh_rejected"] == 0
        assert entry["min_effect"] == pytest.approx(0.005)
        assert entry["q"] == pytest.approx(0.05)
        # OOS-Fenster 19 h < 1 Tag → Span wird auf 1.0 geklemmt → Lag = alle 20 Samples
        assert entry["nw_lag"] == 20
        assert len(entry["families"]) == 4
        for family in entry["families"]:
            assert len(family["variants"]) == 2
            for variant in family["variants"]:
                assert variant["n"] == 20
                assert variant["mean_diff"] == pytest.approx(0.0)
                assert variant["p"] == 1.0
                assert variant["bh_rejected"] is False
        assert (tmp_path / "champion_configs.json").exists()
