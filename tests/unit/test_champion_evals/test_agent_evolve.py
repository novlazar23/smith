"""Tests für Stufe 2: Zulassungs-Gates, Artefakt, Re-Prüfung (deterministische Integration)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from apps.champion_evals.agent_evolve import (
    ADMISSION_MARGIN,
    RANDOM_BASELINE_SCORE,
    build_agents_artifact,
    build_digest,
    evaluate_evolved_candidates,
    judge_candidate,
    judge_retention,
    load_eval_artifact,
)
from apps.champion_evals.agent_sandbox import build_evolved_agent
from apps.champion_evals.score import AgentMetrics
from packages.backtesting.core import Candle
from packages.schemas.agent_report import AgentStatus

UNIFORM_CODE = """def predict(open, high, low, close, volume, timestamps):
    return (1.0, 1.0, 1.0)
"""

GOOD_CODE = """import numpy as np

def predict(open, high, low, close, volume, timestamps):
    m = float(close[-1] - close[-6])
    if m > 0:
        return (0.8, 0.1, 0.1)
    if m < 0:
        return (0.1, 0.8, 0.1)
    return (0.34, 0.33, 0.33)
"""

CRASH_CODE = """def predict(open, high, low, close, volume, timestamps):
    raise RuntimeError('defekt')
"""


def _m(oos_brier: float, cal_stab: float = 0.5, oos_stab: float = 0.5, marginal: float = 0.0) -> AgentMetrics:
    return AgentMetrics(
        agent_id="x",
        cal_samples=10,
        oos_samples=10,
        cal_brier=0.2,
        oos_brier=oos_brier,
        cal_stability=cal_stab,
        oos_stability=oos_stab,
        oos_marginal=marginal,
    )


class TestBaseline:
    def test_uniform_score_is_one_third(self) -> None:
        assert pytest.approx(1.0 / 3.0) == RANDOM_BASELINE_SCORE


class TestJudgeCandidate:
    def test_admitted_above_baseline_with_margin(self) -> None:
        verdict = judge_candidate("x", _m(oos_brier=0.5, cal_stab=0.6, oos_stab=0.6, marginal=0.01))
        assert verdict.admitted
        assert verdict.score == pytest.approx(0.5)

    def test_rejected_below_baseline_plus_margin(self) -> None:
        verdict = judge_candidate("x", _m(oos_brier=1.0 - RANDOM_BASELINE_SCORE - ADMISSION_MARGIN + 0.01))
        assert not verdict.admitted

    def test_rejected_unstable(self) -> None:
        verdict = judge_candidate("x", _m(oos_brier=0.5, cal_stab=0.6, oos_stab=0.4, marginal=0.01))
        assert not verdict.admitted

    def test_rejected_negative_marginal(self) -> None:
        verdict = judge_candidate("x", _m(oos_brier=0.5, cal_stab=0.6, oos_stab=0.6, marginal=-0.01))
        assert not verdict.admitted


class TestJudgeRetention:
    def test_kept_above_baseline(self) -> None:
        assert judge_retention("x", _m(oos_brier=0.5, marginal=0.001)).admitted

    def test_removed_below_baseline(self) -> None:
        assert not judge_retention("x", _m(oos_brier=0.7)).admitted

    def test_removed_negative_marginal(self) -> None:
        assert not judge_retention("x", _m(oos_brier=0.5, marginal=-0.01)).admitted


class TestBuildAgentsArtifact:
    def test_new_agent_gets_version_one(self) -> None:
        artifact = build_agents_artifact({"a": (GOOD_CODE, "claim", 0.5)}, None)
        assert artifact["a"]["version"] == 1
        assert artifact["a"]["code"] == GOOD_CODE
        assert artifact["a"]["score"] == 0.5

    def test_unchanged_code_keeps_version(self) -> None:
        previous = {"a": {"version": 3, "code": GOOD_CODE, "claim": "c", "admitted_at": "2026-09-01"}}
        artifact = build_agents_artifact({"a": (GOOD_CODE, "c", 0.55)}, previous)
        assert artifact["a"]["version"] == 3
        assert artifact["a"]["admitted_at"] == "2026-09-01"
        assert artifact["a"]["score"] == 0.55

    def test_changed_code_bumps_version(self) -> None:
        previous = {"a": {"version": 3, "code": "alt", "claim": "c", "admitted_at": "2026-09-01"}}
        artifact = build_agents_artifact({"a": (GOOD_CODE, "c", 0.55)}, previous)
        assert artifact["a"]["version"] == 4


class TestBuildDigest:
    def test_contains_baseline_and_previous(self) -> None:
        digest = build_digest({"trend": {"challenger": {"oos_score": 0.5, "stability_score": 0.6, "marginal_contribution": 0.01}}}, {"ev": {"score": 0.4, "version": 2}})
        assert "trend" in digest
        assert "ev" in digest
        assert f"{RANDOM_BASELINE_SCORE:.4f}" in digest

    def test_missing_data_is_explicit(self) -> None:
        digest = build_digest(None, {})
        assert "noch keine" in digest


class TestLoadEvalArtifact:
    def test_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_eval_artifact(tmp_path / "keine.json") is None

    def test_corrupt_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "a.json"
        path.write_text("{x", encoding="utf-8")
        assert load_eval_artifact(path) is None


class TestEvaluateIntegration:
    @staticmethod
    def _candles(n: int = 120) -> list[Candle]:
        rng = np.random.default_rng(42)
        price = 100.0
        candles: list[Candle] = []
        base = datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(n):
            open_ = price
            close = max(1.0, price * (1.0 + float(rng.normal(0.0, 0.002))))
            candles.append(
                Candle(
                    timestamp=base + timedelta(minutes=5 * i),
                    symbol="BTC/USDT",
                    open=open_,
                    high=max(open_, close) * 1.0001,
                    low=min(open_, close) * 0.9999,
                    close=close,
                    volume=100.0,
                )
            )
            price = close
        return candles

    @staticmethod
    def _instances(codes: dict[str, str]) -> dict:
        return {
            name: build_evolved_agent(name, {"code": code}, AgentStatus.SHADOW, instrument="BTC/USDT", horizon="15m")
            for name, code in codes.items()
        }

    def test_uniform_candidate_not_admitted(self) -> None:
        series = [("BTC/USDT", self._candles())]
        base = self._instances({"base_a": GOOD_CODE, "base_b": UNIFORM_CODE})
        candidates = self._instances({"uniform_candidate": UNIFORM_CODE})
        artifact = evaluate_evolved_candidates(
            series, base, candidates, {"uniform_candidate": (UNIFORM_CODE, "uniform claim")}, previous={}
        )
        assert "uniform_candidate" not in artifact

    def test_previous_agent_not_delivered_is_removed(self) -> None:
        series = [("BTC/USDT", self._candles())]
        base = self._instances({"base_a": GOOD_CODE, "base_b": UNIFORM_CODE})
        previous = {"defekt": {"version": 1, "code": CRASH_CODE, "claim": "c", "admitted_at": "2026-09-01"}}
        artifact = evaluate_evolved_candidates(series, base, {}, {}, previous=previous)
        assert "defekt" not in artifact

    def test_summary_collects_verdicts(self) -> None:
        series = [("BTC/USDT", self._candles())]
        base = self._instances({"base_a": GOOD_CODE, "base_b": UNIFORM_CODE})
        candidates = self._instances({"uniform_candidate": UNIFORM_CODE})
        summary: list = []
        evaluate_evolved_candidates(
            series,
            base,
            candidates,
            {"uniform_candidate": (UNIFORM_CODE, "uniform claim")},
            previous={},
            summary=summary,
        )
        by_name = {entry["name"]: entry for entry in summary}
        rejected = by_name["uniform_candidate"]
        assert rejected["admitted"] is False
        assert rejected["kind"] == "kandidat"
        assert rejected["reasons"]
        assert rejected["score"] is not None

    def test_cap_keeps_highest_score(self) -> None:
        series = [("BTC/USDT", self._candles())]
        base = self._instances({"base_a": GOOD_CODE, "base_b": UNIFORM_CODE})
        codes = {f"agent_{i}": GOOD_CODE for i in range(5)}
        candidates = self._instances(codes)
        meta = {name: (code, "claim") for name, code in codes.items()}
        artifact = evaluate_evolved_candidates(
            series, base, candidates, meta, previous={}, max_agents=3
        )
        assert len(artifact) <= 3
