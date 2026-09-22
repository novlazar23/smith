"""Tests für Stufe 2: Zulassungs-Gates, Artefakt, Re-Prüfung (deterministische Integration)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from apps.champion_evals.agent_evolve import (
    ADMISSION_MARGIN,
    EVOLVED_AGENTS_LAST_RUN_FILENAME,
    RANDOM_BASELINE_SCORE,
    build_agents_artifact,
    build_digest,
    evaluate_evolved_candidates,
    judge_candidate,
    judge_retention,
    load_eval_artifact,
    run_agent_evolution,
)
from apps.champion_evals.agent_sandbox import build_evolved_agent
from apps.champion_evals.candidate_archive import (
    ARCHIVE_FILENAME,
    code_hash,
    known_code_hashes,
    load_archive,
    record_candidates,
)
from apps.champion_evals.proposer import PERSONAS
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


class TestPrepareCandidates:
    def test_rejections_are_recorded_in_summary(self) -> None:
        from apps.champion_evals.agent_evolve import prepare_candidates

        proposals = [
            {"name": "good_one", "code": GOOD_CODE, "claim": "c"},
            {"name": "bad_jail", "code": "import os\ndef predict(o,h,l,c,v,t): return (1,0,0)", "claim": "c"},
            {"name": "bad_smoke", "code": CRASH_CODE, "claim": "c"},
        ]
        summary: list[dict[str, Any]] = []
        instances, _, _ = prepare_candidates(proposals, summary=summary)
        assert set(instances) == {"good_one"}
        by_name = {entry["name"]: entry for entry in summary}
        assert set(by_name) == {"bad_jail", "bad_smoke"}
        assert all(
            entry["admitted"] is False and entry["kind"] == "kandidat" and entry["score"] is None for entry in summary
        )
        assert any("Jail" in reason for reason in by_name["bad_jail"]["reasons"])
        assert any("Smoke-Test" in reason for reason in by_name["bad_smoke"]["reasons"])


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

    def test_summary_has_shadow_fields(self) -> None:
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
        uniform = by_name["uniform_candidate"]
        assert uniform["shadow_p"] == 1.0
        assert uniform["shadow_holm_rejected"] is False

    def test_shadow_p_non_degenerate_path(self) -> None:
        series = [("BTC/USDT", self._candles())]
        base = self._instances({"base_a": GOOD_CODE, "base_b": UNIFORM_CODE})
        candidates = self._instances({"good_candidate": GOOD_CODE})
        summary: list = []
        evaluate_evolved_candidates(
            series,
            base,
            candidates,
            {"good_candidate": (GOOD_CODE, "good claim")},
            previous={},
            summary=summary,
        )
        entry = next(e for e in summary if e["name"] == "good_candidate")
        assert 0.0 <= entry["shadow_p"] <= 1.0
        assert isinstance(entry["shadow_holm_rejected"], bool)


# LLM-Vorschlags-Code (≥ 100 Zeichen für die Proposer-Validierung;
# uniform → deterministisch abgelehnt, Jail-passend).
UNIFORM_LLM_CODE = """def predict(open, high, low, close, volume, timestamps):
    # Konstant uniforme Wahrscheinlichkeiten: triviale Logik ohne
    # Kursrichtung — deterministischer Testkandidat für die Gates.
    return (1.0, 1.0, 1.0)
"""
UNIFORM_LLM_CLAIM = "Konstant uniform ohne Information (Testkandidat)."


class _StubLLMClient:
    """LLM-Client-Doppel: jeder Aufruf liefert dieselbe Antwort."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.timeout = 60.0

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        return self.answer


def _stage2_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output=str(tmp_path / "champion_evals.json"),
        agents_output=str(tmp_path / "evolved_agents.json"),
        configs_output=str(tmp_path / "champion_configs.json"),
        horizon="15m",
        up_threshold=0.01,
        down_threshold=-0.01,
        candle_limit=200,
        min_candles=30,
        evaluate_every=5,
        horizon_bars=3,
        calibration_ratio=0.5,
        max_evolved=3,
        evolve_agents=2,
        candidate_file=None,
        candidate_name=None,
        candidate_claim=None,
        llm_model=None,
    )


def _llm_answer() -> str:
    return json.dumps(
        [{"name": "uniform_candidate", "claim": UNIFORM_LLM_CLAIM, "code": UNIFORM_LLM_CODE}]
    )


class TestRunAgentEvolutionArchive:
    """Kandidaten-Archiv im kompletten Lauf: Code-Hash-Dedup + Einträge."""

    def test_archived_code_is_not_retested(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        args = _stage2_args(tmp_path)
        # Archiv enthält denselben Code bereits (unter anderem Namen).
        record_candidates(
            tmp_path / ARCHIVE_FILENAME,
            [
                {
                    "run_at": "2026-09-21T04:00:00",
                    "name": "old_name_same_code",
                    "persona": "trend",
                    "claim": UNIFORM_LLM_CLAIM,
                    "code_hash": code_hash(UNIFORM_LLM_CODE),
                    "admitted": False,
                    "score": 1.0 / 3.0,
                    "oos_brier": 2.0 / 3.0,
                    "reasons": ["OOS-Score unter der Hurdle"],
                    "shadow_p": 1.0,
                    "shadow_holm_rejected": False,
                    "effective_margin": ADMISSION_MARGIN,
                }
            ],
        )
        client = _StubLLMClient(_llm_answer())
        rc = run_agent_evolution(
            args=args,
            series=[("BTC/USDT", TestEvaluateIntegration._candles())],
            llm_client_factory=lambda: client,
        )
        assert rc == 0
        last_run = json.loads((tmp_path / EVOLVED_AGENTS_LAST_RUN_FILENAME).read_text(encoding="utf-8"))
        assert last_run["candidates"] == []
        assert [entry["name"] for entry in load_archive(tmp_path / ARCHIVE_FILENAME)] == ["old_name_same_code"]
        assert any("bereits im Archiv" in record.message for record in caplog.records)

    def test_candidate_file_archived_code_is_not_retested(self, tmp_path: Path) -> None:
        args = _stage2_args(tmp_path)
        args.evolve_agents = 0
        code_file = tmp_path / "uniform_candidate.py"
        code_file.write_text(UNIFORM_LLM_CODE, encoding="utf-8")
        args.candidate_file = str(code_file)
        args.candidate_name = "uniform_candidate"
        args.candidate_claim = UNIFORM_LLM_CLAIM
        record_candidates(
            tmp_path / ARCHIVE_FILENAME,
            [{"name": "old", "code_hash": code_hash(UNIFORM_LLM_CODE), "admitted": False}],
        )
        rc = run_agent_evolution(args=args, series=[("BTC/USDT", TestEvaluateIntegration._candles())])
        assert rc == 0
        last_run = json.loads((tmp_path / EVOLVED_AGENTS_LAST_RUN_FILENAME).read_text(encoding="utf-8"))
        assert last_run["candidates"] == []

    def test_evaluated_candidate_is_archived(self, tmp_path: Path) -> None:
        args = _stage2_args(tmp_path)
        client = _StubLLMClient(_llm_answer())
        rc = run_agent_evolution(
            args=args,
            series=[("BTC/USDT", TestEvaluateIntegration._candles())],
            llm_client_factory=lambda: client,
        )
        assert rc == 0
        entries = load_archive(tmp_path / ARCHIVE_FILENAME)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["name"] == "uniform_candidate"
        assert entry["persona"] == PERSONAS[0][0]
        assert entry["claim"] == UNIFORM_LLM_CLAIM
        assert entry["code_hash"] == code_hash(UNIFORM_LLM_CODE)
        assert entry["admitted"] is False
        assert entry["score"] is not None
        assert entry["oos_brier"] == pytest.approx(1.0 - entry["score"])
        assert entry["reasons"]
        assert entry["shadow_p"] is not None
        assert isinstance(entry["shadow_holm_rejected"], bool)
        assert entry["effective_margin"] >= ADMISSION_MARGIN
        # run_at identisch mit dem Letzter-Lauf-Artefakt; Code im Hash-Satz.
        last_run = json.loads((tmp_path / EVOLVED_AGENTS_LAST_RUN_FILENAME).read_text(encoding="utf-8"))
        assert entry["run_at"] == last_run["run_at"]
        assert code_hash(UNIFORM_LLM_CODE) in known_code_hashes(tmp_path / ARCHIVE_FILENAME)

    def test_candidate_file_archived_with_none_persona(self, tmp_path: Path) -> None:
        args = _stage2_args(tmp_path)
        args.evolve_agents = 0
        code_file = tmp_path / "uniform_candidate.py"
        code_file.write_text(UNIFORM_LLM_CODE, encoding="utf-8")
        args.candidate_file = str(code_file)
        args.candidate_name = "uniform_candidate"
        args.candidate_claim = UNIFORM_LLM_CLAIM
        rc = run_agent_evolution(args=args, series=[("BTC/USDT", TestEvaluateIntegration._candles())])
        assert rc == 0
        entries = load_archive(tmp_path / ARCHIVE_FILENAME)
        assert len(entries) == 1
        assert entries[0]["name"] == "uniform_candidate"
        assert entries[0]["persona"] is None
        assert entries[0]["admitted"] is False
