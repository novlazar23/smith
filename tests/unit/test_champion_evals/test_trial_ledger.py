"""Tests für den Trial-Ledger: kumulativer Trial-Count + steigende Hurdle.

Beide Evolutionsstufen prüfen Kandidaten gegen dieselbe rollierende
OOS-Datenbasis — der Ledger zählt die kumulativ getesteten Kandidaten
pro Stufe, und die Zulassungs-/Promotions-Margin steigt pro Verdopplung
des Suchraums (Multiple-Testing-Korrektur gegen Datenwiederverwendung).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
from apps.champion_evals.agent_evolve import (
    ADMISSION_MARGIN,
    RANDOM_BASELINE_SCORE,
    judge_candidate,
    judge_retention,
)
from apps.champion_evals.agent_params import AGENT_TYPES
from apps.champion_evals.evolve import PROMOTION_MARGIN
from apps.champion_evals.score import AgentMetrics
from apps.champion_evals.trial_ledger import (
    TRIALS_FILENAME_CHAMPION,
    TRIALS_FILENAME_EVOLVED,
    admission_margin,
    load_trial_count,
    record_trial_count,
    stage1_batch_size,
)


def _m(oos_brier: float, cal_stab: float = 0.5, oos_stab: float = 0.5, marginal: float = 0.05) -> AgentMetrics:
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


class TestAdmissionMargin:
    """margin = base + per_doubling x max(0, log2(max(1, trials)))."""

    def test_first_trials_no_correction(self) -> None:
        """trials <= 1 -> exakt base (der erste Kandidat zahlt keine Korrektur)."""
        assert admission_margin(0, ADMISSION_MARGIN) == ADMISSION_MARGIN
        assert admission_margin(1, ADMISSION_MARGIN) == ADMISSION_MARGIN

    def test_per_doubling(self) -> None:
        """Pro Verdopplung des Suchraums steigt die Hurdle um 0,005."""
        assert admission_margin(2, ADMISSION_MARGIN) == pytest.approx(ADMISSION_MARGIN + 0.005)
        assert admission_margin(4, ADMISSION_MARGIN) == pytest.approx(ADMISSION_MARGIN + 0.01)
        assert admission_margin(8, ADMISSION_MARGIN) == pytest.approx(ADMISSION_MARGIN + 0.015)

    def test_promotion_margin_base(self) -> None:
        """Gilt auch für die Stufe-1-Basis-Margin (0,005)."""
        assert admission_margin(2, PROMOTION_MARGIN) == pytest.approx(PROMOTION_MARGIN + 0.005)

    def test_monotone(self) -> None:
        """Nicht-fallend über 0..1024 Trials."""
        values = [admission_margin(n, ADMISSION_MARGIN) for n in range(1025)]
        assert all(after >= before for before, after in itertools.pairwise(values))

    def test_custom_per_doubling(self) -> None:
        assert admission_margin(8, 0.1, per_doubling=0.01) == pytest.approx(0.13)


class TestTrialCountPersistence:
    """load/record: fail-soft bei defekten Ledgern, atomare Schreibweise."""

    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        assert load_trial_count(tmp_path / TRIALS_FILENAME_EVOLVED) == 0

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / TRIALS_FILENAME_CHAMPION
        record_trial_count(path, 32)
        assert load_trial_count(path) == 32

    def test_record_payload(self, tmp_path: Path) -> None:
        path = tmp_path / TRIALS_FILENAME_CHAMPION
        record_trial_count(path, 7)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["trials"] == 7
        assert "updated_at" in data

    def test_corrupt_json_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / TRIALS_FILENAME_EVOLVED
        path.write_text("{defekt", encoding="utf-8")
        assert load_trial_count(path) == 0

    def test_non_object_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / TRIALS_FILENAME_EVOLVED
        path.write_text("[1, 2]", encoding="utf-8")
        assert load_trial_count(path) == 0

    def test_missing_field_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / TRIALS_FILENAME_EVOLVED
        path.write_text('{"updated_at": "2026-01-01"}', encoding="utf-8")
        assert load_trial_count(path) == 0

    def test_negative_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / TRIALS_FILENAME_EVOLVED
        path.write_text('{"trials": -3}', encoding="utf-8")
        assert load_trial_count(path) == 0

    def test_bool_is_not_a_count(self, tmp_path: Path) -> None:
        """bool ist ein int-Subtyp, gilt aber nicht als Zähler."""
        path = tmp_path / TRIALS_FILENAME_EVOLVED
        path.write_text('{"trials": true}', encoding="utf-8")
        assert load_trial_count(path) == 0


class TestStage1BatchSize:
    """Batch = len(AGENT_TYPES) x variants (pro Familie pro Variante ein Trial)."""

    def test_batch_is_families_times_variants(self) -> None:
        assert stage1_batch_size(8) == len(AGENT_TYPES) * 8
        assert stage1_batch_size(8) == 32


class TestMarginThreading:
    """Die angehobene Hurdle verwirft Grenzkandidaten, die die Basis schaffen."""

    def test_candidate_between_base_and_raised_margin(self) -> None:
        """Score = Basis + 0,025: mit Basis-Margin (0,02) zugelassen,
        mit trials=8 (Hurdle +0,015) verworfen."""
        metrics = _m(oos_brier=1.0 - (RANDOM_BASELINE_SCORE + 0.025))
        assert judge_candidate("a", metrics, promotion_margin=ADMISSION_MARGIN).admitted
        raised = admission_margin(8, base=ADMISSION_MARGIN)
        assert raised == pytest.approx(ADMISSION_MARGIN + 0.015)
        verdict = judge_candidate("a", metrics, promotion_margin=raised)
        assert not verdict.admitted
        assert any("Zufalls-Basis" in reason for reason in verdict.reasons)

    def test_retention_uses_no_margin(self) -> None:
        """Retention (Bestand) prüft ohne Zulassungs-Margin — unverändert."""
        metrics = _m(oos_brier=1.0 - (RANDOM_BASELINE_SCORE + 0.001))
        assert judge_retention("a", metrics).admitted
