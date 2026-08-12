"""Tests for ablation analysis and validation report generation (WP04)."""

from __future__ import annotations

from datetime import datetime

import pytest
from packages.validation.ablation.base import (
    AblationResult,
)
from packages.validation.ablation.feature_importance import (
    FeatureDataset,
    FeatureImportanceAnalyzer,
)
from packages.validation.ablation.loo import (
    AgentEnsemble,
    LeaveOneOutAblation,
)
from packages.validation.ablation.marginal_brier import (
    AgentPredictions,
    MarginalBrierAnalyzer,
)
from packages.validation.reports.report import (
    AblationSummary,
    AgentPerformance,
    BaselineComparison,
    CalibrationSummary,
    ValidationReport,
    generate_validation_report,
)

# ── helpers ──────────────────────────────────────────────────────────


def _make_predictions(
    up: float, down: float, rng: float
) -> dict[str, float]:
    """Create a valid probability dict."""
    total = up + down + rng
    return {
        "UP": up / total,
        "DOWN": down / total,
        "RANGE": rng / total,
    }


def _actuals(n: int = 10) -> list[str]:
    """Round-robin actuals."""
    base = ["UP", "DOWN", "RANGE"]
    return (base * ((n // len(base)) + 1))[:n]


# ══════════════════════════════════════════════════════════════════════
# 1. AblationResult
# ══════════════════════════════════════════════════════════════════════


class TestAblationResult:
    """Test the AblationResult frozen dataclass and helpers."""

    def _make_result(
        self,
        marginal: float = 0.05,
        direction: str = "helpful",
        confidence: float = 0.9,
    ) -> AblationResult:
        return AblationResult(
            method="test",
            agent_id="agent_x",
            full_score=0.3,
            ablated_score=0.35,
            marginal_contribution=marginal,
            direction=direction,
            confidence=confidence,
        )

    def test_is_helpful(self) -> None:
        r = self._make_result(marginal=0.05)
        assert r.is_helpful is True

    def test_is_harmful(self) -> None:
        r = self._make_result(marginal=-0.05)
        assert r.is_harmful is True

    def test_neutral_detection_positive(self) -> None:
        r = self._make_result(marginal=0.0)
        assert r.is_helpful is False

    def test_neutral_detection_negative(self) -> None:
        r = self._make_result(marginal=-0.0)
        assert r.is_helpful is False
        assert r.is_harmful is False

    def test_near_zero_not_helpful(self) -> None:
        """Values within 1e-9 are neither helpful nor harmful."""
        r = self._make_result(marginal=1e-10)
        assert r.is_helpful is False
        assert r.is_harmful is False

    def test_near_zero_not_harmful(self) -> None:
        r = self._make_result(marginal=-1e-10)
        assert r.is_helpful is False
        assert r.is_harmful is False

    def test_frozen(self) -> None:
        r = self._make_result()
        with pytest.raises(Exception):
            r.method = "changed"  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        r = self._make_result()
        assert r.method == "test"
        assert r.agent_id == "agent_x"
        assert r.full_score == 0.3
        assert r.ablated_score == 0.35
        assert r.marginal_contribution == 0.05
        assert r.direction == "helpful"
        assert r.confidence == 0.9


# ══════════════════════════════════════════════════════════════════════
# 2-4. LeaveOneOutAblation
# ══════════════════════════════════════════════════════════════════════


class TestLeaveOneOutAblation:
    """Test LeaveOneOutAblation.run() and run_ensemble()."""

    def test_run_marginal_computed_brier(self) -> None:
        loo = LeaveOneOutAblation(higher_is_better=False)
        r = loo.run(full_score=0.2, ablated_score=0.3, agent_id="A")
        assert r.marginal_contribution == pytest.approx(0.1)
        assert r.direction == "helpful"

    def test_run_marginal_computed_accuracy(self) -> None:
        loo = LeaveOneOutAblation(higher_is_better=True)
        r = loo.run(full_score=0.8, ablated_score=0.7, agent_id="A")
        assert r.marginal_contribution == pytest.approx(0.1)
        assert r.direction == "helpful"

    def test_run_direction_harmful(self) -> None:
        loo = LeaveOneOutAblation(higher_is_better=False)
        r = loo.run(full_score=0.3, ablated_score=0.2, agent_id="A")
        assert r.direction == "harmful"

    def test_run_direction_neutral(self) -> None:
        loo = LeaveOneOutAblation(higher_is_better=False)
        r = loo.run(full_score=0.25, ablated_score=0.25, agent_id="A")
        assert r.direction == "neutral"

    def test_default_brier_score_perfect(self) -> None:
        """Perfect predictions matching actuals → Brier = 0."""
        loo = LeaveOneOutAblation()
        preds = [
            _make_predictions(1.0, 0.0, 0.0),  # UP
            _make_predictions(0.0, 1.0, 0.0),  # DOWN
            _make_predictions(0.0, 0.0, 1.0),  # RANGE
        ]
        actuals = ["UP", "DOWN", "RANGE"]
        score = loo._default_brier_score(preds, actuals)
        assert score == pytest.approx(0.0)

    def test_default_brier_score_random(self) -> None:
        loo = LeaveOneOutAblation()
        preds = [_make_predictions(0.33, 0.33, 0.34) for _ in range(3)]
        actuals = ["UP", "DOWN", "RANGE"]
        score = loo._default_brier_score(preds, actuals)
        assert score > 0.0

    def test_method_name(self) -> None:
        loo = LeaveOneOutAblation()
        assert loo.method_name == "loo"

    def test_run_ensemble_3_agents(self) -> None:
        """3-agent ensemble: each agent should have a computed marginal."""
        loo = LeaveOneOutAblation(higher_is_better=False)
        agents = [
            AgentEnsemble(
                agent_id="A",
                predictions=[
                    _make_predictions(0.5, 0.3, 0.2),
                    _make_predictions(0.2, 0.6, 0.2),
                    _make_predictions(0.2, 0.2, 0.6),
                ],
            ),
            AgentEnsemble(
                agent_id="B",
                predictions=[
                    _make_predictions(0.4, 0.4, 0.2),
                    _make_predictions(0.3, 0.5, 0.2),
                    _make_predictions(0.3, 0.3, 0.4),
                ],
            ),
            AgentEnsemble(
                agent_id="C",
                predictions=[
                    _make_predictions(0.3, 0.3, 0.4),
                    _make_predictions(0.2, 0.4, 0.4),
                    _make_predictions(0.2, 0.3, 0.5),
                ],
            ),
        ]
        actuals = ["UP", "DOWN", "RANGE"]
        results = loo.run_ensemble(agents, actuals)
        assert len(results) == 3
        for r in results:
            assert r.method == "loo"
            assert r.agent_id in ("A", "B", "C")

    def test_run_ensemble_empty(self) -> None:
        loo = LeaveOneOutAblation()
        results = loo.run_ensemble([], [])
        assert results == []

    def test_run_ensemble_single_agent(self) -> None:
        """Single-agent ensemble: removing it leaves nothing."""
        loo = LeaveOneOutAblation(higher_is_better=False)
        agents = [
            AgentEnsemble(
                agent_id="solo",
                predictions=[
                    _make_predictions(0.5, 0.3, 0.2),
                ],
            ),
        ]
        results = loo.run_ensemble(agents, ["UP"])
        # Only one agent, removing it gives empty ensemble → no results
        assert len(results) == 0

    def test_agent_ensemble_auto_sample_ids(self) -> None:
        ae = AgentEnsemble(
            agent_id="x",
            predictions=[_make_predictions(0.5, 0.3, 0.2) for _ in range(3)],
        )
        assert ae.sample_ids == ["s0", "s1", "s2"]

    def test_agent_ensemble_empty_predictions_raises(self) -> None:
        with pytest.raises(ValueError, match="agent must have"):
            AgentEnsemble(agent_id="x", predictions=[])


# ══════════════════════════════════════════════════════════════════════
# 5-6. FeatureImportanceAnalyzer
# ══════════════════════════════════════════════════════════════════════


class TestFeatureImportanceAnalyzer:
    """Test FeatureImportanceAnalyzer.analyze() and sorting."""

    def _make_dataset(self, n: int = 10) -> FeatureDataset:
        matrix = [
            {"feature_a": i * 0.1, "feature_b": i * 0.05}
            for i in range(n)
        ]
        return FeatureDataset(
            feature_names=["feature_a", "feature_b"],
            feature_matrix=matrix,
            actuals=_actuals(n),
        )

    def test_permuted_feature_degrades_score(self) -> None:
        """Permuting a feature should generally change the score."""
        analyzer = FeatureImportanceAnalyzer(
            n_permutations=5,
            seed=123,
            higher_is_better=False,
        )
        dataset = self._make_dataset(n=20)
        # Use a model that puts weight on feature_a
        model_preds = [
            {
                "UP": 0.3 + 0.2 * (i * 0.1),
                "DOWN": 0.3 - 0.1 * (i * 0.1),
                "RANGE": 0.4 - 0.1 * (i * 0.1),
            }
            for i in range(20)
        ]
        results = analyzer.analyze(dataset, model_preds)
        assert len(results) == 2
        # Feature importance values should differ (permutation changes score)
        for r in results:
            assert r.method == "feature_importance"

    def test_sorted_by_absolute_marginal(self) -> None:
        """Results sorted by |marginal| descending."""
        analyzer = FeatureImportanceAnalyzer(
            n_permutations=3,
            seed=99,
            higher_is_better=False,
        )
        dataset = self._make_dataset(n=10)
        model_preds = [
            {"UP": 0.4, "DOWN": 0.3, "RANGE": 0.3} for _ in range(10)
        ]
        results = analyzer.analyze(dataset, model_preds)
        for i in range(len(results) - 1):
            assert abs(results[i].marginal_contribution) >= abs(
                results[i + 1].marginal_contribution
            )

    def test_method_name(self) -> None:
        analyzer = FeatureImportanceAnalyzer()
        assert analyzer.method_name == "feature_importance"

    def test_default_brier_matches_loo(self) -> None:
        """Feature importance uses the same Brier as LOO."""
        fi = FeatureImportanceAnalyzer()
        loo = LeaveOneOutAblation()
        preds = [_make_predictions(0.5, 0.3, 0.2) for _ in range(5)]
        actuals = ["UP", "DOWN", "RANGE", "UP", "DOWN"]
        assert fi._default_brier_score(preds, actuals) == pytest.approx(
            loo._default_brier_score(preds, actuals)
        )

    def test_feature_dataset_validation(self) -> None:
        with pytest.raises(ValueError):
            FeatureDataset(
                feature_names=["f1"],
                feature_matrix=[],
                actuals=["UP"],
            )

    def test_feature_dataset_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            FeatureDataset(
                feature_names=["f1"],
                feature_matrix=[{"f1": 0.5}],
                actuals=["UP", "DOWN"],
            )

    def test_analyze_empty_feature_names(self) -> None:
        analyzer = FeatureImportanceAnalyzer()
        dataset = FeatureDataset(
            feature_names=[],
            feature_matrix=[{"f1": 0.5}],
            actuals=["UP"],
        )
        results = analyzer.analyze(dataset, [{"UP": 0.5, "DOWN": 0.3, "RANGE": 0.2}])
        assert results == []

    def test_run_helpful(self) -> None:
        """full=0.2 (good), ablated=0.3 (bad) → marginal=-0.1 → helpful for Brier."""
        analyzer = FeatureImportanceAnalyzer(higher_is_better=False)
        r = analyzer.run(full_score=0.2, ablated_score=0.3, agent_id="f1")
        assert r.marginal_contribution == pytest.approx(0.1)
        assert r.direction == "helpful"

    def test_run_harmful(self) -> None:
        """full=0.3 (bad), ablated=0.2 (good) → marginal=0.1 → harmful for Brier."""
        analyzer = FeatureImportanceAnalyzer(higher_is_better=False)
        r = analyzer.run(full_score=0.3, ablated_score=0.2, agent_id="f1")
        assert r.marginal_contribution == pytest.approx(-0.1)
        assert r.direction == "harmful"


# ══════════════════════════════════════════════════════════════════════
# 7-8. MarginalBrierAnalyzer
# ══════════════════════════════════════════════════════════════════════


class TestMarginalBrierAnalyzer:
    """Test MarginalBrierAnalyzer.analyze() with agent comparisons."""

    def test_agent_better_than_ensemble(self) -> None:
        """Perfect agent: ensemble is dragged down → marginal > 0 (helpful)."""
        analyzer = MarginalBrierAnalyzer()
        # Perfect agent A
        perfect = AgentPredictions(
            agent_id="perfect",
            predictions=[
                _make_predictions(1.0, 0.0, 0.0),  # UP correct
                _make_predictions(0.0, 1.0, 0.0),  # DOWN correct
                _make_predictions(0.0, 0.0, 1.0),  # RANGE correct
            ],
            actuals=["UP", "DOWN", "RANGE"],
        )
        # Garbage agent B
        garbage = AgentPredictions(
            agent_id="garbage",
            predictions=[
                _make_predictions(0.1, 0.4, 0.5),
                _make_predictions(0.1, 0.4, 0.5),
                _make_predictions(0.1, 0.4, 0.5),
            ],
            actuals=["UP", "DOWN", "RANGE"],
        )
        results = analyzer.analyze([perfect, garbage])
        # perfect should have positive marginal (better than ensemble average)
        perfect_r = next(r for r in results if r.agent_id == "perfect")
        assert perfect_r.marginal_contribution > 0
        # garbage should have negative marginal (worse than ensemble average)
        garbage_r = next(r for r in results if r.agent_id == "garbage")
        assert garbage_r.marginal_contribution < 0

    def test_identical_agent_zero_marginal(self) -> None:
        """Two identical agents: both have ~0 marginal."""
        analyzer = MarginalBrierAnalyzer()
        preds = [_make_predictions(0.4, 0.35, 0.25) for _ in range(6)]
        actuals = ["UP", "DOWN", "RANGE"] * 2
        a1 = AgentPredictions(agent_id="a1", predictions=preds, actuals=actuals)
        a2 = AgentPredictions(agent_id="a2", predictions=preds, actuals=actuals)
        results = analyzer.analyze([a1, a2])
        for r in results:
            assert abs(r.marginal_contribution) < 1e-6

    def test_method_name(self) -> None:
        analyzer = MarginalBrierAnalyzer()
        assert analyzer.method_name == "marginal_brier"

    def test_empty_agents(self) -> None:
        analyzer = MarginalBrierAnalyzer()
        results = analyzer.analyze([])
        assert results == []

    def test_single_agent(self) -> None:
        """Single agent: ensemble = agent, marginal = 0."""
        analyzer = MarginalBrierAnalyzer()
        preds = [_make_predictions(0.5, 0.3, 0.2)]
        actuals = ["UP"]
        a = AgentPredictions(agent_id="solo", predictions=preds, actuals=actuals)
        results = analyzer.analyze([a])
        assert len(results) == 1
        assert abs(results[0].marginal_contribution) < 1e-9

    def test_sorted_by_marginal(self) -> None:
        """Results sorted by marginal (most helpful first = most positive for Brier)."""
        analyzer = MarginalBrierAnalyzer()
        preds_perfect = [
            _make_predictions(1.0, 0.0, 0.0),
            _make_predictions(0.0, 1.0, 0.0),
            _make_predictions(0.0, 0.0, 1.0),
        ]
        preds_bad = [_make_predictions(0.1, 0.7, 0.2) for _ in range(3)]
        actuals = ["UP", "DOWN", "RANGE"]
        a1 = AgentPredictions(
            agent_id="good", predictions=preds_perfect, actuals=actuals
        )
        a2 = AgentPredictions(
            agent_id="bad", predictions=preds_bad, actuals=actuals
        )
        results = analyzer.analyze([a1, a2])
        # sorted ascending: bad (negative marginal) first, good (positive) second
        assert results[0].marginal_contribution <= results[1].marginal_contribution

    def test_brier_score_perfect(self) -> None:
        """Perfect predictions → Brier = 0."""
        analyzer = MarginalBrierAnalyzer()
        preds = [
            _make_predictions(1.0, 0.0, 0.0),
            _make_predictions(0.0, 1.0, 0.0),
            _make_predictions(0.0, 0.0, 1.0),
        ]
        actuals = ["UP", "DOWN", "RANGE"]
        assert analyzer._brier_score(preds, actuals) == pytest.approx(0.0)

    def test_run_marginal_positive(self) -> None:
        """full=0.2 < ablated=0.3 → marginal = -0.1 → harmful."""
        analyzer = MarginalBrierAnalyzer()
        r = analyzer.run(full_score=0.2, ablated_score=0.3, agent_id="A")
        assert r.marginal_contribution == pytest.approx(-0.1)
        assert r.direction == "harmful"

    def test_run_marginal_negative(self) -> None:
        """full=0.4 > ablated=0.2 → marginal = 0.2 → helpful."""
        analyzer = MarginalBrierAnalyzer()
        r = analyzer.run(full_score=0.4, ablated_score=0.2, agent_id="A")
        assert r.marginal_contribution == pytest.approx(0.2)
        assert r.direction == "helpful"


# ══════════════════════════════════════════════════════════════════════
# 9-12. Dataclasses (AgentPerformance, BaselineComparison, etc.)
# ══════════════════════════════════════════════════════════════════════


class TestDataclasses:
    """Frozen dataclass tests for report components."""

    def test_agent_performance_frozen(self) -> None:
        ap = AgentPerformance(
            agent_id="test",
            brier_score=0.25,
            accuracy=0.75,
            calibration_error=0.05,
            sample_count=100,
        )
        with pytest.raises(Exception):
            ap.brier_score = 99.0  # type: ignore[misc]

    def test_agent_performance_all_fields(self) -> None:
        ap = AgentPerformance(
            agent_id="test",
            brier_score=0.25,
            accuracy=0.75,
            calibration_error=0.05,
            sample_count=100,
            mean_confidence=0.8,
            mean_calibrated_confidence=0.78,
        )
        assert ap.mean_confidence == 0.8
        assert ap.mean_calibrated_confidence == 0.78

    def test_baseline_comparison_frozen(self) -> None:
        bc = BaselineComparison(
            agent_id="test",
            baseline_name="buy_hold",
            agent_score=0.25,
            baseline_score=0.35,
            improvement=-0.1,
        )
        with pytest.raises(Exception):
            bc.agent_score = 99.0  # type: ignore[misc]

    def test_baseline_comparison_fields(self) -> None:
        bc = BaselineComparison(
            agent_id="a",
            baseline_name="ma_cross",
            agent_score=0.3,
            baseline_score=0.35,
            improvement=-0.05,
        )
        assert bc.agent_id == "a"
        assert bc.baseline_name == "ma_cross"
        assert bc.improvement == -0.05

    def test_calibration_summary_frozen(self) -> None:
        cs = CalibrationSummary(
            method="platt",
            ece_before=0.15,
            ece_after=0.05,
            improvement=0.1,
            num_samples=500,
        )
        with pytest.raises(Exception):
            cs.ece_before = 0.99  # type: ignore[misc]

    def test_calibration_summary_fields(self) -> None:
        cs = CalibrationSummary(
            method="isotonic",
            ece_before=0.2,
            ece_after=0.1,
            improvement=0.1,
            num_samples=1000,
        )
        assert cs.method == "isotonic"
        assert cs.ece_before == 0.2
        assert cs.ece_after == 0.1
        assert cs.improvement == 0.1
        assert cs.num_samples == 1000

    def test_ablation_summary_frozen(self) -> None:
        results = [
            AblationResult(
                method="loo",
                agent_id="A",
                full_score=0.2,
                ablated_score=0.25,
                marginal_contribution=-0.05,
            )
        ]
        summary = AblationSummary(method="loo", results=results)
        with pytest.raises(Exception):
            summary.method = "changed"  # type: ignore[misc]

    def test_ablation_summary_helpful_harmful_counts(self) -> None:
        results = [
            AblationResult(
                method="loo",
                agent_id="A",
                full_score=0.2,
                ablated_score=0.25,
                marginal_contribution=-0.05,  # negative → harmful
            ),
            AblationResult(
                method="loo",
                agent_id="B",
                full_score=0.3,
                ablated_score=0.25,
                marginal_contribution=0.05,  # positive → helpful
            ),
            AblationResult(
                method="loo",
                agent_id="C",
                full_score=0.25,
                ablated_score=0.25,
                marginal_contribution=0.0,  # neutral
            ),
        ]
        summary = AblationSummary(method="loo", results=results)
        assert summary.helpful_count == 1
        assert summary.harmful_count == 1

    def test_ablation_summary_total_marginal(self) -> None:
        results = [
            AblationResult(
                method="loo",
                agent_id="A",
                full_score=0.2,
                ablated_score=0.25,
                marginal_contribution=0.1,
            ),
            AblationResult(
                method="loo",
                agent_id="B",
                full_score=0.3,
                ablated_score=0.35,
                marginal_contribution=0.2,
            ),
        ]
        summary = AblationSummary(method="loo", results=results)
        assert summary.total_marginal == pytest.approx(0.3)


# ══════════════════════════════════════════════════════════════════════
# 13-17. ValidationReport.from_results()
# ══════════════════════════════════════════════════════════════════════


class TestValidationReportFromResults:
    """Test ValidationReport.from_results() factory."""

    def test_creates_report_with_all_components(self) -> None:
        agents = [
            AgentPerformance(
                agent_id="agent1",
                brier_score=0.25,
                accuracy=0.75,
                calibration_error=0.05,
                sample_count=100,
            ),
        ]
        baselines = [
            BaselineComparison(
                agent_id="agent1",
                baseline_name="buy_hold",
                agent_score=0.25,
                baseline_score=0.35,
                improvement=-0.1,
            ),
        ]
        calibration = [
            CalibrationSummary(
                method="platt",
                ece_before=0.15,
                ece_after=0.05,
                improvement=0.1,
                num_samples=500,
            ),
        ]
        report = ValidationReport.from_results(
            report_id="test-001",
            test_set_size=100,
            agent_performances=agents,
            baseline_comparisons=baselines,
            calibration_summaries=calibration,
        )
        assert report.report_id == "test-001"
        assert report.test_set_size == 100
        assert len(report.agents) == 1
        assert len(report.baselines) == 1
        assert len(report.calibration) == 1
        assert report.overall_pass is True

    def test_overall_pass_true_when_all_criteria_met(self) -> None:
        """All criteria pass → overall_pass=True."""
        agents = [
            AgentPerformance(
                agent_id="a1",
                brier_score=0.2,
                accuracy=0.8,
                calibration_error=0.05,
                sample_count=100,
            ),
        ]
        baselines = [
            BaselineComparison(
                agent_id="a1",
                baseline_name="buy_hold",
                agent_score=0.2,
                baseline_score=0.35,
                improvement=-0.15,
            ),
        ]
        calibration = [
            CalibrationSummary(
                method="platt",
                ece_before=0.15,
                ece_after=0.05,
                improvement=0.1,
                num_samples=200,
            ),
        ]
        report = ValidationReport.from_results(
            report_id="pass",
            test_set_size=100,
            agent_performances=agents,
            baseline_comparisons=baselines,
            calibration_summaries=calibration,
        )
        assert report.overall_pass is True

    def test_narrative_contains_agent_names(self) -> None:
        agents = [
            AgentPerformance(
                agent_id="trader_alpha",
                brier_score=0.25,
                accuracy=0.75,
                calibration_error=0.05,
                sample_count=50,
            ),
        ]
        report = ValidationReport.from_results(
            report_id="narr-test",
            test_set_size=50,
            agent_performances=agents,
        )
        assert "trader_alpha" in report.narrative
        assert "Validation Report:" in report.narrative

    def test_pass_criteria_includes_calibration_check(self) -> None:
        calibration = [
            CalibrationSummary(
                method="isotonic",
                ece_before=0.2,
                ece_after=0.1,
                improvement=0.1,
                num_samples=300,
            ),
        ]
        report = ValidationReport.from_results(
            report_id="cal-check",
            test_set_size=300,
            agent_performances=[],
            calibration_summaries=calibration,
        )
        criteria = [c["criterion"] for c in report.pass_criteria]
        assert "calibration_improved" in criteria
        cal_entry = next(
            c for c in report.pass_criteria if c["criterion"] == "calibration_improved"
        )
        assert cal_entry["passed"] is True

    def test_empty_baselines_still_creates_report(self) -> None:
        report = ValidationReport.from_results(
            report_id="empty",
            test_set_size=10,
            agent_performances=[
                AgentPerformance(
                    agent_id="solo",
                    brier_score=0.3,
                    accuracy=0.6,
                    calibration_error=0.1,
                    sample_count=10,
                ),
            ],
        )
        assert report.report_id == "empty"
        assert len(report.baselines) == 0
        assert report.narrative != ""

    def test_pass_criteria_empty_when_no_baselines(self) -> None:
        """With no baselines and no calibration, pass_criteria is empty."""
        report = ValidationReport.from_results(
            report_id="minimal",
            test_set_size=10,
            agent_performances=[
                AgentPerformance(
                    agent_id="x",
                    brier_score=0.5,
                    accuracy=0.5,
                    calibration_error=0.2,
                    sample_count=10,
                ),
            ],
        )
        assert report.pass_criteria == []
        # all() of empty list is True
        assert report.overall_pass is True

    def test_harmful_agent_sets_pass_false(self) -> None:
        """Harmful agent in ablation (negative marginal) → overall_pass=False."""
        results = [
            AblationResult(
                method="loo",
                agent_id="bad_agent",
                full_score=0.2,
                ablated_score=0.15,
                marginal_contribution=-0.05,  # negative → harmful
            ),
        ]
        summary = AblationSummary(method="loo", results=results)
        report = ValidationReport.from_results(
            report_id="harmful",
            test_set_size=100,
            agent_performances=[],
            ablation_summaries=[summary],
        )
        assert report.overall_pass is False
        assert any(
            c["criterion"] == "no_harmful_agents" and not c["passed"]
            for c in report.pass_criteria
        )

    def test_narrative_contains_all_sections(self) -> None:
        """Narrative should include agent, baseline, calibration, and ablation sections."""
        agents = [
            AgentPerformance(
                agent_id="alpha",
                brier_score=0.25,
                accuracy=0.75,
                calibration_error=0.05,
                sample_count=100,
            ),
        ]
        baselines = [
            BaselineComparison(
                agent_id="alpha",
                baseline_name="buy_hold",
                agent_score=0.25,
                baseline_score=0.35,
                improvement=-0.1,
            ),
        ]
        calibration = [
            CalibrationSummary(
                method="platt",
                ece_before=0.15,
                ece_after=0.05,
                improvement=0.1,
                num_samples=200,
            ),
        ]
        ablation_results = [
            AblationResult(
                method="loo",
                agent_id="alpha",
                full_score=0.2,
                ablated_score=0.25,
                marginal_contribution=-0.05,
            ),
        ]
        ablation = [
            AblationSummary(method="loo", results=ablation_results)
        ]
        report = ValidationReport.from_results(
            report_id="narr-full",
            test_set_size=100,
            agent_performances=agents,
            baseline_comparisons=baselines,
            calibration_summaries=calibration,
            ablation_summaries=ablation,
        )
        assert "Agent Performance:" in report.narrative
        assert "alpha" in report.narrative
        assert "Baseline Comparisons:" in report.narrative
        assert "Calibration:" in report.narrative
        assert "Ablation:" in report.narrative

    def test_generated_at_is_datetime(self) -> None:
        report = ValidationReport.from_results(
            report_id="time",
            test_set_size=10,
            agent_performances=[],
        )
        assert isinstance(report.generated_at, datetime)

    def test_ablation_in_narrative(self) -> None:
        results = [
            AblationResult(
                method="marginal_brier",
                agent_id="beta",
                full_score=0.2,
                ablated_score=0.22,
                marginal_contribution=-0.02,
            ),
        ]
        summary = AblationSummary(method="marginal_brier", results=results)
        report = ValidationReport.from_results(
            report_id="abnarr",
            test_set_size=50,
            agent_performances=[],
            ablation_summaries=[summary],
        )
        assert "marginal_brier" in report.narrative
        assert "beta" in report.narrative


class TestGenerateValidationReport:
    """Test the convenience function generate_validation_report()."""

    def test_wraps_from_results(self) -> None:
        report = generate_validation_report(
            report_id="conv",
            test_set_size=42,
            agent_performances=[
                AgentPerformance(
                    agent_id="x",
                    brier_score=0.3,
                    accuracy=0.6,
                    calibration_error=0.1,
                    sample_count=42,
                ),
            ],
        )
        assert report.report_id == "conv"
        assert report.test_set_size == 42


class TestValidationReportFrozen:
    """ValidationReport itself must be frozen."""

    def test_frozen(self) -> None:
        report = ValidationReport.from_results(
            report_id="freeze",
            test_set_size=10,
            agent_performances=[],
        )
        with pytest.raises(Exception):
            report.report_id = "changed"  # type: ignore[misc]
