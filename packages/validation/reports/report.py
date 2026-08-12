"""Validation Report Generation.

Creates structured reports summarizing:
- Agent performance (Brier score, accuracy, calibration error)
- Baseline comparisons (vs buy_hold, ma_cross, etc.)
- Calibration summary (before/after ECE)
- Ablation summary (LOO, feature importance, marginal Brier)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import ConfigDict

from ..ablation.base import AblationResult


@dataclass(frozen=True)
class AgentPerformance:
    """Performance metrics for a single agent.

    Attributes:
        agent_id: The agent's identifier.
        brier_score: Mean Brier score over test set.
        accuracy: Classification accuracy.
        calibration_error: ECE (Expected Calibration Error).
        sample_count: Number of test samples.
        mean_confidence: Average raw confidence.
        mean_calibrated_confidence: Average calibrated confidence.
    """

    model_config = ConfigDict(frozen=True)

    agent_id: str
    brier_score: float
    accuracy: float
    calibration_error: float
    sample_count: int
    mean_confidence: float = 0.0
    mean_calibrated_confidence: float = 0.0


@dataclass(frozen=True)
class BaselineComparison:
    """Comparison of agents against baselines.

    Attributes:
        agent_id: The agent being compared.
        baseline_name: Name of the baseline (e.g. "buy_hold").
        agent_score: Agent's Brier score.
        baseline_score: Baseline's Brier score.
        improvement: agent_score - baseline_score (negative = agent better).
    """

    model_config = ConfigDict(frozen=True)

    agent_id: str
    baseline_name: str
    agent_score: float
    baseline_score: float
    improvement: float


@dataclass(frozen=True)
class CalibrationSummary:
    """Summary of calibration results.

    Attributes:
        method: Calibration method used (platt/isotonic/temperature).
        ece_before: ECE before calibration.
        ece_after: ECE after calibration.
        improvement: ece_before - ece_after.
        num_samples: Number of calibration samples.
    """

    model_config = ConfigDict(frozen=True)

    method: str
    ece_before: float
    ece_after: float
    improvement: float
    num_samples: int


@dataclass(frozen=True)
class AblationSummary:
    """Summary of ablation results.

    Attributes:
        method: Ablation method (loo/feature_importance/marginal_brier).
        results: Individual ablation results.
        helpful_count: Number of helpful ablations.
        harmful_count: Number of harmful ablations.
        total_marginal: Sum of all marginal contributions.
    """

    model_config = ConfigDict(frozen=True)

    method: str
    results: list[AblationResult]

    @property
    def helpful_count(self) -> int:
        """Number of ablations where the agent contributed positively."""
        return sum(1 for r in self.results if r.is_helpful)

    @property
    def harmful_count(self) -> int:
        """Number of ablations where the agent contributed negatively."""
        return sum(1 for r in self.results if r.is_harmful)

    @property
    def total_marginal(self) -> float:
        """Sum of all marginal contributions."""
        return sum(r.marginal_contribution for r in self.results)


@dataclass(frozen=True)
class ValidationReport:
    """Complete validation report.

    Attributes:
        report_id: Unique report identifier.
        generated_at: Timestamp of report generation.
        test_set_size: Number of test samples.
        agents: Per-agent performance metrics.
        baselines: Baseline comparisons.
        calibration: Calibration summaries.
        ablation: Ablation summaries.
        overall_pass: Whether all validation criteria are met.
        pass_criteria: List of individual pass/fail criteria.
        narrative: Human-readable summary.
    """

    model_config = ConfigDict(frozen=True)

    report_id: str
    generated_at: datetime = field(default_factory=datetime.now)
    test_set_size: int = 0
    agents: list[AgentPerformance] = field(default_factory=list)
    baselines: list[BaselineComparison] = field(default_factory=list)
    calibration: list[CalibrationSummary] = field(default_factory=list)
    ablation: list[AblationSummary] = field(default_factory=list)
    overall_pass: bool = True
    pass_criteria: list[dict[str, Any]] = field(default_factory=list)
    narrative: str = ""

    @classmethod
    def from_results(
        cls,
        report_id: str,
        test_set_size: int,
        agent_performances: list[AgentPerformance],
        baseline_comparisons: list[BaselineComparison] | None = None,
        calibration_summaries: list[CalibrationSummary] | None = None,
        ablation_summaries: list[AblationSummary] | None = None,
    ) -> ValidationReport:
        """Factory method to construct a report from component results.

        Args:
            report_id: Unique report identifier.
            test_set_size: Number of test samples.
            agent_performances: Per-agent performance metrics.
            baseline_comparisons: Optional baseline comparisons.
            calibration_summaries: Optional calibration summaries.
            ablation_summaries: Optional ablation summaries.

        Returns:
            Fully populated ValidationReport.
        """
        baseline_comparisons = baseline_comparisons or []
        calibration_summaries = calibration_summaries or []
        ablation_summaries = ablation_summaries or []

        pass_criteria: list[dict[str, Any]] = []

        # Criterion 1: At least one agent beats buy_hold
        for bc in baseline_comparisons:
            if bc.baseline_name == "buy_hold":
                buy_hold_brier = bc.baseline_score
                pass_criteria.append(
                    {
                        "criterion": "agent_beats_buy_hold",
                        "passed": any(
                            bc2.agent_score < bc.baseline_score
                            for bc2 in baseline_comparisons
                            if bc2.agent_id == bc.agent_id
                        ),
                        "details": f"At least one agent Brier < {buy_hold_brier:.4f}",
                    }
                )
                break

        # Criterion 2: Calibration improved or unchanged
        for cs in calibration_summaries:
            pass_criteria.append(
                {
                    "criterion": "calibration_improved",
                    "passed": cs.improvement >= 0,
                    "details": (
                        f"{cs.method}: ECE {cs.ece_before:.4f} -> {cs.ece_after:.4f}"
                    ),
                }
            )

        # Criterion 3: No agent significantly harmful in ablation
        for a_summary in ablation_summaries:
            for ar in a_summary.results:
                if ar.is_harmful:
                    pass_criteria.append(
                        {
                            "criterion": "no_harmful_agents",
                            "passed": False,
                            "details": f"{ar.agent_id} is harmful in {a_summary.method}",
                        }
                    )

        overall_pass = all(pc["passed"] for pc in pass_criteria)

        narrative = cls._generate_narrative(
            test_set_size,
            agent_performances,
            baseline_comparisons,
            calibration_summaries,
            ablation_summaries,
        )

        return cls(
            report_id=report_id,
            test_set_size=test_set_size,
            agents=agent_performances,
            baselines=baseline_comparisons,
            calibration=calibration_summaries,
            ablation=ablation_summaries,
            overall_pass=overall_pass,
            pass_criteria=pass_criteria,
            narrative=narrative,
        )

    @staticmethod
    def _generate_narrative(
        test_set_size: int,
        agents: list[AgentPerformance],
        baselines: list[BaselineComparison],
        calibration: list[CalibrationSummary],
        ablation: list[AblationSummary],
    ) -> str:
        """Generate human-readable summary."""
        lines: list[str] = []
        lines.append(
            f"Validation Report: {test_set_size} test samples, {len(agents)} agents"
        )
        lines.append("")

        lines.append("Agent Performance:")
        for a in agents:
            lines.append(
                f"  {a.agent_id}: Brier={a.brier_score:.4f}, "
                f"Accuracy={a.accuracy:.4f}, ECE={a.calibration_error:.4f}"
            )
        lines.append("")

        if baselines:
            lines.append("Baseline Comparisons:")
            for bc in baselines:
                lines.append(
                    f"  {bc.agent_id} vs {bc.baseline_name}: "
                    f"{bc.improvement:+.4f}"
                )
            lines.append("")

        if calibration:
            lines.append("Calibration:")
            for cs in calibration:
                lines.append(
                    f"  {cs.method}: ECE {cs.ece_before:.4f} -> {cs.ece_after:.4f} "
                    f"({cs.improvement:+.4f})"
                )
            lines.append("")

        if ablation:
            lines.append("Ablation:")
            for a_summary in ablation:
                lines.append(f"  {a_summary.method}:")
                for ar in a_summary.results[:5]:  # Top 5
                    lines.append(
                        f"    {ar.agent_id}: marginal={ar.marginal_contribution:+.4f} "
                        f"({ar.direction})"
                    )

        return "\n".join(lines)


def generate_validation_report(
    report_id: str,
    test_set_size: int,
    agent_performances: list[AgentPerformance],
    baseline_comparisons: list[BaselineComparison] | None = None,
    calibration_summaries: list[CalibrationSummary] | None = None,
    ablation_summaries: list[AblationSummary] | None = None,
) -> ValidationReport:
    """Convenience function to generate a validation report.

    Args:
        report_id: Unique report identifier.
        test_set_size: Number of test samples.
        agent_performances: Per-agent performance metrics.
        baseline_comparisons: Optional baseline comparisons.
        calibration_summaries: Optional calibration summaries.
        ablation_summaries: Optional ablation summaries.

    Returns:
        Fully populated ValidationReport.
    """
    return ValidationReport.from_results(
        report_id=report_id,
        test_set_size=test_set_size,
        agent_performances=agent_performances,
        baseline_comparisons=baseline_comparisons,
        calibration_summaries=calibration_summaries,
        ablation_summaries=ablation_summaries,
    )
