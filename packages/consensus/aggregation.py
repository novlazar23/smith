"""Calibrated consensus aggregation with dependency-adjusted weights.

Combines dependency-reduced agent weights with calibration from
packages.validation.calibration to produce calibrated probabilities,
dissent detection, and a final consensus decision.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

from packages.consensus.base import (
    ConsensusDecision,
    VoteDirection,
    WeightConfig,
)
from packages.schemas.agent_report import AgentReport
from packages.validation.calibration import (
    CalibrationSample,
    Calibrator,
    PlattCalibrator,
)


@dataclass
class CalibratedConsensusResult:
    """Result of calibrated consensus aggregation.

    Attributes:
        decision: Final consensus decision.
        vote_distribution: Raw vote direction distribution.
        calibrated_probabilities: Calibrated up/down/range probabilities.
        agent_weights: Dependency-adjusted agent weights.
        dissent_detected: True if agents show significant disagreement.
        dissent_score: Max(|long_ratio - short_ratio|) used for dissent.
        confidence: Aggregation confidence score.
        reason: Human-readable explanation.
    """

    decision: ConsensusDecision
    vote_distribution: dict[VoteDirection, float]
    calibrated_probabilities: dict[str, float]
    agent_weights: dict[str, float]
    dissent_detected: bool
    dissent_score: float
    confidence: float
    reason: str


class CalibratedConsensusAggregator:
    """Aggregates AgentReports with dependency-adjusted weights and calibration.

    Takes a WeightConfig, an optional Calibrator, and historical accuracy data
    to produce calibrated consensus decisions.

    Attributes:
        config: Weight configuration.
        calibrator: Optional calibrator (defaults to PlattCalibrator).
    """

    def __init__(
        self,
        config: WeightConfig | None = None,
        calibrator: Calibrator | None = None,
    ) -> None:
        self.config = config or WeightConfig()
        self.calibrator = calibrator or PlattCalibrator()

    def _compute_agent_weight(
        self, report: AgentReport, adjusted_weight: float | None = None
    ) -> float:
        """Compute the final weight for an agent.

        Shadow agents always get weight 0.0. Otherwise uses the adjusted
        weight from dependency analysis or falls back to the standard
        status-based computation.

        Args:
            report: The agent report.
            adjusted_weight: Pre-computed dependency-adjusted weight.

        Returns:
            Final weight in [0.0, 1.0].
        """
        status = report.status.value
        if status == "shadow":
            return 0.0

        if adjusted_weight is not None:
            return adjusted_weight

        base = self.config.base_weight
        multiplier = self.config.status_multiplier.get(status, 1.0)
        return base * multiplier

    def _determine_vote(
        self, probabilities: dict[str, float]
    ) -> VoteDirection:
        """Determine vote direction from probabilities."""
        if probabilities.get("up", 0.0) > 0.6:
            return VoteDirection.LONG
        if probabilities.get("down", 0.0) > 0.6:
            return VoteDirection.SHORT
        if probabilities.get("range", 0.0) > 0.5:
            return VoteDirection.RANGE
        return VoteDirection.ABSTAIN

    def _aggregate_scores(
        self,
        reports: list[AgentReport],
        adjusted_weights: dict[str, float] | None = None,
    ) -> tuple[float, float, float, float]:
        """Aggregate weighted scores across all reports.

        Args:
            reports: Agent reports.
            adjusted_weights: Optional dependency-adjusted weights dict.

        Returns:
            Tuple of (long_score, short_score, range_score, abstain_weight).
        """
        long_score = 0.0
        short_score = 0.0
        range_score = 0.0
        abstain_weight = 0.0

        for report in reports:
            weight = self._compute_agent_weight(
                report,
                adjusted_weights.get(report.agent_id)
                if adjusted_weights
                else None,
            )
            vote = self._determine_vote(report.probabilities)

            if vote == VoteDirection.LONG:
                long_score += weight
            elif vote == VoteDirection.SHORT:
                short_score += weight
            elif vote == VoteDirection.RANGE:
                range_score += weight
            else:
                abstain_weight += weight

        return long_score, short_score, range_score, abstain_weight

    def _compute_calibrated_probabilities(
        self,
        reports: list[AgentReport],
        adjusted_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute calibrated up/down/range probabilities.

        If a calibrator is fitted, applies calibration to raw confidences.
        Otherwise returns the simple weighted averages.

        Args:
            reports: Agent reports.
            adjusted_weights: Optional dependency-adjusted weights.

        Returns:
            Dict with 'up', 'down', 'range' calibrated probabilities.
        """
        # Build weighted averages as raw probabilities
        weight_total = 0.0
        up_sum = 0.0
        down_sum = 0.0
        range_sum = 0.0

        for report in reports:
            weight = self._compute_agent_weight(
                report,
                adjusted_weights.get(report.agent_id)
                if adjusted_weights
                else None,
            )
            weight_total += weight
            up_sum += weight * report.probabilities.get("up", 0.0)
            down_sum += weight * report.probabilities.get("down", 0.0)
            range_sum += weight * report.probabilities.get("range", 0.0)

        if weight_total > 0:
            up_raw = up_sum / weight_total
            down_raw = down_sum / weight_total
            range_raw = range_sum / weight_total
        else:
            up_raw = 0.0
            down_raw = 0.0
            range_raw = 0.0

        # Apply calibration if available
        if self.calibrator.is_fitted:
            up_raw = max(1e-7, min(1 - 1e-7, up_raw))
            down_raw = max(1e-7, min(1 - 1e-7, down_raw))
            range_raw = max(1e-7, min(1 - 1e-7, range_raw))
            up_calibrated = self.calibrator.calibrate(up_raw)
            down_calibrated = self.calibrator.calibrate(down_raw)
            range_calibrated = self.calibrator.calibrate(range_raw)
        else:
            up_calibrated = up_raw
            down_calibrated = down_raw
            range_calibrated = range_raw

        return {
            "up_calibrated": up_calibrated,
            "down_calibrated": down_calibrated,
            "range_calibrated": range_calibrated,
        }

    def _prepare_calibration_samples(
        self, reports: list[AgentReport]
    ) -> list[CalibrationSample]:
        """Build calibration samples from agent reports.

        Each report provides a raw confidence (from calibrated_confidence
        or raw_confidence, defaulting to max probability) and an implied
        actual outcome (1.0 if LONG vote, 0.0 otherwise).

        Args:
            reports: Agent reports.

        Returns:
            List of CalibrationSample objects.
        """
        samples: list[CalibrationSample] = []
        for report in reports:
            raw = report.calibrated_confidence or report.raw_confidence
            if raw is None:
                # Fallback: use max probability as confidence
                raw = max(
                    report.probabilities.values()
                ) if report.probabilities else 0.5

            vote = self._determine_vote(report.probabilities)
            actual = 1.0 if vote == VoteDirection.LONG else 0.0

            samples.append(
                CalibrationSample(
                    raw_confidence=float(raw),
                    actual=actual,
                    direction=report.probabilities.get("up", "up"),
                    sample_id=report.agent_id,
                )
            )
        return samples

    def aggregate(
        self,
        reports: list[AgentReport],
        adjusted_weights: dict[str, float] | None = None,
        fit_calibrator: bool = False,  # noqa: FBT001,FBT002
    ) -> CalibratedConsensusResult:
        """Perform calibrated consensus aggregation.

        Args:
            reports: Agent reports to aggregate.
            adjusted_weights: Optional dependency-adjusted weights dict
                from DependencyAnalyzer.reduce_weights().
            fit_calibrator: If True, fit the calibrator on the input reports
                before aggregating.

        Returns:
            CalibratedConsensusResult with decision, calibrated probabilities,
            and dissent detection.
        """
        if not reports:
            raise ValueError("reports must not be empty")

        for report in reports:
            if not report.probabilities:
                raise ValueError(
                    f"report {report.report_id} has empty probabilities"
                )

        # Fit calibrator if requested
        if fit_calibrator and len(reports) >= 2:
            samples = self._prepare_calibration_samples(reports)
            with suppress(ValueError):
                self.calibrator.fit(samples)

        # Aggregate weighted scores
        long_score, short_score, range_score, abstain_weight = (
            self._aggregate_scores(reports, adjusted_weights)
        )

        total = long_score + short_score + range_score + abstain_weight

        if total == 0.0:
            decision = ConsensusDecision.NO_TRADE
            confidence = 0.0
            dissent_detected = False
            dissent_score = 0.0
        else:
            long_ratio = long_score / total
            short_ratio = short_score / total
            range_ratio = range_score / total

            # Detect dissent: close long/short scores indicate no clear consensus
            # Only detect dissent when there are actually opposing votes
            dissent_score = abs(long_ratio - short_ratio)
            dissent_detected = (
                dissent_score < 0.1 and (long_ratio > 0 or short_ratio > 0)
            )

            if dissent_detected:
                decision = ConsensusDecision.NO_TRADE
                confidence = 0.0
            elif long_ratio > self.config.min_consensus_threshold:
                decision = ConsensusDecision.LONG_BIAS
                confidence = long_ratio
            elif short_ratio > self.config.min_consensus_threshold:
                decision = ConsensusDecision.SHORT_BIAS
                confidence = short_ratio
            elif range_ratio > self.config.min_consensus_threshold:
                decision = ConsensusDecision.RANGE
                confidence = range_ratio
            else:
                decision = ConsensusDecision.NO_TRADE
                confidence = max(
                    long_ratio, short_ratio, range_ratio
                )

        # Build vote distribution
        vote_distribution = {
            VoteDirection.LONG: long_score / total if total > 0 else 0.0,
            VoteDirection.SHORT: short_score / total if total > 0 else 0.0,
            VoteDirection.RANGE: range_score / total if total > 0 else 0.0,
            VoteDirection.ABSTAIN: abstain_weight
            / total
            if total > 0
            else 0.0,
        }

        # Compute calibrated probabilities
        calibrated_probs = self._compute_calibrated_probabilities(
            reports, adjusted_weights
        )

        # Build agent weights dict
        agent_weights: dict[str, float] = {}
        for report in reports:
            agent_weights[report.agent_id] = self._compute_agent_weight(
                report,
                adjusted_weights.get(report.agent_id)
                if adjusted_weights
                else None,
            )

        reason = (
            f"{decision.value}: long={long_score:.3f}, "
            f"short={short_score:.3f}, range={range_score:.3f}, "
            f"dissent={dissent_detected}"
        )

        return CalibratedConsensusResult(
            decision=decision,
            vote_distribution=vote_distribution,
            calibrated_probabilities=calibrated_probs,
            agent_weights=agent_weights,
            dissent_detected=dissent_detected,
            dissent_score=dissent_score,
            confidence=confidence,
            reason=reason,
        )
