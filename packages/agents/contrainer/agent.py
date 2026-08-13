"""Contrarian Agent — generates counter-hypothesis to majority view.

Runs BEFORE consensus and provides a counter-hypothesis to the majority
view. Shadow agents are excluded from majority calculation.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import ClassVar

from packages.consensus.base import VoteDirection
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
    InvalidationCondition,
)

from .models import ContrarianConfig, ContrarianHypothesis


class ContrarianAgent:
    """Adversarial review agent that generates counter-hypothesis to majority view.

    Analyzes existing AgentReports to determine the majority vote,
    then generates a contrarian report with inverted probabilities.
    Shadow agents are excluded from majority calculation.
    """

    COUNTER_ARGUMENTS: ClassVar[dict[str, str]] = {
        VoteDirection.LONG: (
            "Bearish signals contradict the bullish consensus. "
            "Overextended price action, distribution patterns, "
            "and deteriorating market structure suggest a reversal."
        ),
        VoteDirection.SHORT: (
            "Bullish fundamentals and technical support invalidate the "
            "bearish case. Accumulation patterns, improving structure, "
            "and positive divergence indicate upward potential."
        ),
        VoteDirection.RANGE: (
            "Trend signals suggest the range bias is overstated. "
            "Momentum indicators and volume profiles indicate "
            "a breakout is more likely than continued consolidation."
        ),
    }

    def __init__(
        self,
        config: ContrarianConfig | None = None,
    ) -> None:
        self._config = config or ContrarianConfig(agent_id="contrarian")

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent."""
        return self._config.agent_id

    @property
    def config(self) -> ContrarianConfig:
        """Configuration for this agent."""
        return self._config

    def analyze(
        self,
        reports: list[AgentReport],
    ) -> tuple[ContrarianHypothesis, AgentReport]:
        """Analyze existing reports and generate contrarian output.

        Excludes shadow agents from majority calculation.
        Generates a report with inverted probabilities and counter-evidence.

        Args:
            reports: AgentReports from other agents.

        Returns:
            Tuple of (contrarian hypothesis, contrarian AgentReport).

        Raises:
            ValueError: If no valid reports are provided.
        """
        if not reports:
            raise ValueError("reports must not be empty")

        # Exclude shadow agents for majority calculation
        active_reports = [
            r for r in reports if r.status != AgentStatus.SHADOW
        ]

        if not active_reports:
            return self._create_abstain_report()

        # Determine majority and minority directions
        majority_direction, minority_direction, is_forced = (
            self._determine_majority(active_reports)
        )

        # For forced contrarian (all agents agree), skip ratio check
        if not is_forced:
            # Check if minority ratio is sufficient
            minority_count = sum(
                1 for r in active_reports
                if self._get_vote(r) == minority_direction
            )
            minority_ratio = minority_count / len(active_reports)

            if minority_ratio < self._config.min_minority_ratio:
                return self._create_abstain_report()

        # Generate counter-argument based on majority direction
        counter_argument = self.COUNTER_ARGUMENTS.get(
            majority_direction,
            "Adversarial analysis suggests the opposite of the majority view.",
        )

        # Calculate confidence based on minority agreement
        if is_forced:
            confidence = 0.7  # Default confidence for forced contrarian
        else:
            minority_count = sum(
                1 for r in active_reports
                if self._get_vote(r) == minority_direction
            )
            minority_ratio = minority_count / len(active_reports)
            confidence = min(1.0, minority_ratio * 2.0)

        # Generate evidence from minority agents (or contrarian rationale)
        evidence = self._collect_evidence(
            active_reports, minority_direction
        )

        hypothesis = ContrarianHypothesis(
            counter_argument=counter_argument,
            confidence=confidence,
            evidence=evidence,
            majority_direction=majority_direction,
            minority_direction=minority_direction,
        )

        report = self._build_report(
            minority_direction,
            confidence,
            evidence,
            active_reports,
        )

        return hypothesis, report

    def _determine_majority(
        self, reports: list[AgentReport]
    ) -> tuple[str, str, bool]:
        """Determine majority, minority, and whether the minority is forced.

        When all agents agree, returns the opposite direction as a forced
        contrarian signal with is_forced=True.
        """
        votes: Counter[str] = Counter()
        for report in reports:
            votes[self._get_vote(report)] += 1

        if not votes:
            return VoteDirection.ABSTAIN, VoteDirection.ABSTAIN, False

        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        majority_direction = sorted_votes[0][0]

        unique_directions = set(votes.keys())

        if len(unique_directions) > 1:
            minority_direction = sorted_votes[-1][0]
            return majority_direction, minority_direction, False
        else:
            # No genuine minority — contrarian flips to the opposite.
            minority_direction = self._contrarian_opposite(
                majority_direction
            )
            return majority_direction, minority_direction, True

    @staticmethod
    def _contrarian_opposite(direction: str) -> str:
        """Return the contrarian opposite of a direction."""
        mapping: dict[str, str] = {
            VoteDirection.LONG: VoteDirection.SHORT,
            VoteDirection.SHORT: VoteDirection.LONG,
            VoteDirection.RANGE: VoteDirection.LONG,
        }
        return mapping.get(direction, VoteDirection.ABSTAIN)

    def _get_vote(self, report: AgentReport) -> str:
        """Determine the vote direction for a report."""
        probs = report.probabilities
        if probs.get("up", 0.0) > 0.6:
            return VoteDirection.LONG
        if probs.get("down", 0.0) > 0.6:
            return VoteDirection.SHORT
        if probs.get("range", 0.0) > 0.5:
            return VoteDirection.RANGE
        return VoteDirection.ABSTAIN

    def _collect_evidence(
        self,
        reports: list[AgentReport],
        direction: str,
    ) -> list[str]:
        """Collect evidence supporting the given direction."""
        evidence: list[str] = []
        for report in reports:
            if self._get_vote(report) == direction:
                for ev in report.evidence[:3]:
                    evidence.append(
                        f"{report.agent_id}: {ev.feature}={ev.value} "
                        f"({ev.direction}, rel={ev.relevance:.2f})"
                    )
        return evidence if evidence else ["Insufficient evidence from minority agents"]

    def _build_report(
        self,
        direction: str,
        confidence: float,
        evidence: list[str],
        reports: list[AgentReport],
    ) -> AgentReport:
        """Build contrarian AgentReport with inverted probabilities."""
        # Invert probabilities
        if direction == VoteDirection.LONG:
            probs = {"up": 0.7, "down": 0.2, "range": 0.1}
        elif direction == VoteDirection.SHORT:
            probs = {"up": 0.1, "down": 0.7, "range": 0.2}
        elif direction == VoteDirection.RANGE:
            probs = {"up": 0.2, "down": 0.2, "range": 0.6}
        else:
            probs = {"up": 0.33, "down": 0.33, "range": 0.34}

        # Build evidence references
        evidence_refs = [
            EvidenceReference(
                reference=f"contrarian:{i}",
                feature=f"signal_{i}",
                value=ev,
                direction="neutral",
                relevance=0.5,
            )
            for i, ev in enumerate(evidence[:5])
        ]

        # Build invalidations
        invalidations = [
            InvalidationCondition(
                condition=f"Majority shifts to {direction}",
                indicator="consensus",
                threshold=0.5,
                direction="above",
            ),
        ]

        instrument = reports[0].instrument if reports else "UNKNOWN"
        run_id = reports[0].run_id if reports else "abstain"

        return AgentReport(
            report_id=f"contrarian-{self.agent_id}-{datetime.now(UTC).isoformat()}",
            run_id=run_id,
            agent_id=self.agent_id,
            agent_version=self._config.agent_version,
            instrument=instrument,
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis=f"Contrarian: Opposing {self._direction_name(direction)}",
            probabilities=probs,
            evidence=evidence_refs,
            invalidations=invalidations,
            status=AgentStatus.SHADOW,
            raw_confidence=confidence,
            calibrated_confidence=confidence,
            expected_return=None,
        )

    def _create_abstain_report(
        self,
    ) -> tuple[ContrarianHypothesis, AgentReport]:
        """Create abstain report when minority is insufficient."""
        hypothesis = ContrarianHypothesis(
            counter_argument="Insufficient minority agreement.",
            confidence=0.0,
            evidence=[],
            majority_direction="UNKNOWN",
            minority_direction="UNKNOWN",
        )
        return hypothesis, AgentReport(
            report_id=f"contrarian-{self.agent_id}-abstain",
            run_id="abstain",
            agent_id=self.agent_id,
            agent_version=self._config.agent_version,
            instrument="UNKNOWN",
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis="Abstained: Insufficient minority agreement",
            probabilities={"up": 0.33, "down": 0.33, "range": 0.34},
            evidence=[
                EvidenceReference(
                    reference="contrarian:no-data",
                    feature="no_data",
                    value="abstain",
                    direction="neutral",
                    relevance=0.0,
                ),
            ],
            invalidations=[],
            status=AgentStatus.SHADOW,
            raw_confidence=0.0,
            calibrated_confidence=0.0,
            expected_return=None,
        )

    @staticmethod
    def _direction_name(direction: str) -> str:
        """Convert direction to human-readable name."""
        return direction.value if isinstance(direction, VoteDirection) else direction
