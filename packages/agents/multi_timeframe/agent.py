"""Multi-Timeframe Agent — cross-timeframe consensus analysis.

Analyzes signals across 1m, 5m, 15m, 1h, 4h, 1d timeframes
and produces an aggregated report with cross-timeframe consensus.
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

from .models import (
    MultiTimeframeConfig,
    MultiTimeframeReport,
    TimeframeSignal,
)

TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")


class MultiTimeframeAgent:
    """Cross-timeframe analysis agent that aggregates signals across timeframes.

    Analyzes each timeframe independently, detects conflicts between
    timeframes, and produces an aggregated report with overall consensus.
    """

    DEFAULT_WEIGHTS: ClassVar[dict[str, float]] = {
        "1m": 0.05,
        "5m": 0.1,
        "15m": 0.15,
        "1h": 0.25,
        "4h": 0.25,
        "1d": 0.2,
    }

    def __init__(
        self,
        config: MultiTimeframeConfig | None = None,
    ) -> None:
        self._config = config or MultiTimeframeConfig()

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent."""
        return self._config.agent_id

    @property
    def config(self) -> MultiTimeframeConfig:
        """Configuration for this agent."""
        return self._config

    def analyze(
        self,
        timeframe_data: dict[str, dict[str, float]],
    ) -> tuple[MultiTimeframeReport, AgentReport]:
        """Analyze signals across timeframes and produce aggregated report.

        Args:
            timeframe_data: Dict mapping timeframe -> {up, down, range} probabilities.

        Returns:
            Tuple of (MultiTimeframeReport, AgentReport).

        Raises:
            ValueError: If no valid timeframe data is provided.
        """
        if not timeframe_data:
            raise ValueError("timeframe_data must not be empty")

        # Build timeframe signals
        signals: list[TimeframeSignal] = []
        for tf in self._config.timeframes:
            if tf in timeframe_data:
                probs = timeframe_data[tf]
                weight = self._config.timeframe_weights.get(tf, 0.1)
                direction = self._determine_direction(probs)
                confidence = self._calculate_confidence(probs)
                signals.append(TimeframeSignal(
                    timeframe=tf,
                    direction=direction,
                    probability=self._max_prob(probs),
                    weight=weight,
                    confidence=confidence,
                ))

        # Weighted vote
        direction_votes: Counter[str] = Counter()
        total_weight = 0.0
        for sig in signals:
            direction_votes[sig.direction] += int(sig.weight)
            total_weight += sig.weight

        if total_weight == 0 or not direction_votes:
            return self._create_abstain_report()

        # Determine overall direction by weight
        overall_direction = max(direction_votes, key=lambda k: direction_votes.get(k, 0))

        # Calculate conflicts
        conflicts = self._detect_conflicts(signals)

        # Calculate overall agreement
        agreement = self._calculate_agreement(signals, overall_direction)

        # Calculate overall confidence
        confidence = sum(
            sig.confidence * sig.weight for sig in signals
        ) / max(total_weight, 0.001)

        report = MultiTimeframeReport(
            direction=overall_direction,
            confidence=confidence,
            timeframe_signals=signals,
            conflicts=conflicts,
            overall_agreement=agreement,
        )

        agent_report = self._build_agent_report(
            report, signals, conflicts
        )

        return report, agent_report

    def _determine_direction(self, probs: dict[str, float]) -> str:
        """Determine direction from probabilities for a single timeframe."""
        up = probs.get("up", 0.33)
        down = probs.get("down", 0.33)
        rng = probs.get("range", 0.34)
        if up > down and up > rng:
            return VoteDirection.LONG
        if down > up and down > rng:
            return VoteDirection.SHORT
        return VoteDirection.RANGE

    def _max_prob(self, probs: dict[str, float]) -> float:
        """Return the maximum probability value."""
        return max(probs.values()) if probs else 0.0

    def _calculate_confidence(self, probs: dict[str, float]) -> float:
        """Calculate confidence as the max probability."""
        return max(probs.values()) if probs else 0.0

    def _detect_conflicts(
        self, signals: list[TimeframeSignal]
    ) -> list[str]:
        """Detect conflicting signals between timeframes."""
        conflicts: list[str] = []
        directions = {s.timeframe: s.direction for s in signals}
        long_tfs = [tf for tf, d in directions.items() if d == VoteDirection.LONG]
        short_tfs = [tf for tf, d in directions.items() if d == VoteDirection.SHORT]
        if long_tfs and short_tfs:
            conflicts.append(
                f"SHORT {short_tfs} vs LONG {long_tfs}"
            )
        return conflicts

    def _calculate_agreement(
        self,
        signals: list[TimeframeSignal],
        overall_direction: str,
    ) -> float:
        """Calculate agreement ratio (how many signals agree with overall)."""
        if not signals:
            return 0.0
        agreeing = sum(
            1 for s in signals if s.direction == overall_direction
        )
        return agreeing / len(signals)

    def _create_abstain_report(
        self,
    ) -> tuple[MultiTimeframeReport, AgentReport]:
        """Create abstain report when no valid data."""
        report = MultiTimeframeReport(
            direction=VoteDirection.ABSTAIN,
            confidence=0.0,
            timeframe_signals=[],
            conflicts=[],
            overall_agreement=0.0,
        )
        agent_report = AgentReport(
            report_id=f"multi_tf-{self.agent_id}-abstain",
            run_id="abstain",
            agent_id=self.agent_id,
            agent_version=self._config.agent_version,
            instrument="UNKNOWN",
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis="Abstained: No valid timeframe data",
            probabilities={"up": 0.33, "down": 0.33, "range": 0.34},
            evidence=[],
            invalidations=[],
            status=AgentStatus.SHADOW,
            raw_confidence=0.0,
            calibrated_confidence=0.0,
            expected_return=None,
        )
        return report, agent_report

    def _build_agent_report(
        self,
        report: MultiTimeframeReport,
        signals: list[TimeframeSignal],
        conflicts: list[str],
    ) -> AgentReport:
        """Build AgentReport from MultiTimeframeReport."""
        probs = {
            "up": 0.0, "down": 0.0, "range": 0.0,
        }
        for sig in signals:
            if sig.direction == VoteDirection.LONG:
                probs["up"] += sig.weight * sig.confidence
            elif sig.direction == VoteDirection.SHORT:
                probs["down"] += sig.weight * sig.confidence
            else:
                probs["range"] += sig.weight * sig.confidence

        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs = {"up": 0.33, "down": 0.33, "range": 0.34}

        evidence_refs = [
            EvidenceReference(
                reference=f"tf:{sig.timeframe}",
                feature=f"tf_{sig.timeframe}_direction",
                value=f"{sig.direction} (p={sig.probability:.2f})",
                direction=sig.direction,
                relevance=sig.confidence,
            )
            for sig in signals
        ]
        evidence_refs.extend(
            EvidenceReference(
                reference=f"conflict:{i}",
                feature="timeframe_conflict",
                value=conflict,
                direction="neutral",
                relevance=0.5,
            )
            for i, conflict in enumerate(conflicts)
        )

        return AgentReport(
            report_id=f"multi_tf-{self.agent_id}-{datetime.now(UTC).isoformat()}",
            run_id="multi_tf",
            agent_id=self.agent_id,
            agent_version=self._config.agent_version,
            instrument="UNKNOWN",
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis=(
                f"Multi-TF: {report.direction} "
                f"(conf={report.confidence:.2f}, agree={report.overall_agreement:.2f})"
            ),
            probabilities=probs,
            evidence=evidence_refs,
            invalidations=[
                InvalidationCondition(
                    condition="Cross-timeframe consensus shifts",
                    indicator="consensus_direction",
                    threshold=0.5,
                    direction="above",
                ),
            ],
            status=AgentStatus.SHADOW,
            raw_confidence=report.confidence,
            calibrated_confidence=report.confidence,
            expected_return=None,
        )
