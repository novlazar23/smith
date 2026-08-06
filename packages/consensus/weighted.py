"""Weighted consensus engine that aggregates AgentReports.

Computes a weighted consensus vote from multiple agent reports, taking
into account each agent's status-based weight and their predicted
direction (long/short/range/abstain).
"""

from __future__ import annotations

from packages.schemas.agent_report import AgentReport

from .base import (
    ConsensusDecision,
    ConsensusResult,
    VoteDirection,
    WeightConfig,
)


class WeightedConsensusEngine:
    """Aggregiert AgentReports zu einem gewichteten Konsens."""

    def __init__(self, config: WeightConfig | None = None) -> None:
        self.config = config or WeightConfig()

    def validate_input(self, reports: list[AgentReport]) -> None:
        """Validiert die Eingabedaten vor der Konsensberechnung.

        Args:
            reports: Liste von AgentReports zur Auswertung.

        Raises:
            ValueError: Wenn die Liste leer ist oder Reports keine Wahrscheinlichkeiten haben.
        """
        if not reports:
            raise ValueError("reports must not be empty")
        for report in reports:
            if not report.probabilities:
                raise ValueError(
                    f"report {report.report_id} has empty probabilities"
                )

    def _compute_agent_weight(self, report: AgentReport) -> float:
        """Berechnet das Gewicht eines einzelnen Agenten."""
        base = self.config.base_weight
        status = report.status.value
        multiplier = self.config.status_multiplier.get(status, 1.0)
        return base * multiplier

    def _determine_vote(self, probabilities: dict[str, float]) -> VoteDirection:
        """Bestimmt die Abstimmungsrichtung aus den Wahrscheinlichkeiten.

        Args:
            probabilities: Dict mit Schlüsseln 'up', 'down', 'range'.

        Returns:
            VoteDirection basierend auf den Wahrscheinlichkeiten.
        """
        if probabilities.get("up", 0.0) > 0.6:
            return VoteDirection.LONG
        if probabilities.get("down", 0.0) > 0.6:
            return VoteDirection.SHORT
        if probabilities.get("range", 0.0) > 0.5:
            return VoteDirection.RANGE
        return VoteDirection.ABSTAIN

    def compute_consensus(self, reports: list[AgentReport]) -> ConsensusResult:
        """Berechnet den gewichteten Konsens aus einer Liste von AgentReports.

        Args:
            reports: Liste von AgentReports zur Auswertung.

        Returns:
            ConsensusResult mit Entscheidung, Gewichten und Konfidenz.

        Raises:
            ValueError: Bei ungültigen Eingaben (siehe validate_input).
        """
        self.validate_input(reports)

        # Compute per-agent weights and votes
        agent_data: list[tuple[str, VoteDirection, float]] = []
        for report in reports:
            weight = self._compute_agent_weight(report)
            vote = self._determine_vote(report.probabilities)
            agent_data.append((report.agent_id, vote, weight))

        # Aggregate weighted scores
        long_score = 0.0
        short_score = 0.0
        range_score = 0.0
        abstain_weight = 0.0

        for _, vote, weight in agent_data:
            if vote == VoteDirection.LONG:
                long_score += weight
            elif vote == VoteDirection.SHORT:
                short_score += weight
            elif vote == VoteDirection.RANGE:
                range_score += weight
            else:
                abstain_weight += weight

        total = long_score + short_score + range_score + abstain_weight
        if total == 0.0:
            decision = ConsensusDecision.NO_TRADE
            confidence = 0.0
        else:
            long_ratio = long_score / total
            short_ratio = short_score / total
            range_ratio = range_score / total

            if long_ratio > self.config.min_consensus_threshold:
                decision = ConsensusDecision.LONG_BIAS
            elif short_ratio > self.config.min_consensus_threshold:
                decision = ConsensusDecision.SHORT_BIAS
            elif range_ratio > self.config.min_consensus_threshold:
                decision = ConsensusDecision.RANGE
            else:
                decision = ConsensusDecision.NO_TRADE

            confidence = max(long_score, short_score, range_score) / total

        # Build vote distribution
        vote_distribution = {
            VoteDirection.LONG: long_score / total if total > 0 else 0.0,
            VoteDirection.SHORT: short_score / total if total > 0 else 0.0,
            VoteDirection.RANGE: range_score / total if total > 0 else 0.0,
            VoteDirection.ABSTAIN: abstain_weight / total if total > 0 else 0.0,
        }

        # Build agent agreements / disagreements
        agent_weights = {}
        agent_agreements: list[str] = []
        agent_disagreements: list[str] = []

        for agent_id, vote, weight in agent_data:
            agent_weights[agent_id] = weight
            if self._vote_matches_decision(vote, decision):
                agent_agreements.append(agent_id)
            else:
                agent_disagreements.append(agent_id)

        # Build reason string
        total_weight = sum(agent_weights.values())
        agreed_weight = sum(
            w for aid, _, w in agent_data if aid in agent_agreements
        )
        reason = (
            f"{decision.value}: {len(agent_agreements)} agents agree "
            f"(weight {agreed_weight:.1f}/{total_weight:.1f})"
        )

        return ConsensusResult(
            decision=decision,
            vote_distribution=vote_distribution,
            agent_weights=agent_weights,
            agent_agreements=agent_agreements,
            agent_disagreements=agent_disagreements,
            confidence=confidence,
            reason=reason,
        )

    @staticmethod
    def _vote_matches_decision(vote: VoteDirection, decision: ConsensusDecision) -> bool:
        """Prüft, ob eine Stimme zur Entscheidung passt."""
        mapping = {
            ConsensusDecision.LONG_BIAS: VoteDirection.LONG,
            ConsensusDecision.SHORT_BIAS: VoteDirection.SHORT,
            ConsensusDecision.RANGE: VoteDirection.RANGE,
            ConsensusDecision.NO_TRADE: VoteDirection.ABSTAIN,
        }
        return vote == mapping.get(decision)
