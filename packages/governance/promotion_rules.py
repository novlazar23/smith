"""Promotion Rules — Shadow Mode, Kalibrierung, marginaler Nutzen.

Neue Agenten starten immer als SHADOW → final_weight == 0.0 im Consensus.
Aktivierung nur bei: Mindest-OOS, Kalibrierungsgrenze, positiver marginaler Nutzen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class AgentState(StrEnum):
    """Lebenszyklus-Status eines Agents."""

    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    QUARANTINED = "QUARANTINED"
    DISABLED = "DISABLED"


@dataclass
class ShadowAgent:
    """Ein Agent im Shadow-Modus — wird im Consensus nicht gewichtet."""

    agent_id: str
    version: str
    calibration_score: float = 0.0
    oos_score: float = 0.0
    marginal_contribution: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    state: AgentState = field(default=AgentState.SHADOW)

    @property
    def consensus_weight(self) -> float:
        """Shadow-Agents haben kein Gewicht im Consensus."""
        return 0.0


@dataclass
class ActiveAgent:
    """Ein aktiver Agent mit gewichtetem Consensus-Einfluss."""

    agent_id: str
    version: str
    calibration_score: float = 1.0
    oos_score: float = 0.0
    marginal_contribution: float = 0.0
    last_calibration: datetime = field(default_factory=lambda: datetime.now(UTC))
    state: AgentState = field(default=AgentState.ACTIVE)

    @property
    def consensus_weight(self) -> float:
        """Aktive Agents haben volles Gewicht."""
        return 1.0

    @property
    def is_degraded(self) -> bool:
        return self.state == AgentState.DEGRADED


@dataclass
class DegradedAgent(ActiveAgent):
    """Ein degradierter Agent — reduziertes Gewicht."""

    state: AgentState = field(default=AgentState.DEGRADED)

    @property
    def consensus_weight(self) -> float:
        return 0.5


@dataclass
class PromotionCriteria:
    """Kriterien für die Promotion von SHADOW → ACTIVE."""

    min_oos_score: float = 0.60
    min_calibration_score: float = 0.70
    min_marginal_contribution: float = 0.01
    min_shadow_days: int = 5
    required_reviews: int = 1


class PromotionRuleEngine:
    """Evaluiert, ob ein SHADOW-Agent aktiviert werden darf."""

    def __init__(self, criteria: PromotionCriteria | None = None) -> None:
        self.criteria = criteria or PromotionCriteria()

    def evaluate_promotion(
        self, agent: ShadowAgent, review_status: str = "approved"
    ) -> tuple[bool, list[str]]:
        """Prüft, ob ein Shadow-Agent promoted werden darf.

        Returns:
            (can_promote, reasons) — reasons enthält alle erfüllten/gefallenen Kriterien.
        """
        reasons: list[str] = []

        if agent.oos_score < self.criteria.min_oos_score:
            reasons.append(
                f"oos_score {agent.oos_score:.2f} < "
                f"min {self.criteria.min_oos_score:.2f}"
            )
        else:
            reasons.append(f"oos_score {agent.oos_score:.2f} >= {self.criteria.min_oos_score:.2f} ✅")

        if agent.calibration_score < self.criteria.min_calibration_score:
            reasons.append(
                f"calibration_score {agent.calibration_score:.2f} < "
                f"min {self.criteria.min_calibration_score:.2f}"
            )
        else:
            reasons.append(f"calibration_score {agent.calibration_score:.2f} >= {self.criteria.min_calibration_score:.2f} ✅")

        if agent.marginal_contribution < self.criteria.min_marginal_contribution:
            reasons.append(
                f"marginal_contribution {agent.marginal_contribution:.4f} < "
                f"min {self.criteria.min_marginal_contribution:.4f}"
            )
        else:
            reasons.append(f"marginal_contribution {agent.marginal_contribution:.4f} >= {self.criteria.min_marginal_contribution:.4f} ✅")

        if review_status != "approved":
            reasons.append(f"Review-Status: {review_status} (erfordert: approved)")
        else:
            reasons.append("Review: approved ✅")

        can_promote = (
            agent.oos_score >= self.criteria.min_oos_score
            and agent.calibration_score >= self.criteria.min_calibration_score
            and agent.marginal_contribution >= self.criteria.min_marginal_contribution
            and review_status == "approved"
        )

        return can_promote, reasons
