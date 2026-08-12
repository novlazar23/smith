"""Champion-Challenger — Versionierung, Evaluation, Promotion.

Promotion: bessere OOS-Kalibrierung, gleiche Stabilität, positiver marginaler Nutzen.
Keine neuen krit. Risiken, erfolgreicher Shadow-Betrieb.
Jede Variante: agent_id, champion_version, challenger_version, evaluation_window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class ChampionChallengerConfig:
    """Konfiguration für das Champion-Challenger-System."""

    min_oos_improvement: float = 0.02
    max_stability_degradation: float = 0.05
    shadow_evaluation_days: int = 10
    min_evaluation_samples: int = 50


@dataclass
class AgentVersion:
    """Eine Version eines Agents mit Evaluationsmetriken."""

    agent_id: str
    version: str
    oos_score: float = 0.0
    calibration_score: float = 0.0
    stability_score: float = 1.0
    marginal_contribution: float = 0.0
    shadow_days: int = 0
    samples: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_champion(self) -> bool:
        return self.version == "champion"

    @property
    def is_challenger(self) -> bool:
        return self.version == "challenger"


@dataclass
class EvaluationResult:
    """Ergebnis eines Champion-Challenger-Vergleichs."""

    agent_id: str
    champion_version: str
    challenger_version: str
    evaluation_window_start: datetime
    evaluation_window_end: datetime
    champion_oos: float
    challenger_oos: float
    champion_stability: float
    challenger_stability: float
    champion_marginal: float
    challenger_marginal: float
    champion_new_risks: list[str] = field(default_factory=list)
    champion_new_risks_empty: bool = True
    shadow_success: bool = True
    promoted: bool = False
    promotion_reason: str = ""


class ChampionChallengerEngine:
    """Vergleicht Champion und Challenger evaluiert Promotion."""

    def __init__(self, config: ChampionChallengerConfig | None = None) -> None:
        self.config = config or ChampionChallengerConfig()

    def evaluate(
        self,
        agent_id: str,
        champion: AgentVersion,
        challenger: AgentVersion,
        champion_new_risks: list[str] | None = None,
        shadow_success: bool = True,
    ) -> EvaluationResult:
        """Vergleicht Champion und Challenger und evaluiert Promotion.

        Promotion-Regeln:
        1. Bessere OOS-Kalibrierung (min_oos_improvement)
        2. Gleiche/niedrigere Stabilitätsdegradation (max_stability_degradation)
        3. Positiver marginaler Nutzen
        4. Keine neuen krit. Risiken
        5. Erfolgreicher Shadow-Betrieb
        """
        champion_new_risks = champion_new_risks or []

        oos_improvement = challenger.oos_score - champion.oos_score
        stability_degradation = champion.stability_score - challenger.stability_score

        promoted = (
            oos_improvement >= self.config.min_oos_improvement
            and stability_degradation <= self.config.max_stability_degradation
            and challenger.marginal_contribution > 0
            and len(champion_new_risks) == 0
            and shadow_success
        )

        reason_parts: list[str] = []
        if promoted:
            reason_parts.append(
                f"OOS improved: {champion.oos_score:.2f} → {challenger.oos_score:.2f} "
                f"(+{oos_improvement:.2f})"
            )
            if stability_degradation > 0:
                reason_parts.append(
                    f"Stability degraded by {stability_degradation:.2f} ≤ {self.config.max_stability_degradation:.2f}"
                )
            reason_parts.append(f"Marginal contribution: {challenger.marginal_contribution:.4f} > 0")
            reason_parts.append("No new critical risks")
            reason_parts.append("Shadow operation successful")
        else:
            if oos_improvement < self.config.min_oos_improvement:
                reason_parts.append(
                    f"OOS improvement {oos_improvement:.2f} < "
                    f"{self.config.min_oos_improvement:.2f}"
                )
            if stability_degradation > self.config.max_stability_degradation:
                reason_parts.append(
                    f"Stability degradation {stability_degradation:.2f} > "
                    f"{self.config.max_stability_degradation:.2f}"
                )
            if challenger.marginal_contribution <= 0:
                reason_parts.append(
                    f"Marginal contribution {challenger.marginal_contribution:.4f} ≤ 0"
                )
            if champion_new_risks:
                reason_parts.append(f"New risks: {', '.join(champion_new_risks)}")
            if not shadow_success:
                reason_parts.append("Shadow operation not successful")

        return EvaluationResult(
            agent_id=agent_id,
            champion_version=champion.version,
            challenger_version=challenger.version,
            evaluation_window_start=challenger.created_at,
            evaluation_window_end=datetime.now(UTC),
            champion_oos=champion.oos_score,
            challenger_oos=challenger.oos_score,
            champion_stability=champion.stability_score,
            challenger_stability=challenger.stability_score,
            champion_marginal=champion.marginal_contribution,
            challenger_marginal=challenger.marginal_contribution,
            champion_new_risks=champion_new_risks,
            champion_new_risks_empty=len(champion_new_risks) == 0,
            shadow_success=shadow_success,
            promoted=promoted,
            promotion_reason="; ".join(reason_parts),
        )
