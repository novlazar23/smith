"""DecisionEngine — evaluates consensus + risk → FinalDecisionData.

The engine follows a strict priority chain:

    risk_veto > risk_not_approved > insufficient_agents >
    low_confidence > high_uncertainty > insufficient_edge > approve
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .base import (
    DecisionRule,
    FinalDecisionData,
    FinalDecisionType,
    GovernanceConfig,
)
from .blocking import BlockingRules


class DecisionEngine:
    """Zentrale Engine, die Konsens- und Risikoeingaben zu einer finalen
    Entscheidung kombiniert.

    Parameters:
        config: Optional ``GovernanceConfig``.  Bei ``None`` wird die
                Standardkonfiguration verwendet.
    """

    def __init__(self, config: GovernanceConfig | None = None) -> None:
        self.config = config or GovernanceConfig()
        self.blocking = BlockingRules(self.config)

    # ------------------------------------------------------------------
    # Core decision method
    # ------------------------------------------------------------------

    def evaluate(
        self,
        consensus_decision: str,
        consensus_confidence: float,
        consensus_vote_distribution: dict[str, float],
        risk_approved: bool,
        risk_veto: bool,
        risk_reduction_factor: float,
        risk_blocking_reasons: list[str],
        num_active_agents: int,
        max_uncertainty: float | None = None,
        instrument: str = "UNKNOWN",
        run_id: str = "default",
    ) -> FinalDecisionData:
        """Bewertet alle Eingaben und erzeugt eine finale Entscheidung.

        Parameters:
            consensus_decision: Eine der vier Konsens-Entscheidungen
                                (``"LONG_BIAS"``, ``"SHORT_BIAS"``,
                                ``"RANGE"``, ``"NO_TRADE"``).
            consensus_confidence: Float im Bereich 0.0-1.0.
            consensus_vote_distribution: Verteilung der Agenten-Stimmen, z. B.
                                         ``{"up": 0.6, "down": 0.2, "range": 0.2}``.
            risk_approved: Gibt an, ob das Risk-Modul die Entscheidung freigegeben hat.
            risk_veto: True, wenn das Risk-Modul ein hartes Veto ausübt.
            risk_reduction_factor: Reduktionsfaktor des Risikomoduls (0.0-1.0).
            risk_blocking_reasons: Liste der Begründungen für Risk-Blöcke.
            num_active_agents: Anzahl aktiv beteiligter Agenten.
            max_uncertainty: Optionale Überschreibung des Maximal-Undo-Werts.
            instrument: Das gehandelte Instrument.
            run_id: Kennung des aktuellen Laufs.

        Returns:
            Eine ``FinalDecisionData``-Instanz mit der finalen Entscheidung.
        """
        # --- Hard blocks (priority order) ---
        if risk_veto:
            reason = (
                "; ".join(risk_blocking_reasons)
                if risk_blocking_reasons
                else "Risk veto active"
            )
            return FinalDecisionData(
                run_id=run_id,
                instrument=instrument,
                horizons=list(consensus_vote_distribution.keys()),
                analysis_time=datetime.now(UTC),
                decision=FinalDecisionType.NO_TRADE_RISK,
                reason=reason,
                blocking_reasons=["risk_veto: " + reason] if risk_blocking_reasons else ["risk_veto"],
                forecast={"vote_distribution": consensus_vote_distribution},
                risk={"approved": risk_approved, "veto": risk_veto, "reduction_factor": risk_reduction_factor},
            )

        if num_active_agents < self.config.required_agents:
            return FinalDecisionData(
                run_id=run_id,
                instrument=instrument,
                horizons=list(consensus_vote_distribution.keys()),
                analysis_time=datetime.now(UTC),
                decision=FinalDecisionType.NO_TRADE_DATA_QUALITY,
                reason=(
                    f"Insufficient agents: {num_active_agents} < {self.config.required_agents} required"
                ),
                blocking_reasons=[
                    f"insufficient_agents: {num_active_agents} < {self.config.required_agents}"
                ],
                forecast={"vote_distribution": consensus_vote_distribution},
                risk={"approved": risk_approved, "veto": risk_veto, "reduction_factor": risk_reduction_factor},
            )

        if not risk_approved:
            reason = (
                "; ".join(risk_blocking_reasons)
                if risk_blocking_reasons
                else "Risk not approved"
            )
            return FinalDecisionData(
                run_id=run_id,
                instrument=instrument,
                horizons=list(consensus_vote_distribution.keys()),
                analysis_time=datetime.now(UTC),
                decision=FinalDecisionType.NO_TRADE_RISK,
                reason=reason,
                blocking_reasons=["risk_not_approved"],
                forecast={"vote_distribution": consensus_vote_distribution},
                risk={"approved": risk_approved, "veto": risk_veto, "reduction_factor": risk_reduction_factor, "reasons": risk_blocking_reasons},
            )

        # --- Soft blocks ---
        if consensus_confidence < self.config.min_confidence:
            return FinalDecisionData(
                run_id=run_id,
                instrument=instrument,
                horizons=list(consensus_vote_distribution.keys()),
                analysis_time=datetime.now(UTC),
                decision=FinalDecisionType.NO_TRADE_MODEL_UNCERTAINTY,
                reason=(
                    f"Model uncertainty: confidence {consensus_confidence:.2f} "
                    f"below minimum {self.config.min_confidence:.2f}"
                ),
                blocking_reasons=[
                    f"low_confidence: {consensus_confidence:.2f} < {self.config.min_confidence:.2f}"
                ],
                forecast={"vote_distribution": consensus_vote_distribution, "confidence": consensus_confidence},
                risk={"approved": risk_approved, "veto": risk_veto, "reduction_factor": risk_reduction_factor},
            )

        effective_max_uncertainty = max_uncertainty if max_uncertainty is not None else 0.0
        if effective_max_uncertainty > self.config.max_uncertainty:
            return FinalDecisionData(
                run_id=run_id,
                instrument=instrument,
                horizons=list(consensus_vote_distribution.keys()),
                analysis_time=datetime.now(UTC),
                decision=FinalDecisionType.NO_TRADE_MODEL_UNCERTAINTY,
                reason=(
                    f"Model uncertainty: max_uncertainty {effective_max_uncertainty:.2f} "
                    f"above maximum {self.config.max_uncertainty:.2f}"
                ),
                blocking_reasons=[
                    f"high_uncertainty: {effective_max_uncertainty:.2f} > {self.config.max_uncertainty:.2f}"
                ],
                forecast={"vote_distribution": consensus_vote_distribution, "confidence": consensus_confidence},
                risk={"approved": risk_approved, "veto": risk_veto, "reduction_factor": risk_reduction_factor},
            )

        # --- Consensus threshold check ---
        decision = self._resolve_consensus(consensus_decision, consensus_confidence)

        reason = self._build_reason(decision, consensus_decision, consensus_confidence)

        return FinalDecisionData(
            run_id=run_id,
            instrument=instrument,
            horizons=list(consensus_vote_distribution.keys()),
            analysis_time=datetime.now(UTC),
            decision=decision,
            reason=reason,
            forecast={"vote_distribution": consensus_vote_distribution, "confidence": consensus_confidence},
            risk={"approved": risk_approved, "veto": risk_veto, "reduction_factor": risk_reduction_factor},
        )

    # ------------------------------------------------------------------
    # evaluate_with_rules
    # ------------------------------------------------------------------

    def evaluate_with_rules(
        self,
        consensus_decision: str,
        consensus_confidence: float,
        consensus_vote_distribution: dict[str, float],
        risk_approved: bool,
        risk_veto: bool,
        risk_reduction_factor: float,
        risk_blocking_reasons: list[str],
        num_active_agents: int,
        max_uncertainty: float | None = None,
        instrument: str = "UNKNOWN",
        run_id: str = "default",
    ) -> tuple[FinalDecisionData, list[str]]:
        """Wie ``evaluate``, aber gibt zusätzlich eine Liste der getrackten
        Regel-Treffer zurück.

        Returns:
            Tupel aus ``FinalDecisionData`` und Liste der matchenden Regel-IDs.
        """
        result = self.evaluate(
            consensus_decision=consensus_decision,
            consensus_confidence=consensus_confidence,
            consensus_vote_distribution=consensus_vote_distribution,
            risk_approved=risk_approved,
            risk_veto=risk_veto,
            risk_reduction_factor=risk_reduction_factor,
            risk_blocking_reasons=risk_blocking_reasons,
            num_active_agents=num_active_agents,
            max_uncertainty=max_uncertainty,
            instrument=instrument,
            run_id=run_id,
        )

        rule_hits: list[str] = []
        for rule in self.config.rules:
            hit = self._check_rule(rule, consensus_decision, consensus_confidence)
            if hit:
                rule_hits.append(rule.rule_id)
                if rule.blocking:
                    result.blocking_reasons.append(
                        f"rule_block: {rule.rule_id} ({rule.condition})"
                    )

        return result, rule_hits

    # ------------------------------------------------------------------
    # Decision details
    # ------------------------------------------------------------------

    def get_decision_details(self, result: FinalDecisionData) -> dict[str, Any]:
        """Gibt ein detailliertes Wörterbuch der Entscheidung zurück.

        Enthält Entscheidung, Grund, Blockier-Gründe, Konfidenz und Zeitstempel.
        """
        confidence = result.forecast.get("confidence", None)
        return {
            "decision": result.decision,
            "reason": result.reason,
            "blocking_reasons": result.blocking_reasons,
            "confidence": confidence,
            "timestamp": result.analysis_time,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_consensus(
        self, consensus_decision: str, consensus_confidence: float
    ) -> FinalDecisionType:
        """Löst Konsens-Entscheidung anhand der Schwellenwerte auf."""
        if consensus_decision == "LONG_BIAS" and consensus_confidence >= self.config.consensus_long_threshold:
            return FinalDecisionType.LONG_BIAS

        if consensus_decision == "SHORT_BIAS" and consensus_confidence >= self.config.consensus_short_threshold:
            return FinalDecisionType.SHORT_BIAS

        if consensus_decision == "RANGE" and consensus_confidence >= self.config.consensus_range_threshold:
            return FinalDecisionType.RANGE

        return FinalDecisionType.NO_TRADE_INSUFFICIENT_EDGE

    def _build_reason(
        self, decision: FinalDecisionType,
        consensus_decision: str,
        consensus_confidence: float,
    ) -> str:
        """Erzeugt einen leserlichen Grund-String für die Entscheidung."""
        if decision == FinalDecisionType.LONG_BIAS:
            return f"Long bias approved: consensus={consensus_confidence:.2f}"
        if decision == FinalDecisionType.SHORT_BIAS:
            return f"Short bias approved: consensus={consensus_confidence:.2f}"
        if decision == FinalDecisionType.RANGE:
            return f"Range bias approved: consensus={consensus_confidence:.2f}"
        return f"Insufficient edge: {consensus_decision} at confidence {consensus_confidence:.2f}"

    def _check_rule(
        self, rule: DecisionRule,
        consensus_decision: str,
        consensus_confidence: float,
    ) -> bool:
        """Prüft, ob eine einzelne ``DecisionRule`` greift.

        Unterstützt die drei vordefinierten Bedingungen; alle anderen
        Bedingungen werden als True interpretiert (implizite Erlaubnis).
        """
        if rule.condition == "consensus_long_above_threshold":
            return consensus_decision == "LONG_BIAS" and consensus_confidence >= rule.threshold
        if rule.condition == "consensus_short_above_threshold":
            return consensus_decision == "SHORT_BIAS" and consensus_confidence >= rule.threshold
        if rule.condition == "consensus_range_above_threshold":
            return consensus_decision == "RANGE" and consensus_confidence >= rule.threshold
        # Default: if we can't evaluate, assume rule triggers (custom logic)
        return False
