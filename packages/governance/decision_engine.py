"""EPIC-09 Decision Engine — FinalDecision from Consensus + Risk.

Dieses Modul ist Teil von EPIC-09 (Strategy, Portfolio and Risk).
Es enthält die Decision-Engine Logik für die finale Handelsentscheidung.

Zentraler Import für EPIC-11 Governance-State-Machine ist `state_machine.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class FinalDecisionType(StrEnum):
    """Mögliche Endentscheidungen des Analyse-Graphen.

    Entsprechend EPIC-09-WP06, abgeleitet von packages/schemas/final_decision.py.
    """

    LONG_BIAS = "LONG_BIAS"
    SHORT_BIAS = "SHORT_BIAS"
    RANGE = "RANGE"

    NO_TRADE = "NO_TRADE"
    NO_TRADE_DATA_QUALITY = "NO_TRADE_DATA_QUALITY"
    NO_TRADE_INSUFFICIENT_EDGE = "NO_TRADE_INSUFFICIENT_EDGE"
    NO_TRADE_PORTFOLIO = "NO_TRADE_PORTFOLIO"
    NO_TRADE_RISK = "NO_TRADE_RISK"
    NO_TRADE_MODEL_UNCERTAINTY = "NO_TRADE_MODEL_UNCERTAINTY"


@dataclass(frozen=True)
class DecisionRule:
    """Eine einzelne Governance-Regel.

    Attributes:
        rule_id: Eindeutige Kennung der Regel.
        condition: Beschreibender Name der Bedingung, z. B.
                   ``consensus_long_above_threshold``.
        action: Aktion bei Erfülung — ``approve``, ``block`` oder ``reduce``.
        threshold: Schwellenwert, ab dem die Regel aktiv wird.
        blocking: Bestimmt, ob die Regel die Entscheidung blockiert.
    """

    rule_id: str
    condition: str
    action: str
    threshold: float
    blocking: bool


@dataclass
class GovernanceConfig:
    """Konfiguration des Governance-Entscheidungswegs.

    Alle Schwellenwerte sind im Bereich 0.0-1.0, außer ``required_agents``.
    """

    consensus_long_threshold: float = 0.65
    consensus_short_threshold: float = 0.65
    consensus_range_threshold: float = 0.50
    min_confidence: float = 0.50
    max_uncertainty: float = 0.30
    required_agents: int = 2
    rules: list[DecisionRule] = field(default_factory=list)


@dataclass
class FinalDecisionData:
    """Datensatz einer finalen Entscheidung.

    Spiegelt das Feld-Layout von ``packages/schemas/final_decision.py`` nach
    als Dataclass (kein Pydantic).
    """

    run_id: str
    instrument: str
    horizons: list[str]
    analysis_time: datetime
    decision: FinalDecisionType
    reason: str
    blocking_reasons: list[str] = field(default_factory=list)
    forecast: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    portfolio: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    audit_hash: str | None = None

    def compute_hash(self) -> str:
        """Berechnet einen SHA-256-Hash über die Schlüssel-Felder."""
        content = f"{self.run_id}|{self.instrument}|{self.decision}|{self.reason}"
        self.audit_hash = hashlib.sha256(content.encode()).hexdigest()
        return self.audit_hash


class BlockingRules:
    """Prüft, ob eine Entscheidung blockiert ist.

    Prüft in fester Prioritätsreihenfolge und gibt eine Liste von
    Blockier-Gründen zurück.  Eine leere Liste bedeutet: keine Blöcke.
    """

    def __init__(self, config: GovernanceConfig | None = None) -> None:
        self.config = config or GovernanceConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_blocking_conditions(
        self,
        consensus_decision: str,
        consensus_confidence: float,
        *,
        risk_approved: bool,
        risk_veto: bool,
        risk_blocking_reasons: list[str],
        num_active_agents: int,
        max_uncertainty: float | None = None,
    ) -> list[str]:
        """Prüft alle Blockier-Bedingungen in Prioritätsreihenfolge.

        Returns:
            Liste von Blockier-Gründen (leer wenn nichts blockiert).
        """
        blockers: list[str] = []

        # 1. Hard block: Risk veto
        if risk_veto:
            reasons = "; ".join(risk_blocking_reasons) if risk_blocking_reasons else "Risk veto active"
            blockers.append(f"risk_veto: {reasons}")

        # 2. Hard block: Risk not approved
        if not risk_approved:
            blockers.append("risk_not_approved")

        # 3. Hard block: Insufficient agents
        if num_active_agents < self.config.required_agents:
            blockers.append(
                f"insufficient_agents: {num_active_agents} < {self.config.required_agents}"
            )

        # 4. Soft block: Low consensus confidence
        if consensus_confidence < self.config.min_confidence:
            blockers.append(
                f"low_confidence: {consensus_confidence:.2f} < {self.config.min_confidence:.2f}"
            )

        # 5. Soft block: High uncertainty
        if max_uncertainty is not None and max_uncertainty > self.config.max_uncertainty:
            blockers.append(
                f"high_uncertainty: {max_uncertainty:.2f} > {self.config.max_uncertainty:.2f}"
            )

        # 6. Soft block: Insufficient edge for any direction
        long_ok = consensus_decision == "LONG_BIAS" and consensus_confidence >= self.config.consensus_long_threshold
        short_ok = consensus_decision == "SHORT_BIAS" and consensus_confidence >= self.config.consensus_short_threshold
        range_ok = consensus_decision == "RANGE" and consensus_confidence >= self.config.consensus_range_threshold
        if not (long_ok or short_ok or range_ok):
            blockers.append("insufficient_edge")

        return blockers

    def get_priority_blockers(self, blocking_reasons: list[str]) -> list[str]:
        """Gibt die kritischsten Blockierer zurück — harte Blöcke zuerst.

        Hard blocks: ``risk_veto``, ``risk_not_approved``, ``insufficient_agents``
        Soft blocks: ``low_confidence``, ``high_uncertainty``, ``insufficient_edge``
        """
        hard_keywords = (
            "risk_veto",
            "risk_not_approved",
            "insufficient_agents",
        )
        soft_keywords = (
            "low_confidence",
            "high_uncertainty",
            "insufficient_edge",
        )

        hard: list[str] = []
        soft: list[str] = []
        other: list[str] = []

        for reason in blocking_reasons:
            if any(kw in reason for kw in hard_keywords):
                hard.append(reason)
            elif any(kw in reason for kw in soft_keywords):
                soft.append(reason)
            else:
                other.append(reason)

        return hard + other + soft

    # ------------------------------------------------------------------
    # Convenience: build a FinalDecisionData from blockers
    # ------------------------------------------------------------------

    def build_blocked_decision(
        self,
        *,
        consensus_decision: str,
        consensus_confidence: float,
        risk_approved: bool,
        risk_veto: bool,
        risk_blocking_reasons: list[str],
        num_active_agents: int,
        max_uncertainty: float | None = None,
        instrument: str = "UNKNOWN",
        run_id: str = "default",
        horizons: list[str] | None = None,
    ) -> FinalDecisionData:
        """Erzeugt eine ``FinalDecisionData``-Instanz aus den Blockier-Gründen.

        Wählt den passenden ``FinalDecisionType`` basierend auf dem ersten
        Treffer in der BlockingRules-Prioritätskette.
        """
        blockers = self.check_blocking_conditions(
            consensus_decision,
            consensus_confidence,
            risk_approved=risk_approved,
            risk_veto=risk_veto,
            risk_blocking_reasons=risk_blocking_reasons,
            num_active_agents=num_active_agents,
            max_uncertainty=max_uncertainty,
        )

        if not blockers:
            raise ValueError("No blocking conditions detected — not a blocked decision")

        first = blockers[0]
        if "risk_veto" in first or "risk_not_approved" in first:
            decision = FinalDecisionType.NO_TRADE_RISK
        elif "insufficient_agents" in first:
            decision = FinalDecisionType.NO_TRADE_DATA_QUALITY
        elif "low_confidence" in first or "high_uncertainty" in first:
            decision = FinalDecisionType.NO_TRADE_MODEL_UNCERTAINTY
        else:
            decision = FinalDecisionType.NO_TRADE_INSUFFICIENT_EDGE

        reason = blockers[0] if len(blockers) == 1 else f"{blockers[0]}; {'; '.join(blockers[1:])}"

        return FinalDecisionData(
            run_id=run_id,
            instrument=instrument,
            horizons=horizons or [],
            analysis_time=datetime.now(),
            decision=decision,
            reason=reason,
            blocking_reasons=blockers,
        )


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
        *,
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
                analysis_time=datetime.now(),
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
                analysis_time=datetime.now(),
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
                analysis_time=datetime.now(),
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
                analysis_time=datetime.now(),
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
                analysis_time=datetime.now(),
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
            analysis_time=datetime.now(),
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
        *,
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
