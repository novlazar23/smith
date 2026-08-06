"""Blocking rules for the Governance Decision-Engine.

Determines *what* blocks a decision before the engine decides *which* final
decision type to emit.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .base import FinalDecisionData, FinalDecisionType, GovernanceConfig


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
            risk_approved,
            risk_veto,
            risk_blocking_reasons,
            num_active_agents,
            max_uncertainty,
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
            analysis_time=datetime.now(UTC),
            decision=decision,
            reason=reason,
            blocking_reasons=blockers,
        )
