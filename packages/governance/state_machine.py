"""Governance State Machine — Agent Lifecycle Management.

State transitions:
    SHADOW → ACTIVE (promotion)
    ACTIVE ↔ DEGRADED (performance degradation/recovery)
    ACTIVE → QUARANTINED (critical failure)
    QUARANTINED → SHADOW (review & reset)
    QUARANTINED → DISABLED (permanent removal)
    DISABLED → SHADOW (re-initialization)

State weights for consensus:
    ACTIVE:   1.0 (full weight)
    DEGRADED: 0.25-0.75 (configurable, based on severity)
    SHADOW:   0.0 (no influence)
    QUARANTINED: 0.0 (no influence)
    DISABLED: 0.0 (no influence)

All transitions are logged in the AuditTrail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from .audit import AuditTrail


class AgentState(StrEnum):
    """Lebenszyklus-Status eines Governance-Agents."""

    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    QUARANTINED = "QUARANTINED"
    DISABLED = "DISABLED"


@dataclass
class StateTransition:
    """Ein einzelner Statusübergang."""

    agent_id: str
    from_state: AgentState
    to_state: AgentState
    reason: str
    actor: str  # "system", "reviewer_id"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


# State weights — zentral definiert für konsistente Gewichtung im Consensus
STATE_WEIGHTS: dict[AgentState, float] = {
    AgentState.ACTIVE: 1.0,
    AgentState.DEGRADED: 0.5,
    AgentState.SHADOW: 0.0,
    AgentState.QUARANTINED: 0.0,
    AgentState.DISABLED: 0.0,
}

# Zulässige Übergänge — definiert die erlaubte Transition-Matrix
VALID_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.SHADOW: frozenset({AgentState.ACTIVE, AgentState.DISABLED}),
    AgentState.ACTIVE: frozenset({AgentState.DEGRADED, AgentState.QUARANTINED, AgentState.DISABLED}),
    AgentState.DEGRADED: frozenset({AgentState.ACTIVE, AgentState.QUARANTINED, AgentState.SHADOW}),
    AgentState.QUARANTINED: frozenset({AgentState.SHADOW, AgentState.DISABLED}),
    AgentState.DISABLED: frozenset({AgentState.SHADOW}),
}


@dataclass
class AgentRecord:
    """Verwaltungszustand eines einzelnen Agents."""

    agent_id: str
    state: AgentState = field(default=AgentState.SHADOW)
    version: str = "0.0.0"
    degraded_weight: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    history: list[StateTransition] = field(default_factory=list)

    @property
    def consensus_weight(self) -> float:
        """Berechnet das Consensus-Gewicht basierend auf dem aktuellen Status."""
        base = STATE_WEIGHTS.get(self.state, 0.0)
        # Bei DEGRADED: benutzerdefiniertes Gewicht verwenden
        if self.state == AgentState.DEGRADED:
            return self.degraded_weight
        return base


class GovernanceStateMachine:
    """Steuerung des Agenten-Lebenszyklus mit Audit-Trail.

    Parameters:
        audit_trail: Optionaler AuditTrail. Wird intern erstellt, wenn keiner
                     übergeben wird.
    """

    def __init__(self, audit_trail: AuditTrail | None = None) -> None:
        self.audit = audit_trail or AuditTrail()
        self._agents: dict[str, AgentRecord] = {}

    def register(self, agent_id: str, version: str = "0.0.0") -> AgentRecord:
        """Registriert einen neuen Agent im SHADOW-Status."""
        record = AgentRecord(agent_id=agent_id, version=version)
        self._agents[agent_id] = record
        self.audit.log_state_transition(
            agent_id=agent_id,
            previous_state=None,
            new_state=AgentState.SHADOW,
            actor="system",
            details={"reason": "initial_registration", "version": version},
        )
        return record

    def transition(
        self,
        agent_id: str,
        to_state: AgentState,
        reason: str,
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> StateTransition:
        """Führt einen Statusübergang durch, wenn er zulässig ist.

        Raises:
            ValueError: Wenn der Übergang nicht in VALID_TRANSITIONS erlaubt ist.
            KeyError: Wenn der Agent nicht registriert ist.
        """
        record = self._agents[agent_id]
        from_state = record.state

        allowed = VALID_TRANSITIONS.get(from_state)
        if to_state not in allowed:
            raise ValueError(
                f"Transition {from_state} → {to_state} is not allowed "
                f"for agent {agent_id}. Allowed: {allowed}"
            )

        # DEGRADED → gewicht konfigurieren
        if to_state == AgentState.DEGRADED and metadata:
            record.degraded_weight = metadata.get("weight", 0.5)

        transition = StateTransition(
            agent_id=agent_id,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            actor=actor,
            metadata=metadata or {},
        )

        record.state = to_state
        record.updated_at = transition.timestamp
        record.history.append(transition)

        self.audit.log_state_transition(
            agent_id=agent_id,
            previous_state=from_state,
            new_state=to_state,
            actor=actor,
            details={
                "reason": reason,
                "from_state": from_state,
                "version": record.version,
                **(metadata or {}),
            },
        )

        return transition

    def can_transition(self, agent_id: str, to_state: AgentState) -> bool:
        """Prüft, ob ein Übergang zulässig ist (ohne ihn auszuführen)."""
        record = self._agents.get(agent_id)
        if record is None:
            return False
        allowed = VALID_TRANSITIONS.get(record.state)
        return to_state in allowed

    def get_agent(self, agent_id: str) -> AgentRecord | None:
        """Liefert den Record eines Agents oder None."""
        return self._agents.get(agent_id)

    def get_all_agents(self) -> dict[str, AgentRecord]:
        """Liefert alle registrierten Agents."""
        return dict(self._agents)

    def get_state_counts(self) -> dict[AgentState, int]:
        """Zählt Agents nach Status."""
        counts: dict[AgentState, int] = dict.fromkeys(AgentState, 0)
        for record in self._agents.values():
            counts[record.state] += 1
        return counts

    def get_consensus_weights(self, agent_ids: list[str]) -> dict[str, float]:
        """Berechnet Consensus-Gewichte für eine Liste von Agent-IDs."""
        return {
            aid: self._agents[aid].consensus_weight
            for aid in agent_ids
            if aid in self._agents
        }
