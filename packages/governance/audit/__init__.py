"""Audit Trail — Statusübergänge, Entscheidungen, Reviews.

Audit Trail: Statusübergänge, Entscheidungen, Reviews.
Kill Criteria automatisiert auswertbar.
Rollback-Fähigkeit getestet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class AuditEntry:
    """Ein einzelner Audit-Trail-Eintrag."""

    entry_id: str
    event_type: str  # "state_transition", "decision", "review", "promotion", "quarantine"
    agent_id: str
    actor: str  # "system", "reviewer_id", "engine"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)
    previous_state: str | None = None
    new_state: str | None = None
    approved: bool | None = None
    rollback_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "actor": self.actor,
            "details": self.details,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "approved": self.approved,
        }


class AuditTrail:
    """Zentrales Audit-Logging für Governance-Operationen."""

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []
        self._counter = 0

    def log_state_transition(
        self,
        agent_id: str,
        previous_state: str,
        new_state: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        self._counter += 1
        entry = AuditEntry(
            entry_id=f"AUDIT-{self._counter:06d}",
            timestamp=datetime.now(UTC),
            event_type="state_transition",
            agent_id=agent_id,
            actor=actor,
            details=details or {},
            previous_state=previous_state,
            new_state=new_state,
        )
        self.entries.append(entry)
        return entry

    def log_decision(
        self,
        agent_id: str,
        decision: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        self._counter += 1
        entry = AuditEntry(
            entry_id=f"AUDIT-{self._counter:06d}",
            timestamp=datetime.now(UTC),
            event_type="decision",
            agent_id=agent_id,
            actor=actor,
            details=details or {},
            new_state=decision,
        )
        self.entries.append(entry)
        return entry

    def log_review(
        self,
        agent_id: str,
        reviewer: str,
        approved: bool,
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        self._counter += 1
        entry = AuditEntry(
            entry_id=f"AUDIT-{self._counter:06d}",
            timestamp=datetime.now(UTC),
            event_type="review",
            agent_id=agent_id,
            actor=reviewer,
            details=details or {},
            approved=approved,
        )
        self.entries.append(entry)
        return entry

    def log_promotion(
        self,
        agent_id: str,
        from_state: str,
        to_state: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        return self.log_state_transition(
            agent_id=agent_id,
            previous_state=from_state,
            new_state=to_state,
            actor=actor,
            details={"event": "promotion", **(details or {})},
        )

    def log_quarantine(
        self,
        agent_id: str,
        reason: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        return self.log_state_transition(
            agent_id=agent_id,
            previous_state="ACTIVE",
            new_state="QUARANTINED",
            actor=actor,
            details={"event": "quarantine", "reason": reason, **(details or {})},
        )

    def get_agent_history(self, agent_id: str) -> list[AuditEntry]:
        return [e for e in self.entries if e.agent_id == agent_id]

    @property
    def total_entries(self) -> int:
        return len(self.entries)
