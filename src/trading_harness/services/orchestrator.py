from __future__ import annotations

from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from trading_harness.models import (
    ALLOWED_TRANSITIONS,
    AuditEntry,
    RunOutcome,
    RunState,
    TradingRun,
    transition,
)


class TradingRunService:
    """Manages trading run lifecycle with audit trail.

    Uses the state machine defined in models. Run data is in-memory for MVP.
    """

    def __init__(self) -> None:
        self._runs: dict[str, TradingRun] = {}
        self._audit_log: list[AuditEntry] = []
        self._lock = RLock()

    def create(self, snapshot_id: str, run_id: str | None = None) -> TradingRun:
        run = TradingRun(snapshot_id=snapshot_id, id=run_id or f"run-{uuid4()}")
        with self._lock:
            self._runs[run.id] = run
            self._log("create", "trading_run", run.id, new_state=run.state)
        return run

    def get(self, run_id: str) -> TradingRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def all(self) -> list[TradingRun]:
        with self._lock:
            return list(self._runs.values())

    def transition(self, run_id: str, target: RunState, actor: str = "system") -> TradingRun:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            previous = run.state
            new_state = transition(run.state, target)
            run.state = new_state
            run.updated_at = datetime.now(UTC)
            self._log("transition", "trading_run", run_id, previous_state=previous, new_state=new_state, actor=actor)
            return run

    def add_decision(self, run_id: str, decision: dict[str, Any]) -> TradingRun:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            run.decisions.append(decision)
            return run

    def complete(self, run_id: str, outcome: RunOutcome, reason: str) -> TradingRun:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            previous = run.state
            run.state = RunState.COMPLETE
            run.outcome = outcome
            run.outcome_reason = reason
            run.updated_at = datetime.now(UTC)
            self._log("complete", "trading_run", run_id, previous_state=previous.value, new_state=outcome.value)
            return run

    def fail(self, run_id: str, error: str) -> TradingRun:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            run.state = RunState.FAILED
            run.error = error
            run.updated_at = datetime.now(UTC)
            self._log("fail", "trading_run", run_id, new_state="FAILED")
            return run

    def _log(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        *,
        actor: str = "system",
        previous_state: str | None = None,
        new_state: str | None = None,
    ) -> None:
        self._audit_log.append(
            AuditEntry(
                actor=actor,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                previous_state=previous_state,
                new_state=new_state,
            )
        )

    def audit(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        *,
        actor: str = "shadow-loop",
        previous_state: str | None = None,
        new_state: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Öffentlicher Audit-Eintrag mit optionalen Details (WI-ST-05, Spec ST.13).

        Additiv: bestehende ``_log``-Aufrufe bleiben unverändert; der Shadow-Loop
        nutzt diese Methode, weil ST.13 strukturierte ``details`` verlangt.
        """
        entry = AuditEntry(
            actor=actor,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            previous_state=previous_state,
            new_state=new_state,
            details=dict(details) if details else {},
        )
        with self._lock:
            self._audit_log.append(entry)
        return entry

    def get_audit_log(self, entity_id: str | None = None) -> list[AuditEntry]:
        with self._lock:
            if entity_id:
                return [e for e in self._audit_log if e.entity_id == entity_id]
            return list(self._audit_log)

    @property
    def state_machine(self) -> dict[RunState, set[RunState]]:
        return ALLOWED_TRANSITIONS
