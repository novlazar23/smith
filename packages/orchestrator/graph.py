"""TradingGraphState, PipelineStage, StageManager — Package-level Graph.

Gegenueber apps/orchestrator/ ist dieses Paket eine eigenstaendige
Implementierung mit vereinfachter Stage-Sequenz:

  REQUEST -> FIRST_ROUND -> SEAL -> SECOND_ROUND -> CONSENSUS -> DECISION -> PUBLISH

Jede Stage wird erst befuellt, wenn die vorherige abgeschlossen ist.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from packages.schemas.agent_report import AgentReport

# ── Stages Enum ───────────────────────────────────────────────────

class PipelineStage(StrEnum):
    """Sieben Stufen der Konsens-Pipeline.

    REQUEST -> FIRST_ROUND -> SEAL -> SECOND_ROUND ->
    CONSENSUS -> DECISION -> PUBLISH
    """

    REQUEST = "request"
    FIRST_ROUND = "first_round"
    SEAL = "seal"
    SECOND_ROUND = "second_round"
    CONSENSUS = "consensus"
    DECISION = "decision"
    PUBLISH = "publish"


# ── TradingGraphState ─────────────────────────────────────────────

@dataclass(frozen=True)
class TradingGraphState:
    """Zustand der Konsens-Pipeline mit allen Feldern.

    Jede Stufe wird erst befuellt, wenn die vorherige abgeschlossen ist.
    """

    run_id: str
    instrument: str
    market_snapshot: dict[str, Any] = field(default_factory=dict)
    regime_report: dict[str, Any] | None = None
    feature_snapshot_id: str | None = None
    first_round_reports: list[AgentReport] = field(default_factory=list)
    first_round_hash: str = ""
    seal_records: list[dict[str, Any]] = field(default_factory=list)
    second_round_reports: list[AgentReport] = field(default_factory=list)
    round_summary: dict[str, Any] = field(default_factory=dict)
    consensus_result: dict[str, Any] = field(default_factory=dict)
    decision: str = ""
    current_stage: str = PipelineStage.REQUEST
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None


# ── AuditEvent ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuditEvent:
    """Ein unveraenderbares Audit-Event pro Stage-Uebergang."""

    stage: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    event_id: str = ""

    def __post_init__(self) -> None:
        if not self.event_id:
            raw = f"{self.stage}:{self.timestamp.isoformat()}"
            object.__setattr__(
                self, "event_id", hashlib.sha256(raw.encode()).hexdigest()[:16]
            )


# ── OrchestratorGraph (StageManager) ───────────────────────────────

VALID_TRANSITIONS = {
    PipelineStage.REQUEST: frozenset({PipelineStage.FIRST_ROUND}),
    PipelineStage.FIRST_ROUND: frozenset({PipelineStage.SEAL}),
    PipelineStage.SEAL: frozenset({PipelineStage.SECOND_ROUND}),
    PipelineStage.SECOND_ROUND: frozenset({PipelineStage.CONSENSUS}),
    PipelineStage.CONSENSUS: frozenset({PipelineStage.DECISION}),
    PipelineStage.DECISION: frozenset({PipelineStage.PUBLISH}),
    PipelineStage.PUBLISH: frozenset(),
}


class OrchestratorGraph:
    """Verwaltet current_stage, AuditEvent-Liste und Transition-Logik."""

    def __init__(self) -> None:
        self._current_stage: PipelineStage | None = None
        self._audit_events: list[AuditEvent] = []
        self._transition_log: list[tuple[PipelineStage | None, PipelineStage]] = []

    @property
    def current_stage(self) -> PipelineStage | None:
        """Aktuelle Stage oder None."""
        return self._current_stage

    @property
    def audit_events(self) -> tuple[AuditEvent, ...]:
        """Audit-Event-Liste (unveränderbar)."""
        return tuple(self._audit_events)

    @property
    def transition_log(self) -> list[tuple[PipelineStage | None, PipelineStage]]:
        """Log aller Stage-Uebergange."""
        return list(self._transition_log)

    def stage_count(self) -> int:
        """Anzahl der abgeschlossenen Stages."""
        return len(self._audit_events)

    def has_completed(self, stage: PipelineStage) -> bool:
        """True wenn die Stage bereits durchlaufen wurde."""
        return any(e.stage == stage.value for e in self._audit_events)

    def can_transition(self, target: PipelineStage) -> bool:
        """Prueft ob der Uebergang zur Ziel-stage erlaubt ist."""
        if self._current_stage is None:
            return target == PipelineStage.REQUEST
        allowed = VALID_TRANSITIONS.get(self._current_stage, frozenset())
        return target in allowed

    def transition(
        self,
        target: PipelineStage,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> AuditEvent:
        """Fuehrt einen Stage-Uebergang durch und erstellt ein Audit-Event.

        Raises:
            ValueError: Wenn der Uebergang nicht erlaubt ist.
        """
        if not self.can_transition(target):
            expected = (
                PipelineStage.REQUEST
                if self._current_stage is None
                else self._current_stage
            )
            raise ValueError(
                f"Ungueltiger Uebergang von '{self._current_stage.value if self._current_stage else 'none'}' "
                f"zu '{target.value}'. Erwarte '{expected.value}'."
            )

        now = datetime.now(UTC)
        event = AuditEvent(
            stage=target.value,
            timestamp=now,
            inputs=inputs or {},
            outputs=outputs or {},
            duration_ms=duration_ms,
        )

        self._transition_log.append((self._current_stage, target))
        self._audit_events.append(event)
        object.__setattr__(self, "_current_stage", target)
        return event

    def seal(self, data: str) -> str:
        """Erzeugt einen SHA-256 Hash fuer Unveraenderbarkeit."""
        return hashlib.sha256(data.encode()).hexdigest()


# ── First Round ────────────────────────────────────────────────────

def run_first_round(
    state: TradingGraphState,
    graph: OrchestratorGraph,
    agents: list[Any],
    market_data: dict[str, Any],
) -> tuple[TradingGraphState, OrchestratorGraph]:
    """Fuehrt Round 1 aus: parallele Agenten-Ausfuehrung OHNE Peer-Reports.

    Jeder Agent bekommt Rohdaten (market_data + features) — NICHT die
    Reports anderer Agenten. Jede Antwort wird als AgentReport zurueckgegeben.

    Args:
        state: Aktueller Graphzustand.
        graph: OrchestratorGraph fuer Stage-Management.
        agents: Liste von Agenten mit einer ``analyze`` Methode, die
                (market_data, features) nimmt und AgentReport zurueckgibt.
        market_data: Rohdaten (candles, orderbook, features, ...).

    Returns:
        (TradingGraphState, OrchestratorGraph) mit populated first_round_reports.

    Raises:
        ValueError: Wenn keine Agenten verfuegbar sind.
    """
    if not agents:
        raise ValueError("run_first_round: at least one agent is required")

    # Jeder Agent laeuft unabhaengig — keine Peer-Reports
    reports: list[AgentReport] = []
    for agent in agents:
        report = agent.analyze(market_data)
        reports.append(report)

    # State aktualisieren
    new_state = state.__class__(
        run_id=state.run_id,
        instrument=state.instrument,
        market_snapshot=state.market_snapshot,
        first_round_reports=reports,
        current_stage=PipelineStage.FIRST_ROUND.value,
        errors=[],
        warnings=[],
    )

    graph.transition(
        PipelineStage.FIRST_ROUND,
        inputs={"agent_count": len(agents)},
        outputs={"report_count": len(reports)},
    )

    return new_state, graph


# ── Convenience: build initial state ────────────────────────────────

def create_initial_state(
    run_id: str,
    instrument: str,
    market_data: dict[str, Any] | None = None,
) -> TradingGraphState:
    """Erzeugt einen initialen TradingGraphState."""
    return TradingGraphState(
        run_id=run_id,
        instrument=instrument,
        market_snapshot=market_data or {},
    )
