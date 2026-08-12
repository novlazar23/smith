"""Graph-Kernel: TradingGraphState, Stage-Management, Audit-Trail, Versiegelung.

§10 Graphzustand + §11 18 Graphknoten.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from apps.orchestrator.stages_enum import AnalysisStage
from pydantic import BaseModel, ConfigDict, Field


class TradingGraphState(BaseModel):
    """Zustand des Analyse-Graphen mit allen Feldern aus Spec §10.

    Jede Stufe wird erst befüllt, wenn die vorherige abgeschlossen ist.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    instrument: str
    request: dict[str, Any] = Field(default_factory=dict)
    analysis_time: Any = None  # datetime — avoids circular import at load time
    model_version: str = "1.0.0"
    prompt_version: str = "1.0.0"
    market_snapshot_id: str | None = None
    market_snapshot: dict[str, Any] | None = None
    feature_snapshot_id: str | None = None
    portfolio_snapshot_id: str | None = None
    data_quality: dict[str, Any] | None = None
    validation_status: str | None = None
    regime_report: dict[str, Any] | None = None
    first_round_reports: list[dict[str, Any]] | None = None
    second_round_reports: list[dict[str, Any]] | None = None
    peer_reports: list[dict[str, Any]] | None = None
    historical_validation: dict[str, Any] | None = None
    dependency_report: dict[str, Any] | None = None
    contrarian_report: dict[str, Any] | None = None
    multi_timeframe_report: dict[str, Any] | None = None
    consensus_report: dict[str, Any] | None = None
    strategy_report: dict[str, Any] | None = None
    portfolio_report: dict[str, Any] | None = None
    risk_report: dict[str, Any] | None = None
    seal_hash: str | None = None
    sealed_at: Any = None
    features: dict[str, Any] | None = None
    evaluation_schedule: dict[str, Any] | None = None
    final_decision: Any = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    current_stage: str = "created"
    status: str = "pending"
    started_at: Any = None
    completed_at: Any = None


# --- Valide Übergänge (Hauptsequenz) ---
VALID_TRANSITIONS: dict[AnalysisStage, frozenset[AnalysisStage]] = {
    AnalysisStage.CREATE_RUN: frozenset({AnalysisStage.BUILD_MARKET_SNAPSHOT}),
    AnalysisStage.BUILD_MARKET_SNAPSHOT: frozenset({AnalysisStage.VALIDATE_DATA}),
    AnalysisStage.VALIDATE_DATA: frozenset({AnalysisStage.COMPUTE_FEATURES}),
    AnalysisStage.COMPUTE_FEATURES: frozenset({AnalysisStage.CLASSIFY_REGIME}),
    AnalysisStage.CLASSIFY_REGIME: frozenset({AnalysisStage.RUN_FIRST_ROUND}),
    AnalysisStage.RUN_FIRST_ROUND: frozenset({
        AnalysisStage.SEAL_FIRST_ROUND,
        AnalysisStage.RUN_SECOND_ROUND,
    }),
    AnalysisStage.SEAL_FIRST_ROUND: frozenset({
        AnalysisStage.HISTORICAL_VALIDATE,
    }),
    AnalysisStage.HISTORICAL_VALIDATE: frozenset({
        AnalysisStage.MEASURE_DEPENDENCIES,
    }),
    AnalysisStage.MEASURE_DEPENDENCIES: frozenset({
        AnalysisStage.RUN_SECOND_ROUND,
        AnalysisStage.CONTRARIAN_REVIEW,
    }),
    AnalysisStage.RUN_SECOND_ROUND: frozenset({
        AnalysisStage.CONTRARIAN_REVIEW,
        AnalysisStage.MULTI_TIMEFRAME,
    }),
    AnalysisStage.CONTRARIAN_REVIEW: frozenset({
        AnalysisStage.MULTI_TIMEFRAME,
    }),
    AnalysisStage.MULTI_TIMEFRAME: frozenset({
        AnalysisStage.CALCULATE_CONSENSUS,
    }),
    AnalysisStage.CALCULATE_CONSENSUS: frozenset({
        AnalysisStage.GENERATE_STRATEGY,
    }),
    AnalysisStage.GENERATE_STRATEGY: frozenset({
        AnalysisStage.EVALUATE_PORTFOLIO,
    }),
    AnalysisStage.EVALUATE_PORTFOLIO: frozenset({
        AnalysisStage.APPLY_RISK_GATES,
    }),
    AnalysisStage.APPLY_RISK_GATES: frozenset({
        AnalysisStage.PUBLISH_DECISION,
    }),
    AnalysisStage.PUBLISH_DECISION: frozenset({
        AnalysisStage.SCHEDULE_EVALUATION,
    }),
    AnalysisStage.SCHEDULE_EVALUATION: frozenset(),
}


@dataclass(frozen=True)
class AuditEvent:
    """Ein unveränderbares Audit-Event pro Stage-Übergang.

    Felder: stage, timestamp, inputs, outputs, duration_ms, event_id.
    """

    stage: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
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


class StageManager:
    """Verwaltet current_stage, AuditEvent-Liste und Transition-Logik.

    Ermöglicht schrittweise Progression durch den Analyse-Graphen.
    """

    def __init__(self) -> None:
        self._current_stage: AnalysisStage | None = None
        self._audit_events: list[AuditEvent] = []
        self._transition_log: list[tuple[AnalysisStage | None, AnalysisStage]] = []

    @property
    def current_stage(self) -> AnalysisStage | None:
        """Aktuelle Stage oder None."""
        return self._current_stage

    @property
    def audit_events(self) -> list[AuditEvent]:
        """Audit-Event-Liste (unveränderbar)."""
        return tuple(self._audit_events)

    @property
    def transition_log(self) -> list[tuple[AnalysisStage | None, AnalysisStage]]:
        """Log aller Stage-Übergänge."""
        return list(self._transition_log)

    def stage_count(self) -> int:
        """Anzahl der abgeschlossenen Stages."""
        return len(self._audit_events)

    def has_completed(self, stage: AnalysisStage) -> bool:
        """True wenn die Stage bereits durchlaufen wurde."""
        return any(e.stage == stage.value for e in self._audit_events)

    def can_transition(self, target: AnalysisStage) -> bool:
        """Prüft ob der Übergang zur Ziel-stage erlaubt ist."""
        if self._current_stage is None:
            return target == AnalysisStage.CREATE_RUN
        allowed = VALID_TRANSITIONS.get(self._current_stage, frozenset())
        return target in allowed

    def transition(
        self,
        target: AnalysisStage,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> AuditEvent:
        """Führt einen Stage-Übergang durch und erstellt ein Audit-Event.

        Raises:
            ValueError: Wenn der Übergang nicht erlaubt ist.
        """
        if not self.can_transition(target):
            if self._current_stage is None:
                expected = AnalysisStage.CREATE_RUN
            else:
                expected = self._current_stage
            raise ValueError(
                f"Ungültiger Übergang von '{self._current_stage.value if self._current_stage else 'none'}' "
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

    def seal(self, target: AnalysisStage, data: str) -> str:
        """Erzeugt einen Hash über die Stage-Daten für Unveränderbarkeit.

        Returns:
            SHA256-Hash der Stage + Daten als hex.
        """
        raw = f"{target.value}:{data}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def verify_seal(self, target: AnalysisStage, data: str, expected_hash: str) -> bool:
        """Verifiziert einen zuvor erzeugten Seal-Hash."""
        return hmac.compare_digest(
            self.seal(target, data), expected_hash
        )


# --- Transition-Funktion (Module-Level) ---

def transition(
    stage: AnalysisStage,
    manager: StageManager,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    duration_ms: float = 0.0,
) -> StageManager:
    """Führt einen Stage-Übergang durch (funktionale API).

    Returns:
        Der modifizierte StageManager (new instance not mutated).
    """
    manager.transition(
        stage, inputs=inputs, outputs=outputs, duration_ms=duration_ms
    )
    return manager
