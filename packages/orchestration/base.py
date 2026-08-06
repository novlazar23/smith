"""Basis-Klassen und Enumerations für den Orchestration-Workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class WorkflowState(StrEnum):
    """Mögliche Zustände eines Workflows."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowStep(StrEnum):
    """Schritte der Analyse-Pipeline."""

    DATA_INGESTION = "data_ingestion"
    INDICATORS = "indicators"
    REGIME_DETECTION = "regime_detection"
    CHART_ANALYSIS = "chart_analysis"
    FIBONACCI_ANALYSIS = "fibonacci_analysis"
    ORDERFLOW_ANALYSIS = "orderflow_analysis"
    AGENT_ANALYSIS = "agent_analysis"
    CONSENSUS = "consensus"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_CHECK = "portfolio_check"
    GOVERNANCE = "governance"
    EXECUTION = "execution"


@dataclass
class AnalysisWorkflow:
    """Repräsentiert einen einzelnen Analyse-Workflow."""

    workflow_id: str
    instrument: str
    run_id: str
    state: WorkflowState = WorkflowState.PENDING
    current_step: WorkflowStep | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step_result(self, step: WorkflowStep, status: str,
                        duration_ms: float, result: Any = None) -> None:  # noqa: ANN401
        """Speichert das Ergebnis eines einzelnen Schritts."""
        self.steps.append({
            "step": step.value,
            "status": status,
            "duration_ms": duration_ms,
            "result": result,
        })

    @property
    def is_complete(self) -> bool:
        """True wenn der Workflow erfolgreich abgeschlossen ist."""
        return self.state == WorkflowState.COMPLETED

    @property
    def total_duration_ms(self) -> float:
        """Gesamtdauer aller ausgeführten Schritte in Millisekunden."""
        return sum((s["duration_ms"] for s in self.steps), 0.0)

    @property
    def failed_steps(self) -> list[str]:
        """Liste der fehlgeschlagenen Schrittnamen."""
        return [s["step"] for s in self.steps if s["status"] == "failed"]
