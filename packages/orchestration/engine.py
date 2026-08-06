"""Orchestration Engine — führt Workflows mit Schritten und Callbacks aus."""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from .base import AnalysisWorkflow, WorkflowState, WorkflowStep


class OrchestrationEngine:
    """Steuert Analyse-Workflows: Erstellen, Ausfuhren, Pausieren, Fortsetzen, Abbrechen."""

    DEFAULT_STEP_ORDER: ClassVar[list[WorkflowStep]] = [
        WorkflowStep.DATA_INGESTION,
        WorkflowStep.INDICATORS,
        WorkflowStep.REGIME_DETECTION,
        WorkflowStep.CHART_ANALYSIS,
        WorkflowStep.FIBONACCI_ANALYSIS,
        WorkflowStep.ORDERFLOW_ANALYSIS,
        WorkflowStep.AGENT_ANALYSIS,
        WorkflowStep.CONSENSUS,
        WorkflowStep.RISK_ASSESSMENT,
        WorkflowStep.PORTFOLIO_CHECK,
        WorkflowStep.GOVERNANCE,
        WorkflowStep.EXECUTION,
    ]

    def __init__(self) -> None:
        """Initialisiert die Engine mit der Standard-Schritt-Reihenfolge."""
        self._step_order: list[WorkflowStep] = list(self.DEFAULT_STEP_ORDER)

    def create_workflow(self, workflow_id: str, instrument: str,
                        run_id: str) -> AnalysisWorkflow:
        """Erzeugt einen neuen Workflow im Zustand PENDING."""
        return AnalysisWorkflow(
            workflow_id=workflow_id,
            instrument=instrument,
            run_id=run_id,
        )

    def execute_workflow(self, workflow: AnalysisWorkflow,
                         callbacks: dict[str, Any] | None = None) -> AnalysisWorkflow:
        """Führt alle Schritte des Workflows nacheinander aus."""
        workflow.state = WorkflowState.RUNNING
        workflow.started_at = datetime.now()
        callbacks = callbacks or {}

        for step in self._step_order:
            workflow.current_step = step
            callback = callbacks.get(step.value)

            try:
                step_result = self.execute_step(workflow, step, callback)
            except Exception as exc:
                workflow.state = WorkflowState.FAILED
                workflow.error = str(exc)
                return workflow

            if step_result["status"] != "completed":
                workflow.state = WorkflowState.FAILED
                return workflow

            workflow.add_step_result(step, step_result["status"],
                                     step_result["duration_ms"])

        workflow.state = WorkflowState.COMPLETED
        workflow.completed_at = datetime.now()
        return workflow

    def execute_step(self, workflow: AnalysisWorkflow, step: WorkflowStep,
                     callback: Any = None) -> dict[str, Any]:  # noqa: ANN401
        """Führt einen einzelnen Workflow-Schritt aus."""
        start = datetime.now()

        if callable(callback):
            result = callback(step, workflow)
            if result is not None:
                workflow.metadata[f"{step.value}_result"] = result
            duration_ms = (datetime.now() - start).total_seconds() * 1000
            return {"step": step.value, "status": "completed",
                    "duration_ms": duration_ms}

        duration_ms = (datetime.now() - start).total_seconds() * 1000
        return {"step": step.value, "status": "completed",
                "duration_ms": duration_ms}

    def pause_workflow(self, workflow: AnalysisWorkflow) -> None:
        """Setzt den Workflow-Status auf PAUSED."""
        workflow.state = WorkflowState.PAUSED

    def resume_workflow(self, workflow: AnalysisWorkflow) -> None:
        """Setzt den Workflow-Status zurück auf RUNNING."""
        workflow.state = WorkflowState.RUNNING

    def cancel_workflow(self, workflow: AnalysisWorkflow) -> None:
        """Setzt den Workflow-Status auf CANCELLED."""
        workflow.state = WorkflowState.CANCELLED

    def get_workflow_status(self, workflow: AnalysisWorkflow) -> dict[str, Any]:
        """Gibt einen Status-Dict mit Informationen zum Workflow zurück."""
        completed_steps = sum(1 for s in workflow.steps if s["status"] == "completed")
        failed = [s["step"] for s in workflow.steps if s["status"] == "failed"]

        return {
            "state": workflow.state,
            "current_step": workflow.current_step.value if workflow.current_step else None,
            "total_steps": len(self._step_order),
            "completed_steps": completed_steps,
            "failed_steps": failed,
            "duration": round(workflow.total_duration_ms, 2),
        }

    def validate_transition(self, current_state: WorkflowState,
                            target_state: WorkflowState) -> bool:
        """Prüft ob ein Zustandsübergang erlaubt ist."""
        valid_transitions: dict[WorkflowState, set[WorkflowState]] = {
            WorkflowState.PENDING: {WorkflowState.RUNNING, WorkflowState.CANCELLED},
            WorkflowState.RUNNING: {WorkflowState.COMPLETED, WorkflowState.FAILED,
                                    WorkflowState.PAUSED, WorkflowState.CANCELLED},
            WorkflowState.PAUSED: {WorkflowState.RUNNING, WorkflowState.CANCELLED},
            WorkflowState.COMPLETED: set(),
            WorkflowState.FAILED: set(),
            WorkflowState.CANCELLED: set(),
        }

        allowed = valid_transitions.get(current_state, set())
        return target_state in allowed
