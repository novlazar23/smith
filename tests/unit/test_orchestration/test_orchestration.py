"""Tests für Orchestration Engine — Workflow-State-Machine und Engine-Logik."""

from __future__ import annotations

import time
from typing import Any

import pytest
from packages.orchestration import (
    AnalysisWorkflow,
    OrchestrationEngine,
    WorkflowState,
    WorkflowStep,
)


class TestWorkflowState:
    """Testet WorkflowState-Enum."""

    def test_all_states_present(self) -> None:
        assert WorkflowState.PENDING.value == "pending"
        assert WorkflowState.RUNNING.value == "running"
        assert WorkflowState.COMPLETED.value == "completed"
        assert WorkflowState.FAILED.value == "failed"
        assert WorkflowState.CANCELLED.value == "cancelled"
        assert WorkflowState.PAUSED.value == "paused"


class TestWorkflowStep:
    """Testet WorkflowStep-Enum."""

    def test_all_steps_present(self) -> None:
        expected: list[str] = [
            "data_ingestion",
            "indicators",
            "regime_detection",
            "chart_analysis",
            "fibonacci_analysis",
            "orderflow_analysis",
            "agent_analysis",
            "consensus",
            "risk_assessment",
            "portfolio_check",
            "governance",
            "execution",
        ]
        actual = [s.value for s in WorkflowStep]
        assert actual == expected


class TestAnalysisWorkflow:
    """Testet AnalysisWorkflow-Datenklasse."""

    def test_create_in_pending_state(self) -> None:
        wf = AnalysisWorkflow(
            workflow_id="wf-001",
            instrument="BTC/USD",
            run_id="run-42",
        )
        assert wf.state == WorkflowState.PENDING
        assert wf.current_step is None
        assert wf.steps == []
        assert wf.error is None

    def test_is_complete_returns_true_for_completed(self) -> None:
        wf = AnalysisWorkflow(
            workflow_id="wf-001",
            instrument="BTC/USD",
            run_id="run-42",
        )
        assert wf.is_complete is False
        wf.state = WorkflowState.COMPLETED
        assert wf.is_complete is True

    def test_is_complete_false_when_running(self) -> None:
        wf = AnalysisWorkflow(
            workflow_id="wf-001",
            instrument="BTC/USD",
            run_id="run-42",
        )
        wf.state = WorkflowState.RUNNING
        assert wf.is_complete is False

    def test_total_duration_sums_steps(self) -> None:
        wf = AnalysisWorkflow(
            workflow_id="wf-001",
            instrument="BTC/USD",
            run_id="run-42",
        )
        wf.add_step_result(WorkflowStep.DATA_INGESTION, "completed", 150.0)
        wf.add_step_result(WorkflowStep.INDICATORS, "completed", 250.5)
        assert wf.total_duration_ms == 400.5

    def test_failed_steps_returns_failed_names(self) -> None:
        wf = AnalysisWorkflow(
            workflow_id="wf-001",
            instrument="BTC/USD",
            run_id="run-42",
        )
        wf.add_step_result(WorkflowStep.DATA_INGESTION, "completed", 100.0)
        wf.add_step_result(WorkflowStep.INDICATORS, "failed", 50.0)
        wf.add_step_result(WorkflowStep.REGIME_DETECTION, "completed", 200.0)
        assert wf.failed_steps == ["indicators"]

    def test_failed_steps_empty_when_all_succeed(self) -> None:
        wf = AnalysisWorkflow(
            workflow_id="wf-001",
            instrument="BTC/USD",
            run_id="run-42",
        )
        wf.add_step_result(WorkflowStep.DATA_INGESTION, "completed", 100.0)
        assert wf.failed_steps == []


class TestOrchestrationEngine:
    """Testet OrchestrationEngine."""

    @pytest.fixture
    def engine(self) -> OrchestrationEngine:
        return OrchestrationEngine()

    def _success_callback(self, step: WorkflowStep,
                          workflow: AnalysisWorkflow) -> dict[str, Any]:
        """Callback der immer erfolgreich ist."""
        return {"step": step.value, "ok": True}

    def test_create_workflow(self) -> None:
        engine = OrchestrationEngine()
        wf = engine.create_workflow("wf-001", "ETH/USD", "run-1")
        assert wf.workflow_id == "wf-001"
        assert wf.instrument == "ETH/USD"
        assert wf.run_id == "run-1"
        assert wf.state == WorkflowState.PENDING

    def test_engine_execute_success(self, engine: OrchestrationEngine) -> None:
        wf = engine.create_workflow("wf-001", "BTC/USD", "run-1")
        callbacks = {s.value: self._success_callback for s in WorkflowStep}
        result = engine.execute_workflow(wf, callbacks)

        assert result.state == WorkflowState.COMPLETED
        assert result.completed_at is not None
        assert result.started_at is not None
        assert result.total_duration_ms > 0

        # Alle 12 Schritte sollten im Ergebnis
        assert len(result.steps) == 12
        assert result.failed_steps == []

    def test_engine_step_failure(self, engine: OrchestrationEngine) -> None:
        def failing_callback(step: WorkflowStep,
                             workflow: AnalysisWorkflow) -> None:
            if step == WorkflowStep.REGIME_DETECTION:
                raise RuntimeError("Simulierter Fehler")
            return {"step": step.value, "ok": True}

        wf = engine.create_workflow("wf-002", "ETH/USD", "run-2")
        callbacks = {
            WorkflowStep.DATA_INGESTION.value: self._success_callback,
            WorkflowStep.INDICATORS.value: self._success_callback,
            WorkflowStep.REGIME_DETECTION.value: failing_callback,
            WorkflowStep.CHART_ANALYSIS.value: self._success_callback,
        }

        result = engine.execute_workflow(wf, callbacks)

        assert result.state == WorkflowState.FAILED
        assert "Simulierter Fehler" in result.error

    def test_engine_workflow_status(self, engine: OrchestrationEngine) -> None:
        wf = engine.create_workflow("wf-001", "BTC/USD", "run-1")
        wf.state = WorkflowState.RUNNING
        wf.current_step = WorkflowStep.REGIME_DETECTION
        wf.add_step_result(WorkflowStep.DATA_INGESTION, "completed", 100.0)
        wf.add_step_result(WorkflowStep.INDICATORS, "failed", 50.0)

        status = engine.get_workflow_status(wf)

        assert status["state"] == WorkflowState.RUNNING
        assert status["current_step"] == "regime_detection"
        assert status["total_steps"] == len(WorkflowStep)
        assert status["completed_steps"] == 1
        assert status["failed_steps"] == ["indicators"]
        assert "duration" in status

    def test_engine_pause_resume(self, engine: OrchestrationEngine) -> None:
        wf = engine.create_workflow("wf-001", "BTC/USD", "run-1")
        wf.state = WorkflowState.RUNNING

        engine.pause_workflow(wf)
        assert wf.state == WorkflowState.PAUSED

        engine.resume_workflow(wf)
        assert wf.state == WorkflowState.RUNNING

    def test_engine_cancel(self, engine: OrchestrationEngine) -> None:
        wf = engine.create_workflow("wf-001", "BTC/USD", "run-1")
        wf.state = WorkflowState.RUNNING

        engine.cancel_workflow(wf)
        assert wf.state == WorkflowState.CANCELLED

    def test_engine_validate_transition_valid(self, engine: OrchestrationEngine) -> None:
        assert engine.validate_transition(
            WorkflowState.PENDING, WorkflowState.RUNNING) is True
        assert engine.validate_transition(
            WorkflowState.PENDING, WorkflowState.CANCELLED) is True
        assert engine.validate_transition(
            WorkflowState.RUNNING, WorkflowState.COMPLETED) is True
        assert engine.validate_transition(
            WorkflowState.RUNNING, WorkflowState.FAILED) is True
        assert engine.validate_transition(
            WorkflowState.RUNNING, WorkflowState.PAUSED) is True
        assert engine.validate_transition(
            WorkflowState.RUNNING, WorkflowState.CANCELLED) is True
        assert engine.validate_transition(
            WorkflowState.PAUSED, WorkflowState.RUNNING) is True
        assert engine.validate_transition(
            WorkflowState.PAUSED, WorkflowState.CANCELLED) is True

    def test_engine_validate_transition_invalid(self, engine: OrchestrationEngine) -> None:
        # Kein Übergang aus COMPLETED
        assert engine.validate_transition(
            WorkflowState.COMPLETED, WorkflowState.RUNNING) is False
        assert engine.validate_transition(
            WorkflowState.COMPLETED, WorkflowState.PAUSED) is False

        # Kein Übergang aus FAILED
        assert engine.validate_transition(
            WorkflowState.FAILED, WorkflowState.RUNNING) is False

        # Kein Übergang aus CANCELLED
        assert engine.validate_transition(
            WorkflowState.CANCELLED, WorkflowState.RUNNING) is False

        # Ungültiger Übergang
        assert engine.validate_transition(
            WorkflowState.PENDING, WorkflowState.COMPLETED) is False

    def test_engine_partial_execution(self, engine: OrchestrationEngine) -> None:
        """Workflow stoppt beim ersten Fehler — keine weiteren Schritte."""
        def fail_at_third(step: WorkflowStep,
                          workflow: AnalysisWorkflow) -> None:
            if step == WorkflowStep.REGIME_DETECTION:
                raise RuntimeError("Step 3 schlägt fehl")
            return {"ok": True}

        wf = engine.create_workflow("wf-partial", "BTC/USD", "run-p")
        callbacks = {
            WorkflowStep.DATA_INGESTION.value: self._success_callback,
            WorkflowStep.INDICATORS.value: self._success_callback,
            WorkflowStep.REGIME_DETECTION.value: fail_at_third,
        }

        result = engine.execute_workflow(wf, callbacks)

        assert result.state == WorkflowState.FAILED
        assert len(result.steps) == 2  # Nur die 2 erfolgreichen Schritte
        assert "regime_detection" not in [s["step"] for s in result.steps]

    def test_engine_multiple_workflows(self, engine: OrchestrationEngine) -> None:
        """Mehrere unabhängige Workflows können parallel verarbeitet werden."""
        wf_a = engine.create_workflow("wf-a", "BTC/USD", "run-a")
        wf_b = engine.create_workflow("wf-b", "ETH/USD", "run-b")

        def slow_callback(step: WorkflowStep,
                          workflow: AnalysisWorkflow) -> dict[str, Any]:
            time.sleep(0.01)
            return {"ok": True}

        callbacks = {s.value: slow_callback for s in WorkflowStep}
        result_a = engine.execute_workflow(wf_a, callbacks)
        result_b = engine.execute_workflow(wf_b, callbacks)

        assert result_a.state == WorkflowState.COMPLETED
        assert result_b.state == WorkflowState.COMPLETED
        assert result_a.workflow_id == "wf-a"
        assert result_b.workflow_id == "wf-b"
        assert result_a.instrument == "BTC/USD"
        assert result_b.instrument == "ETH/USD"
