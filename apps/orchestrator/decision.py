"""Risk-Gates und finale Entscheidung: Risk-Gates, Publish, Schedule.

§11.16–11.18
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from apps.orchestrator.graph import StageManager, TradingGraphState
from apps.orchestrator.stages_enum import AnalysisStage
from packages.risk.base import RiskDecision as BaseRiskDecision
from packages.risk.base import RiskGateResult as BaseRiskGateResult
from packages.risk.base import RiskGateType
from packages.schemas.final_decision import (
    FinalDecision,
    FinalDecisionType,
)


@dataclass(frozen=True)
class Decision:
    """Finale Entscheidungs-Datenklasse.

    Enthält run_id, direction, uncertainty, strategy, portfolio,
    risk und reasoning für die Endentscheidung des Graphen.
    """

    run_id: str
    direction: FinalDecisionType
    uncertainty: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    portfolio: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


def apply_risk_gates(
    state: TradingGraphState,
    manager: StageManager,
    risk_manager: Any | None = None,
) -> tuple[TradingGraphState, StageManager]:
    """11.16 apply_risk_gates — Harte/weiche Grenzen.

    Wendet harte Gates (immer blockierend) und weiche Gates
    (Reduktion) an. Ergebnisse: zulassen/reduzieren/blockieren.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.
        risk_manager: RiskManager für Gate-Prüfung.

    Returns:
        (TradingGraphState, StageManager) — mit risk_report.
    """
    strategy_report = state.strategy_report or {}
    portfolio_report = state.portfolio_report or {}
    data_quality = state.data_quality or {}

    if risk_manager is not None:
        risk_decision = risk_manager.evaluate(state)
    else:
        # Default: keine harten Gates blockiert, weiche Gates prüfen
        risk_decision = _default_risk_decision(
            state, strategy_report, portfolio_report, data_quality
        )

    gate_results = {
        g.gate_type: {
            "passed": g.passed,
            "severity": g.severity,
            "blocking_reasons": g.blocking_reasons,
        }
        for g in risk_decision.gates
    }

    state = state.model_copy(update={
        "current_stage": AnalysisStage.APPLY_RISK_GATES.value,
        "risk_report": {
            "approved": risk_decision.approved,
            "veto": risk_decision.veto,
            "reduction_factor": risk_decision.reduction_factor,
            "gates": gate_results,
            "blocking_reasons": risk_decision.blocking_reasons,
        },
    })

    manager.transition(
        AnalysisStage.APPLY_RISK_GATES,
        inputs={
            "approved": risk_decision.approved,
            "veto": risk_decision.veto,
            "gate_count": len(risk_decision.gates),
        },
        outputs={
            "approved": risk_decision.approved,
            "veto": risk_decision.veto,
        },
    )

    return state, manager


def publish_decision(
    state: TradingGraphState,
    manager: StageManager,
) -> Decision:
    """11.17 publish_decision — FINALE Entscheidung.

    Produziert die endgültige Entscheidung: LONG_BIAS, SHORT_BIAS,
    RANGE, NO_TRADE, oder eine der NO_TRADE_* Varianten.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.

    Returns:
        FinalDecision — die endgültige Entscheidungsentscheidung.
    """
    validation_status = state.validation_status or "VALID"
    risk_report = state.risk_report or {}
    consensus_report = state.consensus_report or {}
    strategy_report = state.strategy_report or {}
    contrarian_report = state.contrarian_report or {}

    decision_type, reason, blocking_reasons = _determine_final_decision(
        validation_status, risk_report, consensus_report,
        strategy_report, contrarian_report,
    )

    uncertainty = _compute_uncertainty(consensus_report, contrarian_report)

    final = FinalDecision(
        run_id=state.run_id,
        instrument=state.instrument,
        horizons=state.request.get("timeframes", ["1h"]) if state.request else ["1h"],
        analysis_time=state.analysis_time,
        decision=decision_type,
        reason=reason,
        blocking_reasons=blocking_reasons,
        forecast=consensus_report,
        uncertainty=uncertainty,
        strategy=strategy_report.get("best_proposal", {}) or {},
        portfolio=state.portfolio_report or {},
        risk=risk_report,
    )

    state = state.model_copy(update={
        "current_stage": AnalysisStage.PUBLISH_DECISION.value,
        "final_decision": final,
        "status": "completed",
        "completed_at": datetime.now(UTC),
    })

    manager.transition(
        AnalysisStage.PUBLISH_DECISION,
        inputs={
            "consensus": consensus_report.get("decision", "NO_TRADE"),
            "risk_approved": risk_report.get("approved", True),
        },
        outputs={"decision": decision_type.value},
    )

    return Decision(
        run_id=state.run_id,
        direction=decision_type,
        uncertainty=uncertainty,
        strategy=strategy_report,
        portfolio=state.portfolio_report or {},
        risk=risk_report,
        reasoning=reason,
    )


def schedule_evaluation(
    state: TradingGraphState,
    manager: StageManager,
    evaluation_horizon: str = "1h",
) -> tuple[TradingGraphState, StageManager]:
    """11.18 schedule_evaluation — Evaluation nach Horizontende planen.

    Plant eine Evaluierung des Trades nach dem Ende des aktuellen
    Horizontes und speichert Zielwerte.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.
        evaluation_horizon: Zeitrahmen für die Evaluierung.

    Returns:
        (TradingGraphState, StageManager) — mit evaluation_schedule.
    """
    strategy_report = state.strategy_report or {}
    best_proposal = strategy_report.get("best_proposal") or {}

    evaluation_schedule = {
        "horizon": evaluation_horizon,
        "scheduled_at": (state.analysis_time if state.analysis_time else datetime.now(UTC)).isoformat(),
        "target_prices": best_proposal.get("targets", []),
        "stop_loss": best_proposal.get("stop_loss", 0.0),
        "entry_price": best_proposal.get("entry_price", 0.0),
        "evaluation_criteria": [
            "price_at_horizon",
            "drawdown_peak",
            "volatility_change",
        ],
    }

    state = state.model_copy(update={
        "current_stage": AnalysisStage.SCHEDULE_EVALUATION.value,
        "evaluation_schedule": evaluation_schedule,
        "status": "completed",
    })

    manager.transition(
        AnalysisStage.SCHEDULE_EVALUATION,
        inputs={"horizon": evaluation_horizon},
        outputs={"scheduled_at": evaluation_schedule["scheduled_at"]},
    )

    return state, manager


# --- Private Helpers ---


def _default_risk_decision(
    state: TradingGraphState,
    strategy_report: dict[str, Any],
    portfolio_report: dict[str, Any],
    data_quality: dict[str, Any],
) -> BaseRiskDecision:
    """Erzeugt eine Standard-Risk-Decision wenn kein RiskManager verfügbar ist."""
    gates: list[BaseRiskGateResult] = []
    approved = True
    blocking_reasons: list[str] = []

    # Data Quality Gate
    quality = data_quality.get("overall_quality", 1.0)
    dq_gate = BaseRiskGateResult(
        gate_type=RiskGateType.DATA_QUALITY,
        passed=quality >= 0.95,
        severity="hard",
        blocking_reasons=[] if quality >= 0.95 else ["quality below threshold"],
    )
    gates.append(dq_gate)
    if not dq_gate.passed:
        approved = False
        blocking_reasons.append("data quality below threshold")

    # Negative Edge Gate
    expected_net = strategy_report.get("best_proposal", {}).get("expected_return_net", 0)
    edge_gate = BaseRiskGateResult(
        gate_type=RiskGateType.NEGATIVE_EDGE,
        passed=expected_net > 0.002,
        severity="hard",
        blocking_reasons=[] if expected_net > 0.002 else ["expected net edge below minimum"],
    )
    gates.append(edge_gate)
    if not edge_gate.passed:
        approved = False
        blocking_reasons.append("negative edge")

    return BaseRiskDecision(
        risk_version="1.0.0",
        run_id=state.run_id,
        instrument=state.instrument,
        approved=approved,
        reduction_factor=1.0,
        blocking_reasons=blocking_reasons,
        gates=gates,
    )


def _determine_final_decision(
    validation_status: str,
    risk_report: dict[str, Any],
    consensus_report: dict[str, Any],
    strategy_report: dict[str, Any],
    contrarian_report: dict[str, Any],
) -> tuple[FinalDecisionType, str, list[str]]:
    """Bestimme die finale Entscheidung aus allen Berichten."""
    blocking_reasons: list[str] = []

    # 1. Data quality check
    if validation_status == "NO_TRADE_DATA_QUALITY":
        return (
            FinalDecisionType.NO_TRADE_DATA_QUALITY,
            "Data quality below acceptable threshold",
            ["data quality insufficient"],
        )

    # 2. Risk veto check
    if risk_report:
        if risk_report.get("veto", False):
            blocking_reasons.extend(risk_report.get("blocking_reasons", []))
            return (
                FinalDecisionType.NO_TRADE_RISK,
                f"Risk veto: {', '.join(risk_report.get('blocking_reasons', []))}",
                blocking_reasons,
            )
        if not risk_report.get("approved", True):
            blocking_reasons.extend(risk_report.get("blocking_reasons", []))
            return (
                FinalDecisionType.NO_TRADE_RISK,
                "Risk gates not approved",
                blocking_reasons,
            )

    # 3. Portfolio check
    portfolio_report = None
    if consensus_report:
        portfolio_report = consensus_report.get("portfolio")

    if portfolio_report and not portfolio_report.get("new_trade_allowed", True):
        return (
            FinalDecisionType.NO_TRADE_PORTFOLIO,
            "Portfolio constraints prevent new trade",
            ["portfolio constraints"],
        )

    # 4. Consensus decision
    consensus_decision = consensus_report.get("decision", "NO_TRADE")

    # Map consensus to final
    if consensus_decision in ("NO_TRADE", "NO_TRADE_DATA_QUALITY", "NO_TRADE_INSUFFICIENT_EDGE",
                               "NO_TRADE_RISK", "NO_TRADE_PORTFOLIO", "NO_TRADE_MODEL_UNCERTAINTY"):
        return (
            FinalDecisionType(consensus_decision),
            f"Consensus decision: {consensus_decision}",
            ["insufficient consensus"],
        )

    if consensus_decision == "LONG_BIAS":
        return (FinalDecisionType.LONG_BIAS, "Strong long consensus", [])

    if consensus_decision == "SHORT_BIAS":
        return (FinalDecisionType.SHORT_BIAS, "Strong short consensus", [])

    if consensus_decision == "RANGE":
        return (FinalDecisionType.RANGE, "Market in ranging regime", [])

    # Fallback NO_TRADE
    return (
        FinalDecisionType.NO_TRADE,
        "No actionable consensus reached",
        ["no decisive signal"],
    )


def _compute_uncertainty(
    consensus_report: dict[str, Any],
    contrarian_report: dict[str, Any],
) -> dict[str, Any]:
    """Berechne Unsicherheitsmetriken."""
    confidence = consensus_report.get("confidence", 0.0)
    disagreement_count = len(consensus_report.get("disagreements", []))

    return {
        "confidence": confidence,
        "disagreement_count": disagreement_count,
        "contrarian_verdict": contrarian_report.get("verdict", "PASS"),
        "uncertainty_level": "low" if confidence > 0.8 else ("medium" if confidence > 0.5 else "high"),
    }
