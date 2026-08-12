"""Orchestrator App — Analysegraph mit 18 Graphknoten.

Implementiert das Stage-Management, Audit-Trail und Versiegelung
für den Trading-Orchestra-Analysegraphen (§11 der Spezifikation).
"""

from __future__ import annotations

from .agents import (
    build_multi_timeframe_view,
    run_contrarian_review,
    run_first_round,
    run_second_round,
    seal_first_round,
)
from .consensus import (
    calculate_consensus,
    evaluate_portfolio_step,
    generate_strategy_step,
)
from .decision import (
    Decision,
    apply_risk_gates,
    publish_decision,
    schedule_evaluation,
)
from .graph import (
    AnalysisStage,
    AuditEvent,
    StageManager,
    TradingGraphState,
    transition,
)
from .stages import (
    build_market_snapshot,
    classify_regime,
    compute_features,
    create_run,
    validate_data,
)

__all__: list[str] = [
    "AnalysisStage",
    "AuditEvent",
    "Decision",
    "StageManager",
    "TradingGraphState",
    "apply_risk_gates",
    "build_market_snapshot",
    "build_multi_timeframe_view",
    "calculate_consensus",
    "classify_regime",
    "compute_features",
    "create_run",
    "evaluate_portfolio_step",
    "generate_strategy_step",
    "publish_decision",
    "run_contrarian_review",
    "run_first_round",
    "run_second_round",
    "schedule_evaluation",
    "seal_first_round",
    "transition",
    "validate_data",
]
