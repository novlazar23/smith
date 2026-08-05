"""TradingGraphState — Der gemeinsame Zustandsgraph.

Enthält den vollständigen Zustand des Analyse-Graphen.
Der Graph durchläuft Stufen, die den EPIC-08-Knoten entsprechen.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .agent_report import AgentReport
from .final_decision import FinalDecision
from .risk_decision import RiskDecision


class TradingGraphState(BaseModel):
    """Zustand des Analyse-Graphen.

    Jede Stufe wird erst befüllt, wenn die vorherige abgeschlossen ist.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    request: Any  # AnalysisRequest — avoids circular import at load time
    market_snapshot_id: str | None = None
    feature_snapshot_id: str | None = None
    portfolio_snapshot_id: str | None = None
    data_quality_report: dict[str, Any] | None = None
    regime_report: dict[str, Any] | None = None
    first_round_reports: list[AgentReport] | None = None
    second_round_reports: list[AgentReport] | None = None
    historical_validation: dict[str, Any] | None = None
    dependency_report: dict[str, Any] | None = None
    contrarian_report: dict[str, Any] | None = None
    multi_timeframe_report: dict[str, Any] | None = None
    consensus_report: dict[str, Any] | None = None
    strategy_report: dict[str, Any] | None = None
    portfolio_report: dict[str, Any] | None = None
    risk_report: RiskDecision | None = None
    final_decision: FinalDecision | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    audit_events: list[str] = Field(default_factory=list)
    current_stage: str = "created"
    status: str = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
