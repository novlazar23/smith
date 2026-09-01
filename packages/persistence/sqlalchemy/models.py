"""Models — SQLAlchemy ORM-Models für relationale Tabellen.

Mapping der Pydantic-Schemas auf SQLAlchemy-Entities:
  - TradingGraphState -> trading_graph_states
  - FinalDecision -> final_decisions
  - RiskDecision -> risk_decisions
  - AnalysisRequest -> analysis_requests
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Basis-Class für alle SQLAlchemy-Models."""


class TradingGraphStateModel(Base):
    """SQLAlchemy-Model für TradingGraphState.

    Persistiert den vollständigen Zustand des Analyse-Graphen.
    Jede Stufe wird erst befüllt, wenn die vorherige abgeschlossen ist.
    """

    __tablename__ = "trading_graph_states"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: _generate_uuid()
    )
    request_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    instrument: Mapped[str] = mapped_column(String, nullable=False)
    horizons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    analysis_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    current_stage: Mapped[str] = mapped_column(String, nullable=False, default="created")
    graph_state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    errors: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    warnings: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    audit_events: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class FinalDecisionModel(Base):
    """SQLAlchemy-Model für FinalDecision.

    Persistiert die finale Entscheidung des Analyse-Graphen.
    """

    __tablename__ = "final_decisions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: _generate_uuid()
    )
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    instrument: Mapped[str] = mapped_column(String, nullable=False)
    horizons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    analysis_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decision: Mapped[str] = mapped_column(String, nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    blocking_reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    forecast: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    uncertainty: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    strategy: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    portfolio: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    risk: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    audit_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class RiskDecisionModel(Base):
    """SQLAlchemy-Model für RiskDecision.

    Persistiert das Ergebnis der Risikogate-Prüfung.
    """

    __tablename__ = "risk_decisions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: _generate_uuid()
    )
    risk_version: Mapped[str] = mapped_column(String, nullable=False)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    instrument: Mapped[str] = mapped_column(String, nullable=False)
    approved: Mapped[bool] = mapped_column(nullable=False)
    max_position_size: Mapped[float | None] = mapped_column(nullable=True)
    reduction_factor: Mapped[float] = mapped_column(default=1.0)
    blocking_reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    warnings: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    gates: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AnalysisRequestModel(Base):
    """SQLAlchemy-Model für AnalysisRequest.

    Persistiert Analyse-Anfragen an den Orchestrator.
    """

    __tablename__ = "analysis_requests"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: _generate_uuid()
    )
    instrument: Mapped[str] = mapped_column(String, nullable=False)
    venues: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    horizons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    analysis_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    portfolio_id: Mapped[str | None] = mapped_column(String, nullable=True)
    mode: Mapped[str] = mapped_column(String, nullable=False, default="research")
    requested_agents: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ShadowDecisionModel(Base):
    """SQLAlchemy-Model für Shadow-Entscheidungen des Orchestrator-Services.

    Persistiert die finale Shadow-Entscheidung der OrchestratorPipeline pro
    (Zyklus, Instrument). Rein dokumentierend/auditierend — es werden hier
    **nie** Orders ausgeführt.
    """

    __tablename__ = "shadow_decisions"

    run_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    instrument: Mapped[str] = mapped_column(String, nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String, nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    first_round_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    second_round_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    errors: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


def _generate_uuid() -> str:
    """Generiert eine UUID-String.

    UUID4 wird zur Laufzeit generiert. Für Tests kann diese
    Funktion durch einen Mock ersetzt werden.
    """
    import uuid

    return str(uuid.uuid4())
