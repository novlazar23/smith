"""AnalysisRequest — Der Auftrag an den Analyse-Orchestrator.

Definiert was analysiert werden soll, zu welchem Zeitpunkt,
in welchem Modus und welche Agenten angefordert sind.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalysisMode(StrEnum):
    """Ausführungsmodus einer Analyse.

    LIVE ist im MVP technisch blockiert (Akzeptanztest AT-014).
    """

    RESEARCH = "research"
    BACKTEST = "backtest"
    PAPER = "paper"
    SHADOW = "shadow"


class AnalysisRequest(BaseModel):
    """Anfrage an den Analyse-Orchestrator."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    instrument: str = Field(..., min_length=1)
    venues: list[str] = Field(default_factory=list, min_length=1)
    horizons: list[str] = Field(default_factory=lambda: ["15m", "4h", "1d"])
    analysis_time: datetime
    portfolio_id: str | None = None
    mode: AnalysisMode = AnalysisMode.RESEARCH
    requested_agents: list[str] | None = None
    """Optionale Liste von Agenten-IDs. Wenn leer/None, alle verfügbaren Agenten."""


class AnalysisResult(BaseModel):
    """Ergebnis einer Analyse-Run.

    Struktur entspricht EPIC-08 (Consensus) und EPIC-09 (Final Decision).
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    instrument: str
    horizons: list[str]
    analysis_time: datetime
    mode: AnalysisMode
    final_decision: str = "NO_TRADE"
    forecast: dict[str, Any] = Field(default_factory=dict)
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    strategy: dict[str, Any] = Field(default_factory=dict)
    portfolio: dict[str, Any] = Field(default_factory=dict)
    risk: dict[str, Any] = Field(default_factory=dict)
