"""FinalDecision — Das Ergebnis des gesamten Analyse-Graphen.

Mögliche Entscheidungen (EPIC-09-WP06):
  LONG_BIAS, SHORT_BIAS, RANGE, NO_TRADE,
  NO_TRADE_DATA_QUALITY, NO_TRADE_INSUFFICIENT_EDGE,
  NO_TRADE_PORTFOLIO, NO_TRADE_RISK, NO_TRADE_MODEL_UNCERTAINTY.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FinalDecisionType(StrEnum):
    """Mögliche Endentscheidungen des Graphen."""

    LONG_BIAS = "LONG_BIAS"
    SHORT_BIAS = "SHORT_BIAS"
    RANGE = "RANGE"

    NO_TRADE = "NO_TRADE"
    NO_TRADE_DATA_QUALITY = "NO_TRADE_DATA_QUALITY"
    NO_TRADE_INSUFFICIENT_EDGE = "NO_TRADE_INSUFFICIENT_EDGE"
    NO_TRADE_PORTFOLIO = "NO_TRADE_PORTFOLIO"
    NO_TRADE_RISK = "NO_TRADE_RISK"
    NO_TRADE_MODEL_UNCERTAINTY = "NO_TRADE_MODEL_UNCERTAINTY"


class FinalDecision(BaseModel):
    """Finale Entscheidung des Analyse-Graphen.

    Enthält Begründung und Sperrgründe (EPIC-09-DoD).
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    instrument: str
    horizons: list[str]
    analysis_time: datetime
    decision: FinalDecisionType
    reason: str
    blocking_reasons: list[str] = Field(default_factory=list)
    forecast: dict[str, Any] = Field(default_factory=dict)
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    strategy: dict[str, Any] = Field(default_factory=dict)
    portfolio: dict[str, Any] = Field(default_factory=dict)
    risk: dict[str, Any] = Field(default_factory=dict)
    audit_hash: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
