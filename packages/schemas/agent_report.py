"""AgentReport — Die standardisierte Ausgabe jedes Agenten.

Jeder Agent (Epic-05/06/08) liefert diesen Bericht.
Wahrscheinlichkeiten summieren sich auf 1.0 ± 0.0001.
Jede Aussage benötigt Evidenzreferenzen.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentStatus(StrEnum):
    """Status eines Agenten im Lebenszyklus (EPIC-11)."""

    SHADOW = "shadow"
    ACTIVE = "active"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"
    DISABLED = "disabled"


class EvidenceReference(BaseModel):
    """Verweist auf eine Quelle, die eine Agenten-Aussage stützt."""

    model_config = ConfigDict(frozen=True)

    reference: str = Field(..., min_length=1, description="Eindeutige Referenz-ID.")
    feature: str = Field(..., min_length=1, description="Feature-Name.")
    value: str = Field(..., min_length=1, description="Feature-Wert oder -Zustand.")
    direction: str = Field(
        ...,
        description="Richtung der Evidenz: positive / negative / neutral.",
    )
    relevance: float = Field(
        ge=0.0, le=1.0, description="Relevanz der Evidenz für die Hypothese."
    )


class InvalidationCondition(BaseModel):
    """Bedingung, unter der die Agenten-Hypothese ungültig wird."""

    model_config = ConfigDict(frozen=True)

    condition: str = Field(..., min_length=1)
    indicator: str = Field(..., min_length=1, description="Welches Feature überwacht wird.")
    threshold: float = Field(..., description="Grenzwert, ab dem die Bedingung gilt.")
    direction: str = Field(..., description="Richtung: above / below.")


class AgentReport(BaseModel):
    """Standardisierter Bericht eines Analyse-Agenten."""

    model_config = ConfigDict(frozen=True)

    report_id: str
    run_id: str
    agent_id: str
    agent_version: str
    instrument: str
    horizon: str
    as_of: datetime
    hypothesis: str
    probabilities: dict[str, Any] = Field(
        default_factory=dict,
        description="{'up': float, 'down': float, 'range': float}. Summe = 1.0 ± 0.0001.",
    )
    expected_return: dict[str, Any] | None = Field(
        None,
        description="{'q10': float, 'q50': float, 'q90': float}.",
    )
    evidence: list[EvidenceReference] = Field(
        min_length=1, description="Mindestens eine Evidenz erforderlich (AT-004)."
    )
    counter_evidence: list[EvidenceReference] = Field(
        default_factory=list, description="Gegenhypothesen-Evidenz."
    )
    invalidations: list[InvalidationCondition] = Field(default_factory=list)
    sample_size: int | None = None
    raw_confidence: float | None = Field(None, ge=0.0, le=1.0)
    calibrated_confidence: float | None = Field(None, ge=0.0, le=1.0)
    data_quality: float = Field(ge=0.0, le=1.0, default=1.0)
    uncertainty: str | None = None
    status: AgentStatus = AgentStatus.SHADOW
    narrative: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    feature_snapshot_id: str | None = None

    @field_validator("probabilities")
    @classmethod
    def validate_probability_sum(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Wahrscheinlichkeitssumme muss 1.0 sein (AT-005)."""
        if not v:
            raise ValueError("probabilities must not be empty")
        total = sum(v.values())
        if abs(total - 1.0) > 0.0001:
            raise ValueError(f"probabilities sum to {total}, expected 1.0 ± 0.0001")
        return v
