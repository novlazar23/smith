"""RiskDecision — Das Ergebnis der Risikogate-Prüfung.

Der Risk-Service besitzt ein nicht überschreibbares Vetorecht (Epic-Prinzip #6).
Ein blockiertes Gate kann technisch nicht überschrieben werden.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RiskGateType(StrEnum):
    """Typen von Risikogates (EPIC-09-WP05)."""

    # Hart — blockieren immer
    DATA_QUALITY = "data_quality"
    EXPOSURE = "exposure"
    DRAWDOWN = "drawdown"
    LIQUIDITY = "liquidity"
    SPREAD = "spread"
    NEGATIVE_EDGE = "negative_edge"
    EXPIRED_SIGNAL = "expired_signal"
    SYSTEM_NOT_READY = "system_not_ready"

    # Weich — führen zu Reduktion
    UNCERTAINTY = "uncertainty"
    DISAGREEMENT = "disagreement"
    REGIME_CHANGE = "regime_change"
    NEWS_RISK = "news_risk"


class RiskGateResult(BaseModel):
    """Einzelnes Gate-Ergebnis."""

    model_config = ConfigDict(frozen=True)

    gate_type: RiskGateType
    passed: bool
    severity: str = Field(..., description="hard / soft")
    blocking_reasons: list[str] = Field(default_factory=list)
    max_position_size: float | None = None
    reduction_factor: float | None = None


class RiskDecision(BaseModel):
    """Gesamtes Ergebnis der Risikogate-Prüfung."""

    model_config = ConfigDict(frozen=True)

    risk_version: str
    run_id: str
    instrument: str
    approved: bool
    max_position_size: float | None = None
    reduction_factor: float = Field(default=1.0, ge=0.0, le=1.0)
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    gates: list[RiskGateResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def veto(self) -> bool:
        """True, wenn mindestens ein hartes Gate blockiert."""
        return any(
            not g.passed and g.severity == "hard"
            for g in self.gates
        )
