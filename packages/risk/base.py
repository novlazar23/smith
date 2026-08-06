"""Base types for the risk module.

Defines gate types, gate results, and the composite risk decision.
Uses dataclasses for lightweight, mutable-by-default semantics where
needed during construction (Pydantic schema in schemas/risk_decision.py
is the canonical frozen serialisation form).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


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


@dataclass(frozen=True)
class RiskGateResult:
    """Einzelnes Gate-Ergebnis."""

    gate_type: RiskGateType
    passed: bool
    severity: str  # "hard" or "soft"
    blocking_reasons: list[str] = field(default_factory=list)
    max_position_size: float | None = None
    reduction_factor: float | None = None


@dataclass(frozen=True)
class RiskDecision:
    """Gesamtes Ergebnis der Risikogate-Prüfung."""

    risk_version: str
    run_id: str
    instrument: str
    approved: bool
    max_position_size: float | None = None
    reduction_factor: float = 1.0
    blocking_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gates: list[RiskGateResult] = field(default_factory=list)

    @property
    def veto(self) -> bool:
        """True, wenn mindestens ein hartes Gate blockiert."""
        return any(
            not g.passed and g.severity == "hard"
            for g in self.gates
        )


@dataclass
class PositionSizerConfig:
    """Konfiguration für Positionsgrössen-Berechner."""

    base_risk_pct: float = 0.02
    max_risk_pct: float = 0.10
    min_position_size: float = 0.0
    max_position_size: float = 1.0
