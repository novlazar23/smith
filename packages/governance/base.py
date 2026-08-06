"""Base types for the Governance Decision-Engine.

Defines the FinalDecisionType enum, DecisionRule dataclass,
GovernanceConfig, and FinalDecisionData that drive the engine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class FinalDecisionType(StrEnum):
    """Mögliche Endentscheidungen des Analyse-Graphen.

    Entsprechend EPIC-09-WP06, abgeleitet von packages/schemas/final_decision.py.
    """

    LONG_BIAS = "LONG_BIAS"
    SHORT_BIAS = "SHORT_BIAS"
    RANGE = "RANGE"

    NO_TRADE = "NO_TRADE"
    NO_TRADE_DATA_QUALITY = "NO_TRADE_DATA_QUALITY"
    NO_TRADE_INSUFFICIENT_EDGE = "NO_TRADE_INSUFFICIENT_EDGE"
    NO_TRADE_PORTFOLIO = "NO_TRADE_PORTFOLIO"
    NO_TRADE_RISK = "NO_TRADE_RISK"
    NO_TRADE_MODEL_UNCERTAINTY = "NO_TRADE_MODEL_UNCERTAINTY"


@dataclass(frozen=True)
class DecisionRule:
    """Eine einzelne Governance-Regel.

    Attributes:
        rule_id: Eindeutige Kennung der Regel.
        condition: Beschreibender Name der Bedingung, z. B.
                   ``consensus_long_above_threshold``.
        action: Aktion bei Erfülung — ``approve``, ``block`` oder ``reduce``.
        threshold: Schwellenwert, ab dem die Regel aktiv wird.
        blocking: Bestimmt, ob die Regel die Entscheidung blockiert.
    """

    rule_id: str
    condition: str
    action: str
    threshold: float
    blocking: bool


@dataclass
class GovernanceConfig:
    """Konfiguration des Governance-Entscheidungswegs.

    Alle Schwellenwerte sind im Bereich 0.0-1.0, außer ``required_agents``.
    """

    consensus_long_threshold: float = 0.65
    consensus_short_threshold: float = 0.65
    consensus_range_threshold: float = 0.50
    min_confidence: float = 0.50
    max_uncertainty: float = 0.30
    required_agents: int = 2
    rules: list[DecisionRule] = field(default_factory=list)


@dataclass
class FinalDecisionData:
    """Datensatz einer finalen Entscheidung.

    Spiegelt das Feld-Layout von ``packages/schemas/final_decision.py`` nach
    als Dataclass (kein Pydantic).
    """

    run_id: str
    instrument: str
    horizons: list[str]
    analysis_time: datetime
    decision: FinalDecisionType
    reason: str
    blocking_reasons: list[str] = field(default_factory=list)
    forecast: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    portfolio: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    audit_hash: str | None = None

    def compute_hash(self) -> str:
        """Berechnet einen SHA-256-Hash über die Schlüssel-Felder."""
        content = f"{self.run_id}|{self.instrument}|{self.decision}|{self.reason}"
        self.audit_hash = hashlib.sha256(content.encode()).hexdigest()
        return self.audit_hash
