"""Data models for the strategy module.

Defines StrategyDirection, StrategyProposal, StrategyConfig,
and StrategyVariant — the core data structures for entry/stop/target/exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StrategyDirection(StrEnum):
    """Handlungsrichtung der Strategie-Entscheidung."""

    LONG = "long"
    SHORT = "short"
    NO_TRADE = "no_trade"
    NO_TRADE_DATA_QUALITY = "no_trade_data_quality"
    NO_TRADE_INSUFFICIENT_EDGE = "no_trade_insufficient_edge"
    NO_TRADE_RISK = "no_trade_risk"
    NO_TRADE_PORTFOLIO = "no_trade_portfolio"
    NO_TRADE_MODEL_UNCERTAINTY = "no_trade_model_uncertainty"


@dataclass(frozen=True)
class StrategyConfig:
    """Konfiguration der strategischen Gates.

    Alle Schwellwerte sind als Mindestanforderungen definiert.
    """

    min_edge: float = 0.002  # 0.2% minimaler erwarteter Nettovorteil
    min_prob: float = 0.55  # 55% minimale Wahrscheinlichkeit Target vor Stop
    min_rr: float = 1.5  # 1.5:1 minimales Risk-Reward-Verhältnis
    min_quality: float = 0.95  # 95% minimale Datenqualität

    def validate(self) -> None:
        """Prüft Gültigkeitsbereich aller Konfigurationswerte."""
        if not (0.0 < self.min_edge < 1.0):
            raise ValueError(f"min_edge must be in (0, 1), got {self.min_edge}")
        if not (0.0 < self.min_prob <= 1.0):
            raise ValueError(f"min_prob must be in (0, 1], got {self.min_prob}")
        if self.min_rr <= 0.0:
            raise ValueError(f"min_rr must be > 0, got {self.min_rr}")
        if not (0.0 < self.min_quality <= 1.0):
            raise ValueError(f"min_quality must be in (0, 1], got {self.min_quality}")


@dataclass(frozen=True)
class StrategyProposal:
    """Strategischer Vorschlag — Ergebnis der Strategy-Engine.

    Enthält Entry, Stop, Targets, Wahrscheinlichkeiten, EV nach Kosten,
    MAE/MFE-Schätzung und Risk-Reward-Verhältnis (§16).
    """

    direction: StrategyDirection
    entry_type: str
    entry_price: float
    entry_condition: str
    stop_loss: float
    targets: list[float]
    valid_until: str | None = None
    prob_target_before_stop: float = 0.0
    prob_stop_before_target: float = 0.0
    expected_return_gross: float = 0.0
    expected_return_net: float = 0.0
    expected_costs: float = 0.0
    expected_mae: float = 0.0
    expected_mfe: float = 0.0
    risk_reward_ratio: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyVariant:
    """Alternatives Szenario für Entry/Stop/Target-Kombinationen.

    Ermöglicht die Bewertung mehrerer möglicher Handels-Szenarien.
    """

    variant_id: str
    direction: StrategyDirection
    entry_price: float
    entry_type: str
    stop_loss: float
    targets: list[float]
    probability: float = 0.0
    probabilities_target: list[float] = field(default_factory=list)
    entry_condition: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_amount(self) -> float:
        """Absolutes Risiko (Differenz Entry <-> Stop)."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def total_reward(self) -> float:
        """Gesamtes Reward (höchster Target - Entry)."""
        if not self.targets:
            return 0.0
        if self.direction == StrategyDirection.LONG:
            return max(self.targets) - self.entry_price
        return self.entry_price - min(self.targets)
