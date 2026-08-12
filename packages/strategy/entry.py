"""Entry-Typen und -Bedingungen für die Strategie-Engine.

Definiert EntryType (MARKET, LIMIT, BRACKET), EntryCondition
(Breakout, Pullback, Reversal, etc.), EntrySignal und die
evaluate_entry-Funktion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .models import StrategyConfig


class EntryType(StrEnum):
    """Typ der Order-Eingabe."""

    MARKET = "market"
    LIMIT = "limit"
    BRACKET = "bracket"


class EntryCondition(StrEnum):
    """Bedingung für den Entry."""

    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    REVERSAL = "reversal"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    VOLATILITY_CONTRACTION = "volatility_contraction"


@dataclass(frozen=True)
class EntrySignal:
    """Kombiniertes Entry-Signal mit Preis, Typ und Bedingung."""

    price: float
    entry_type: EntryType
    condition: EntryCondition
    strength: float = 1.0
    metadata: dict = field(default_factory=dict)


def evaluate_entry(
    config: StrategyConfig,
    signal: dict,
    market_snapshot: dict,
) -> EntrySignal:
    """Evaluieren ein Entry-Signal und zurückgeben als EntrySignal.

    Args:
        config: Strategie-Konfiguration mit Schwellwerten.
        signal: Dict mit 'type', 'condition', 'price', 'strength'.
        market_snapshot: Dict mit Markt-Daten ('bid', 'ask', 'spread', 'volume').

    Returns:
        EntrySignal mit validiertem Preis, Typ und Bedingung.

    Raises:
        ValueError: Bei ungültigen Signal-Daten.
    """
    if not signal:
        raise ValueError("signal must not be empty")

    entry_type_raw = signal.get("type", "market").lower()
    try:
        entry_type = EntryType(entry_type_raw)
    except ValueError as exc:
        raise ValueError(
            f"Unknown entry type: {entry_type_raw}. "
            f"Valid: {[e.value for e in EntryType]}"
        ) from exc

    condition_raw = signal.get("condition", "").lower()
    try:
        condition = EntryCondition(condition_raw)
    except ValueError as exc:
        raise ValueError(
            f"Unknown entry condition: {condition_raw}. "
            f"Valid: {[c.value for c in EntryCondition]}"
        ) from exc

    price = signal.get("price")
    if price is None:
        raise ValueError("signal must include 'price'")

    # Markt-Snapshot Validierung
    bid = market_snapshot.get("bid", price)
    ask = market_snapshot.get("ask", price)
    spread = market_snapshot.get("spread", abs(ask - bid) if ask and bid else 0.0)

    # Validiere Spread gegen min_quality Schwellwert
    if spread > 0.0 and config.min_quality > 0.9:
        # Bei hoher Qualität-Anforderung muss der Spread klein sein
        mid = (bid + ask) / 2.0 if bid and ask else price
        if mid > 0.0 and spread / mid > 0.001:
            signal.setdefault("metadata", {})["spread_warning"] = True

    strength = signal.get("strength", 1.0)

    return EntrySignal(
        price=float(price),
        entry_type=entry_type,
        condition=condition,
        strength=float(strength),
        metadata=signal.get("metadata", {}),
    )
