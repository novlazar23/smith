"""Target-Berechnung für die Strategie-Engine.

Berechnet Take-Profit-Level (TP1/TP2/TP3) und Stopp-Loss-Levels
basierend auf Volatilität, ATR und direction. Enthält auch
probabilistische Target-before-Stop- und MFE/MAE-Schätzungen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TargetType(StrEnum):
    """Typ eines Target-Level."""

    TP1 = "tp1"
    TP2 = "tp2"
    TP3 = "tp3"
    STOP_LIMIT = "stop_limit"


@dataclass(frozen=True)
class TargetLevel:
    """Ein einzelnes Target- oder Stopp-Level."""

    price: float
    type: TargetType
    probability: float = 0.0
    atr_distance: float = 0.0
    metadata: dict = field(default_factory=dict)


def calculate_targets(
    entry_price: float,
    volatility: float,
    atr: float,
    direction: str,
    tp1_atr: float = 1.5,
    tp2_atr: float = 3.0,
    tp3_atr: float = 5.0,
    stop_atr: float = 2.0,
) -> list[TargetLevel]:
    """Berechnen Take-Profit- und Stopp-Loss-Levels.

    Die Targets werden als multiplikatives Vielfaches des ATR
    vom Entry-Preis aus berechnet. TP1 hat die höchste
    Eintrittswahrscheinlichkeit, TP3 die niedrigste.

    Args:
        entry_price: Current entry price.
        volatility: Current price volatility (e.g. std of returns).
        atr: Average True Range for the instrument.
        direction: "long" or "short".
        tp1_atr: ATR multiplier for TP1 (default 1.5).
        tp2_atr: ATR multiplier for TP2 (default 3.0).
        tp3_atr: ATR multiplier for TP3 (default 5.0).
        stop_atr: ATR multiplier for stop loss (default 2.0).

    Returns:
        Liste von TargetLevel-Objekten (TP1, TP2, TP3, STOP_LIMIT).

    Raises:
        ValueError: Bei ungültigen Parametern.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if atr <= 0:
        raise ValueError("atr must be positive")
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction}")

    is_long = direction == "long"

    def _direction_adjust(amount: float) -> float:
        return amount if is_long else -amount

    # Target-Preise berechnen
    tp1_price = entry_price + _direction_adjust(atr * tp1_atr)
    tp2_price = entry_price + _direction_adjust(atr * tp2_atr)
    tp3_price = entry_price + _direction_adjust(atr * tp3_atr)
    stop_price = entry_price - _direction_adjust(atr * stop_atr)

    # Wahrscheinlichkeiten basierend auf ATR-Abstand
    # Nähere Targets haben höhere Wahrscheinlichkeit
    tp1_prob = max(0.0, min(1.0, 1.0 - tp1_atr / 10.0))
    tp2_prob = max(0.0, min(1.0, 1.0 - tp2_atr / 10.0))
    tp3_prob = max(0.0, min(1.0, 1.0 - tp3_atr / 10.0))

    # Probabilities müssen monoton fallen
    tp3_prob = min(tp3_prob, tp2_prob)
    tp2_prob = min(tp2_prob, tp1_prob)

    # Target-before-Stop Wahrscheinlichkeit
    # Je grösser ATR-Distanz zum Target vs. Stop, desto unwahrscheinlicher
    stop_distance = atr * stop_atr
    tp1_distance = atr * tp1_atr
    prob_target_before_stop = (
        tp1_distance / (tp1_distance + stop_distance)
        if (tp1_distance + stop_distance) > 0
        else 0.5
    )
    prob_target_before_stop = max(0.0, min(1.0, prob_target_before_stop))

    targets = [
        TargetLevel(
            price=round(tp1_price, 8),
            type=TargetType.TP1,
            probability=round(tp1_prob, 4),
            atr_distance=tp1_atr,
            metadata={
                "prob_target_before_stop": round(prob_target_before_stop, 4),
                "prob_stop_before_target": round(
                    1.0 - prob_target_before_stop, 4
                ),
            },
        ),
        TargetLevel(
            price=round(tp2_price, 8),
            type=TargetType.TP2,
            probability=round(tp2_prob, 4),
            atr_distance=tp2_atr,
        ),
        TargetLevel(
            price=round(tp3_price, 8),
            type=TargetType.TP3,
            probability=round(tp3_prob, 4),
            atr_distance=tp3_atr,
        ),
        TargetLevel(
            price=round(stop_price, 8),
            type=TargetType.STOP_LIMIT,
            probability=round(1.0 - tp1_prob, 4),
            atr_distance=stop_atr,
            metadata={"is_stop": True},
        ),
    ]

    return targets


def estimate_mfe_mae(
    entry_price: float,
    targets: list[TargetLevel],
    volatility: float,
    direction: str,
) -> tuple[float, float]:
    """Schätzen Maximum Favorable/Adverse Excursion.

    MFE (Maximum Favorable Excursion): Maximaler günstiger Preisverlauf.
    MAE (Maximum Adverse Excursion): Maximaler ungünstiger Preisverlauf.

    Basierend auf normaler Preisverteilung und Volatilität.

    Args:
        entry_price: Entry price.
        targets: Liste der TargetLevels (muss STOP_LIMIT enthalten).
        volatility: Current volatility.
        direction: "long" or "short".

    Returns:
        Tuple von (expected_mae, expected_mfe).
    """
    is_long = direction == "long"

    # Finde Stop-Price aus targets
    max_target_price = None
    for t in targets:
        if t.type == TargetType.STOP_LIMIT:
            _stop_price = t.price
        else:
            if max_target_price is None or (
                is_long and t.price > max_target_price
            ) or (not is_long and t.price < max_target_price):
                max_target_price = t.price

    # MAE: Erwarteter ungünstiger Ausreisser (3 Sigma)
    mae_abs = volatility * 3.0 if volatility > 0 else abs(entry_price) * 0.03

    # MFE: Erwarteter günstiger Ausreisser (3 Sigma, capped at target)
    mfe_abs = volatility * 3.0 if volatility > 0 else abs(entry_price) * 0.03
    if max_target_price is not None:
        potential_reward = (
            max_target_price - entry_price
            if is_long
            else entry_price - max_target_price
        )
        mfe_abs = min(mfe_abs, potential_reward * 1.5)

    mae = -mae_abs if is_long else mae_abs
    mfe = mfe_abs if is_long else -mfe_abs

    return round(mae, 8), round(mfe, 8)


def calculate_prob_target_before_stop(
    tp_distance: float,
    stop_distance: float,
    volatility: float,
) -> float:
    """Berechnen Wahrscheinlichkeit Target vor Stop erreicht.

    Verwendet ein relatives Distanz-Verhältnis als probabilistische
    Schätzung unter der Annahme einer random-walk-basierten
    Preisbewegung.

    Args:
        tp_distance: ATR-Distanz zum Take-Profit.
        stop_distance: ATR-Distanz zum Stopp-Loss.
        volatility: Current price volatility.

    Returns:
        Wahrscheinlichkeit Target vor Stop (0.0-1.0).
    """
    if tp_distance <= 0 or stop_distance <= 0:
        return 0.5

    # Random walk: P(hit +tp before -stop) = stop/(tp+stop)
    # TP nahe (klein tp), Stop weit (gross stop) → hohe Wahrscheinlichkeit
    ratio = stop_distance / (tp_distance + stop_distance)

    # Volatilitäts-Bias: höhere Volatilität → mehr Rauschen → näher an 0.5
    volatility_factor = min(1.0, volatility / 0.10) if volatility > 0 else 0.5
    bias = volatility_factor * 0.5

    result = ratio * (1.0 - bias) + 0.5 * bias
    return max(0.0, min(1.0, round(result, 4)))
