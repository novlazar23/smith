"""Basis-Klassen und Typen für Chart-Strukturerkennung."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


@dataclass
class SupportResistanceLevel:
    """Ergebnis einer Support-/Resistance-Level-Erkennung."""

    price: float
    level_type: str  # "support" or "resistance"
    strength: float
    touch_count: int


@dataclass
class SwingPivot:
    """Ergebnis einer Swing-Pivot-Erkennung."""

    price: float
    time: int
    direction: Literal["high", "low"]
    quality_score: float


class ChartPattern(StrEnum):
    """Mögliche Chart-Muster."""

    HH = "hh"
    HL = "hl"
    LH = "lh"
    LL = "ll"
    BOS = "bos"
    CHoCH = "choch"
    RANGE = "range"
    BREAKOUT = "breakout"
    FAILED_BREAKOUT = "failed_breakout"


class Signal(StrEnum):
    """Handels-Signale aus Struktur-Analyse."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    NEUTRAL = "neutral"
    OUTLIER = "outlier"


@dataclass
class ChartStructureResult:
    """Ergebnis der Chart-Strukturerkennung."""

    patterns: list[ChartPattern]
    pivots: list[SwingPivot]
    metadata: dict = field(default_factory=dict)
