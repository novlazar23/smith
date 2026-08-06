"""Basis-Klassen und Konstanten für Fibonacci-Analyse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FIBONACCI_RETRACEMENTS: list[float] = [0.236, 0.382, 0.5, 0.618, 0.786]
FIBONACCI_EXTENSIONS: list[float] = [1.272, 1.618]


@dataclass
class FibonacciPivot:
    """Ein Swing-Pivot-Preis mit Zeitstempel und Typ."""

    price: float
    time: int
    type: Literal["swing_high", "swing_low"]


@dataclass
class FibonacciArea:
    """Ein Fibonacci-Zonen-Bereich mit Metadaten."""

    lower: float
    upper: float
    level_types: list[str] = field(default_factory=list)
    source_timeframes: list[str] = field(default_factory=list)
    confluence_score: float = 0.0


@dataclass
class ConfluenceResult:
    """Ergebnis einer Fibonacci-Konfluenz-Analyse."""

    level: str
    score: float
    matching_prices: list[float] = field(default_factory=list)
