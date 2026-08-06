"""Basis-Klassen und Typen für Regime Detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class MarketRegime(StrEnum):
    """Mögliche Marktregime."""

    BULL = "bull"
    BEAR = "bear"
    CHOPPY = "choppy"


class RegimeResult:
    """Ergebnis der Regime-Erkennung."""

    def __init__(
        self,
        regime: MarketRegime,
        confidence: float,
        scores: dict[MarketRegime, float],
        metadata: dict | None = None,
    ) -> None:
        self.regime = regime
        self.confidence = confidence
        self.scores = scores
        self.metadata = metadata or {}


class BaseRegimeDetector(ABC):
    """Abstrakte Basisklasse für Regime-Detektoren."""

    name: str

    @abstractmethod
    def detect(self, data: dict[str, NDArray[np.float64]]) -> RegimeResult:
        """Erkennt das aktuelle Marktregime aus Marktdaten."""
