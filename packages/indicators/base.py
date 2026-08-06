"""Base Indicator — Basisklasse für technische Indikatoren.

Alle Indikatoren erben von Indicator und implementieren compute().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class IndicatorResult:
    """Ergebnis einer Indikator-Berechnung."""

    name: str
    values: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)


class Indicator(ABC):
    """Basisklasse für technische Indikatoren.

    Alle Indikatoren müssen compute() implementieren.
    """

    name: str = "base"
    min_periods: int = 1

    @abstractmethod
    def compute(self, data: dict[str, NDArray[np.float64]]) -> IndicatorResult:
        """Berechnet den Indikator aus Marktdaten.

        Args:
            data: Dict mit 'open', 'high', 'low', 'close', 'volume' als NDArrays.

        Returns:
            IndicatorResult mit Namen, Werten und Metadaten.
        """

    @staticmethod
    def _validate_data(data: dict[str, NDArray[np.float64]], required: list[str]) -> None:
        """Prüft ob alle erforderlichen Schlüssel vorhanden sind."""
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required data keys: {missing}")

    @staticmethod
    def _check_lengths(data: dict[str, NDArray[np.float64]]) -> None:
        """Prüft ob alle Arrays gleiche Länge haben."""
        lengths = {k: len(v) for k, v in data.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Inconsistent array lengths: {lengths}")
