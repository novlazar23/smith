"""Forecast-Provider-Abstraktion für den TimesFM-Research-Spike.

Phase 1 ist bewusst modellfrei: Die Evolutions-Pipeline bekommt nur
einen ``ForecastProvider`` injiziert, Tests laufen gegen ``FakeProvider``.
Der echte TimesFM-Adapter wird erst in Phase 2 angefasst, wenn
Version, Lizenz und API-Pfad fixiert sind (siehe Plan).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TimesFMUnavailableError(RuntimeError):
    """Echter TimesFM-Pfad ist in der aktuellen Phase nicht verfügbar."""


@dataclass(frozen=True)
class QuantileForecast:
    """Quantil-Prognosen für die nächsten ``horizon`` Bars (Preisskala)."""

    median: np.ndarray
    q10: np.ndarray
    q90: np.ndarray


class ForecastProvider(Protocol):
    """Minimaler Provider-Vertrag für den Spike (keine Default-Dependency)."""

    def predict(self, context: np.ndarray, horizon: int) -> QuantileForecast:
        """Prognostiziert ``horizon`` Schritte ab dem letzten Context-Kurs."""
        ...


def provider_label(provider: ForecastProvider) -> str:
    """Stabiles Provider-Label für Reports/Cache-Keys."""
    label = getattr(provider, "label", None)
    if label is None:
        return type(provider).__name__
    return str(label)


@dataclass(frozen=True)
class FakeProvider:
    """Deterministischer Fake-Provider (Tests/Smoke, kein echtes Modell)."""

    label: str = "fake"
    drift_per_bar: float = 0.0002
    spread_per_bar: float = 0.0005

    def predict(self, context: np.ndarray, horizon: int) -> QuantileForecast:
        if horizon < 1:
            raise ValueError("horizon muss >= 1 sein")
        if context.size < 2:
            raise ValueError("Context braucht mindestens zwei Kurse")
        base = float(context[-1])
        if base <= 0.0:
            raise ValueError("letzter Context-Kurs muss positiv sein")
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        drift = self.drift_per_bar * steps
        spread = self.spread_per_bar * steps
        return QuantileForecast(
            median=base * (1.0 + drift),
            q10=base * (1.0 + drift - spread),
            q90=base * (1.0 + drift + spread),
        )


class TimesFMProvider:
    """Platzhalter für den echten TimesFM-Adapter (Phase 2).

    Bewusst kein ``import timesfm`` auf Modulebene: Der Spike darf die
    Evolutions-Pipeline nicht an eine optionale ML-Dependency koppeln.
    """

    label: str

    def __init__(self, model: str | None = None) -> None:
        self.model = model or "timesfm"
        self.label = f"timesfm:{self.model}"

    def predict(self, context: np.ndarray, horizon: int) -> QuantileForecast:
        try:
            import timesfm  # noqa: F401
        except ImportError as exc:
            raise TimesFMUnavailableError(
                "timesfm ist nicht installiert; Phase 1 nutzt FakeProvider. "
                "Installation erst nach Version/Lizenz-Pinning (Phase 2)."
            ) from exc
        raise TimesFMUnavailableError(
            "Echter TimesFM-Adapter ist in Phase 1 nicht verifiziert; "
            "nutze FakeProvider oder setze Phase 2 an."
        )


def load_timesfm_provider(model: str | None = None) -> ForecastProvider:
    """Liefert den konfigurierten Provider (Default: TimesFM-Platzhalter)."""
    return TimesFMProvider(model)
