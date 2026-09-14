from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
    InvalidationCondition,
)

# Parameterraum pro evolvierbarem Agent: Feldname →
# (Typ "int" | "float", Untergrenze, Obergrenze, Schrittweite)
type ParamSpace = dict[str, tuple[str, float, float, float]]


def _coerce(name: str, kind: str, raw: object) -> float | int:
    """Wandelt einen JSON-Wert fail-closed in den deklarierten Parametertyp um.

    Raises:
        ValueError: Wert ist kein Zahl (Boolean zählt nicht) — korruptes Artefakt.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"Parameter {name} erwartet einen Zahlwert, bekam {type(raw).__name__}: {raw!r}")
    return int(raw) if kind == "int" else float(raw)


@dataclass(frozen=True, slots=True)
class BaseParams:
    """Basis für evolvierbare Agenten-Parameter.

    Subklassen (z. B. ``TrendParams``) sind frozen Dataclasses, deren
    Felder die bisherigen (gehärteten) Agenten-Konstanten als Defaults
    tragen und deren ``PARAM_SPACE`` die mutierbare Spanne je Feld
    beschreibt. ``to_dict``/``from_dict`` ermöglichen die
    Serialisierung nach ``champion_configs.json`` (unbekannte Schlüssel
    werden ignoriert, fehlende Felder bleiben auf ihren Defaults).
    """

    PARAM_SPACE: ClassVar[ParamSpace]

    def to_dict(self) -> dict[str, float | int]:
        """Parametersatz als JSON-serialisierbares Dict (Feldreihenfolge)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        """Baut eine Instanz aus einem Dict (unbekannte Schlüssel ignoriert)."""
        space: ParamSpace = cls.PARAM_SPACE
        values: dict[str, float | int] = {
            name: _coerce(name, space[name][0], raw)
            for name, raw in d.items()
            if name in space
        }
        return cls(**values)


class AgentType(StrEnum):
    """Kategorien von Analyse-Agenten."""

    INDICATOR = "indicator"
    REGIME = "regime"
    CHART = "chart"
    ORDERFLOW = "orderflow"
    PATTERN = "pattern"
    FIBONACCI = "fibonacci"
    ELLIOTT = "elliott"
    HISTORICAL_ANALOGY = "historical_analogy"
    NEWS = "news"
    CROSS_MARKET = "cross_market"
    ANOMALY = "anomaly"
    CONTRARIAN = "contrarian"


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Konfiguration für einen Analyse-Agenten."""

    agent_id: str
    agent_version: str = "0.1.0"
    agent_type: AgentType = AgentType.INDICATOR
    instrument: str = ""
    horizon: str = "1h"
    status: AgentStatus = AgentStatus.SHADOW


class BaseAgent(ABC):
    """Abstrakte Basisklasse für alle Analyse-Agenten.

    Jeder Agent produziert einen standardisierten AgentReport mit
    Wahrscheinlichkeiten (up/down/range), Evidenz und Invalidierungen.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    @property
    def agent_id(self) -> str:
        """Eindeutige ID dieses Agenten."""
        return self._config.agent_id

    @property
    def config(self) -> AgentConfig:
        """Konfiguration dieses Agenten."""
        return self._config

    def _generate_report_id(self) -> str:
        """Erzeugt eine eindeutige Report-ID als UUID4."""
        return uuid.uuid4().hex

    def _make_evidence(
        self,
        feature: str,
        value: str,
        direction: str,
        relevance: float,
    ) -> EvidenceReference:
        """Erstellt eine Evidenzreferenz für den Agentenbericht."""
        ref_id = f"{self.agent_id}:{feature}"
        return EvidenceReference(
            reference=ref_id,
            feature=feature,
            value=value,
            direction=direction,
            relevance=relevance,
        )

    def _make_invalidations(
        self,
        condition: str,
        indicator: str,
        threshold: float,
        direction: str,
    ) -> InvalidationCondition:
        """Erstellt eine Invalidierungsbedingung für den Agentenbericht."""
        return InvalidationCondition(
            condition=condition,
            indicator=indicator,
            threshold=threshold,
            direction=direction,
        )

    @abstractmethod
    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """Führt die Analyse der Eingabedaten durch.

        Args:
            data: Dict mit erforderlichen NDArray-Schlüsseln (spezifisch pro Agent).

        Returns:
            AgentReport mit Wahrscheinlichkeiten, Evidenz und Invalidierungen.

        Raises:
            ValueError: Wenn erforderliche Schlüssel fehlen.
        """
        ...
