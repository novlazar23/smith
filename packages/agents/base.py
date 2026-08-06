from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
    InvalidationCondition,
)


class AgentType(StrEnum):
    """Kategorien von Analyse-Agenten."""

    INDICATOR = "indicator"
    REGIME = "regime"
    CHART = "chart"
    ORDERFLOW = "orderflow"


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
