"""Data models for the contrarian agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContrarianConfig:
    """Konfiguration fuer den Contrarian-Agenten."""

    agent_id: str
    agent_version: str = "0.1.0"
    min_minority_ratio: float = 0.2


@dataclass(frozen=True, slots=True)
class ContrarianHypothesis:
    """Gegenhypothesen, die vom Contrarian-Agenten erzeugt wird."""

    counter_argument: str
    confidence: float
    evidence: list[str]
    majority_direction: str
    minority_direction: str
