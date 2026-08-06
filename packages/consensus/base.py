"""Base types for the consensus module.

Defines vote directions, weight configuration, and the consensus result
data class used throughout the consensus mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VoteDirection(StrEnum):
    """Richtung einer Agenten-Abstimmung."""

    LONG = "long"
    SHORT = "short"
    RANGE = "range"
    ABSTAIN = "abstain"


class WeightConfig:
    """Konfiguration der Gewichtungsregeln für den Konsens."""

    def __init__(
        self,
        base_weight: float = 1.0,
        status_multiplier: dict[str, float] | None = None,
        min_consensus_threshold: float = 0.5,
        max_agent_divergence: float = 0.7,
    ) -> None:
        self.base_weight = base_weight
        self.status_multiplier = status_multiplier or {
            "active": 1.0,
            "shadow": 0.5,
            "degraded": 0.3,
            "quarantined": 0.0,
            "disabled": 0.0,
        }
        self.min_consensus_threshold = min_consensus_threshold
        self.max_agent_divergence = max_agent_divergence


class ConsensusDecision(StrEnum):
    """Ergebnis der Konsensfindung."""

    LONG_BIAS = "LONG_BIAS"
    SHORT_BIAS = "SHORT_BIAS"
    RANGE = "RANGE"
    NO_TRADE = "NO_TRADE"


@dataclass
class ConsensusResult:
    """Ergebnis einer Konsensberechnung."""

    decision: ConsensusDecision
    vote_distribution: dict[VoteDirection, float]
    agent_weights: dict[str, float]
    agent_agreements: list[str]
    agent_disagreements: list[str]
    confidence: float
    reason: str
