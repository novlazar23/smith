"""Consensus mechanism — combines AgentReports with history-based weighting."""

from .base import ConsensusDecision, ConsensusResult, VoteDirection, WeightConfig
from .historical import HistoricalWeightTracker
from .weighted import WeightedConsensusEngine

__all__ = [
    "ConsensusDecision",
    "ConsensusResult",
    "HistoricalWeightTracker",
    "VoteDirection",
    "WeightConfig",
    "WeightedConsensusEngine",
]
