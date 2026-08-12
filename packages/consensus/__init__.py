"""Consensus mechanism — combines AgentReports with history-based weighting."""

from .aggregation import CalibratedConsensusAggregator, CalibratedConsensusResult
from .base import ConsensusDecision, ConsensusResult, VoteDirection, WeightConfig
from .dependency_analysis import (
    AgentDependency,
    DependencyAnalysisResult,
    DependencyAnalyzer,
)
from .historical import HistoricalWeightTracker
from .weighted import WeightedConsensusEngine

__all__ = [
    "AgentDependency",
    "CalibratedConsensusAggregator",
    "CalibratedConsensusResult",
    "ConsensusDecision",
    "ConsensusResult",
    "DependencyAnalysisResult",
    "DependencyAnalyzer",
    "HistoricalWeightTracker",
    "VoteDirection",
    "WeightConfig",
    "WeightedConsensusEngine",
]
