"""Gemeinsame Domänenschemata — Trading Orchestra.

Alle Schemata nutzen Pydantic v2 mit strikter Typisierung.
Schema-Validierung ist Teil der CI-Prüfung (EPIC-01-WP03).
"""

from __future__ import annotations

from .agent_report import AgentReport, AgentStatus, EvidenceReference, InvalidationCondition
from .analysis_request import AnalysisMode, AnalysisRequest, AnalysisResult
from .final_decision import FinalDecision, FinalDecisionType
from .market_event import Candle, MarketEvent, NewsEvent, OrderBookSnapshot, Trade
from .risk_decision import RiskDecision, RiskGateResult, RiskGateType
from .source_metadata import SourceMetadata
from .trading_graph_state import TradingGraphState

__all__ = [
    "AgentReport",
    "AgentStatus",
    "AnalysisMode",
    "AnalysisRequest",
    "AnalysisResult",
    "Candle",
    "EvidenceReference",
    "FinalDecision",
    "FinalDecisionType",
    "InvalidationCondition",
    "MarketEvent",
    "NewsEvent",
    "OrderBookSnapshot",
    "RiskDecision",
    "RiskGateResult",
    "RiskGateType",
    "SourceMetadata",
    "Trade",
    "TradingGraphState",
]
