"""Risk management — position sizing, drawdown limits, risk-adjusted returns.

Provides Kelly and ATR-based position sizing, drawdown monitoring,
and risk-adjusted return metrics (Sharpe, Sortino, Calmar).
"""

from __future__ import annotations

from .base import (
    PositionSizerConfig,
    RiskDecision,
    RiskGateResult,
    RiskGateType,
)
from .drawdown import DrawdownMonitor
from .position_sizing import ATRPositionSizer, KellyPositionSizer
from .risk_adjusted import RiskAdjustedReturns

__all__ = [
    "ATRPositionSizer",
    "DrawdownMonitor",
    "KellyPositionSizer",
    "PositionSizerConfig",
    "RiskAdjustedReturns",
    "RiskDecision",
    "RiskGateResult",
    "RiskGateType",
]
