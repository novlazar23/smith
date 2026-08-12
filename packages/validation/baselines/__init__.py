"""Baseline Strategies for Historical Validation.

Five simple baselines that produce predictions for comparison:
- BuyHold: always predict UP
- MACross: moving average crossover signal
- Momentum: recent return direction
- RSI: overbought/oversold signal
- Regime: trend vs mean-reverting regime

All baselines produce predictions in the same format as agent reports:
- probabilities: dict[str, float] summing to 1.0
- confidence: float in [0, 1]
"""

from __future__ import annotations

from .base import Baseline, BaselinePrediction
from .buy_hold import BuyHoldBaseline
from .ma_cross import MACrossBaseline
from .momentum import MomentumBaseline
from .regime import RegimeBaseline
from .rsi import RSIBaseline

__all__ = [
    "Baseline",
    "BaselinePrediction",
    "BuyHoldBaseline",
    "MACrossBaseline",
    "MomentumBaseline",
    "RSIBaseline",
    "RegimeBaseline",
]
