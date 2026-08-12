"""Multi-Timeframe Agent — Cross-Timeframe Consensus.

Analysiert Signale ueber mehrere Zeitrahmen (1m, 5m, 15m, 1h, 4h, 1d)
und erzeugt einen aggregierten Bericht mit uebergreifendem Konsens.
"""

from __future__ import annotations

from .agent import MultiTimeframeAgent
from .models import MultiTimeframeConfig, MultiTimeframeReport, TimeframeSignal

__all__ = [
    "MultiTimeframeAgent",
    "MultiTimeframeConfig",
    "MultiTimeframeReport",
    "TimeframeSignal",
]
