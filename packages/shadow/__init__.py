"""Shadow Mode — Shadow-Mode Decision Engine, Metrics, Comparator.

Dieses Paket implementiert den Shadow-Modus für EPIC-16:
  - Gleiche Orchestrator-Pipeline wie Production
  - Keine Order-Ausführung, nur Aufzeichnung
  - Brier Score und Kalibrierungs-Metriken
  - Vergleich Shadow vs Paper Execution

Öffentliche APIs:
    - ShadowEngine (engine.py) — Shadow-Mode Decision Engine
    - ShadowMetrics (metrics.py) — Quality Metrics (Brier, Calibration, PnL)
    - ShadowPaperComparison (comparator.py) — Shadow vs Paper comparison
"""

from __future__ import annotations

from .comparator import (
    ComparisonResult,
    PaperDecision,
    ShadowDecisionRecord,
    ShadowPaperComparison,
)
from .engine import ShadowDecision, ShadowEngine
from .metrics import (
    ShadowBrierScore,
    ShadowCalibration,
    ShadowMetrics,
    ShadowPnL,
    ShadowPnLRecord,
)

__all__ = [
    "ComparisonResult",
    "PaperDecision",
    "ShadowBrierScore",
    "ShadowCalibration",
    "ShadowDecision",
    "ShadowDecisionRecord",
    "ShadowEngine",
    "ShadowMetrics",
    "ShadowPaperComparison",
    "ShadowPnL",
    "ShadowPnLRecord",
]
