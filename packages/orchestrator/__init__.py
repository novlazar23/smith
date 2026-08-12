"""Orchestrator package — Graph, First Round, Seal, Second Round.

Dieses Paket implementiert den Konsens-Pipeline auf Package-Ebene:
  1. first_round — parallele Agenten-Ausführung OHNE Peer-Reports
  2. seal        — SHA-256 Hash-Siegel pro Report (unveränderbar)
  3. second_round — Agenten sehen Round-1-Zusammenfassung + Gegenreview
  4. consensus     — gewichteter Konsens, Shadow-Gewicht 0.0

Öffentliche APIs:
  - TradingGraphState, OrchestratorGraph (graph.py)
  - run_first_round, FirstRoundReport (graph.py)
  - SealRecord, seal_first_round (seal.py)
  - RoundSummary, build_round_summary, run_second_round (second_round.py)
  - OrchestratorPipeline, OrchestratorPipelineResult (pipeline.py)
"""

from __future__ import annotations

from packages.orchestrator.first_round import run_first_round
from packages.orchestrator.graph import (
    OrchestratorGraph,
    PipelineStage,
    TradingGraphState,
    create_initial_state,
)
from packages.orchestrator.pipeline import (
    OrchestratorPipeline,
    OrchestratorPipelineResult,
)
from packages.orchestrator.seal import (
    SealRecord,
    seal_first_round,
)
from packages.orchestrator.second_round import (
    RoundContext,
    RoundSummary,
    build_round_summary,
    run_second_round,
)

__all__ = [
    "OrchestratorGraph",
    "OrchestratorPipeline",
    "OrchestratorPipelineResult",
    "PipelineStage",
    "RoundContext",
    "RoundSummary",
    "SealRecord",
    "TradingGraphState",
    "build_round_summary",
    "create_initial_state",
    "run_first_round",
    "run_second_round",
    "seal_first_round",
]
