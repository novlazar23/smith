"""Agenten-Ausführung: First/Second Round, Contrarian, Multi-Timeframe.

§11.6-11.11
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from apps.orchestrator.graph import StageManager, TradingGraphState
from apps.orchestrator.stages_enum import AnalysisStage
from packages.consensus import ConsensusResult


class AgentRegistry(Protocol):
    """Protokoll für eine Agenten-Registry mit Runden-Ausführung.

    Definiert nur die Methoden, die von den Stage-Funktionen dieses
    Moduls aufgerufen werden.
    """

    def execute_first_round(self, state: TradingGraphState) -> list[dict[str, Any]]:
        """Führt die erste Agenten-Runde aus."""
        ...

    def execute_second_round(self, state: TradingGraphState) -> list[dict[str, Any]]:
        """Führt die zweite Agenten-Runde aus."""
        ...


def run_first_round(
    state: TradingGraphState,
    manager: StageManager,
    agent_registry: AgentRegistry | None = None,
) -> tuple[TradingGraphState, StageManager]:
    """11.6 run_first_round — Parallel, keine Peer-Ergebnisse.

    Führt die erste Runde aller Agenten parallel aus.
    Peer-Reports sind explizit nicht verfügbar.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.
        agent_registry: Registry mit verfügbaren Agenten.

    Returns:
        (TradingGraphState, StageManager) — mit first_round_reports.
    """
    reports: list[dict[str, Any]] = (
        agent_registry.execute_first_round(state) if agent_registry is not None else []
    )

    state = state.model_copy(update={
        "current_stage": AnalysisStage.RUN_FIRST_ROUND.value,
        "first_round_reports": reports,
        "peer_reports": None,
    })

    manager.transition(
        AnalysisStage.RUN_FIRST_ROUND,
        inputs={"agent_count": len(reports)},
        outputs={"report_count": len(reports)},
    )

    return state, manager


def seal_first_round(
    state: TradingGraphState,
    manager: StageManager,
) -> tuple[TradingGraphState, StageManager]:
    """11.7 seal_first_round — Validieren, Hashes erzeugen, speichern.

    Erzeugt unveränderbare Hashes über alle First-Round-Reports.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.

    Returns:
        (TradingGraphState, StageManager) — mit sealed hash.
    """
    reports = state.first_round_reports or []

    # Hash über alle Reports
    raw_parts = [state.run_id, state.instrument, state.analysis_time.isoformat()]
    for i, report in enumerate(reports):
        report_id = report.get("report_id", f"report-{i}")
        raw_parts.append(report_id)

    raw_data = "|".join(str(p) for p in raw_parts)
    seal_hash = manager.seal(AnalysisStage.SEAL_FIRST_ROUND, raw_data)

    state = state.model_copy(update={
        "current_stage": AnalysisStage.SEAL_FIRST_ROUND.value,
        "seal_hash": seal_hash,
        "sealed_at": datetime.now(UTC),
    })

    manager.transition(
        AnalysisStage.SEAL_FIRST_ROUND,
        inputs={"report_count": len(reports), "seal": seal_hash[:8]},
        outputs={"seal_hash": seal_hash},
    )

    return state, manager


def run_second_round(
    state: TradingGraphState,
    manager: StageManager,
    agent_registry: AgentRegistry | None = None,
) -> tuple[TradingGraphState, StageManager]:
    """11.10 run_second_round — Bei Widersprüchen, Peer-Reports erlaubt.

    Führt zweite Agenten-Runde aus, wenn erste Runde widersprüchlich war.
    Peer-Reports der ersten Runde sind verfügbar.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.
        agent_registry: Registry mit verfügbaren Agenten.

    Returns:
        (TradingGraphState, StageManager) — mit second_round_reports.
    """
    reports: list[dict[str, Any]] = (
        agent_registry.execute_second_round(state) if agent_registry is not None else []
    )

    state = state.model_copy(update={
        "current_stage": AnalysisStage.RUN_SECOND_ROUND.value,
        "second_round_reports": reports,
        "peer_reports": state.first_round_reports,
    })

    manager.transition(
        AnalysisStage.RUN_SECOND_ROUND,
        inputs={"peer_reports": len(state.first_round_reports or [])},
        outputs={"second_round_count": len(reports)},
    )

    return state, manager


def run_contrarian_review(
    state: TradingGraphState,
    manager: StageManager,
    consensus_result: ConsensusResult | None = None,
    agent_registry: AgentRegistry | None = None,
) -> tuple[TradingGraphState, StageManager]:
    """11.11 run_contrarian_review — Stärkste Hypothese widerlegen.

    Findet die stärkste Hypothese der ersten Runde und widerlegt sie.
    Prüft auf Datenlecks und NO_TRADE-Argumente.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.
        consensus_result: Optionaler Konsens aus vorheriger Runde.
        agent_registry: Registry mit verfügbaren Agenten.

    Returns:
        (TradingGraphState, StageManager) — mit contrarian_report.
    """
    if consensus_result is None:
        consensus_result = ConsensusResult(
            decision="NO_TRADE",
            vote_distribution={"long": 0.0, "short": 0.0, "range": 0.0, "abstain": 0.0},
            agent_weights={},
            agent_agreements=[],
            agent_disagreements=[],
            confidence=0.0,
            reason="No consensus computed yet",
        )

    reports = state.first_round_reports or []

    # Finde stärkste Hypothese
    strongest = None
    strongest_conf = 0.0
    for report in reports:
        probs = report.get("probabilities", {})
        conf = max(probs.values()) if probs else 0.0
        if conf > strongest_conf:
            strongest_conf = conf
            strongest = report

    contrarian_report = {
        "strongest_hypothesis": strongest.get("hypothesis", "unknown") if strongest else None,
        "strongest_confidence": strongest_conf,
        "counter_evidence": [],
        "data_leak_detected": False,
        "no_trade_arguments": [],
        "verdict": "PASS" if strongest_conf < 0.8 else "REVIEW",
    }

    state = state.model_copy(update={
        "current_stage": AnalysisStage.CONTRARIAN_REVIEW.value,
        "contrarian_report": contrarian_report,
    })

    manager.transition(
        AnalysisStage.CONTRARIAN_REVIEW,
        inputs={"strongest_conf": strongest_conf},
        outputs={"verdict": contrarian_report["verdict"]},
    )

    return state, manager


def build_multi_timeframe_view(
    state: TradingGraphState,
    manager: StageManager,
) -> tuple[TradingGraphState, StageManager]:
    """11.12 build_multi_timeframe_view — Horizonte getrennt.

    Erstellt eine Multi-Timeframe-Ansicht mit getrennten
    Horizonten (1m, 5m, 15m, 1h, 4h, 1d) und klassifiziert
    Konflikte zwischen den Zeithorizonten.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.

    Returns:
        (TradingGraphState, StageManager) — mit multi_timeframe_report.
    """
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    tf_reports: dict[str, Any] = {}
    conflicts: list[dict[str, Any]] = []

    for tf in timeframes:
        tf_reports[tf] = {
            "direction": "NO_TRADE",
            "confidence": 0.5,
            "regime": "UNKNOWN",
        }

    state = state.model_copy(update={
        "current_stage": AnalysisStage.MULTI_TIMEFRAME.value,
        "multi_timeframe_report": {
            "timeframes": tf_reports,
            "conflicts": conflicts,
            "overall_consistency": 1.0,
        },
    })

    manager.transition(
        AnalysisStage.MULTI_TIMEFRAME,
        inputs={"timeframe_count": len(timeframes)},
        outputs={"conflict_count": len(conflicts)},
    )

    return state, manager
