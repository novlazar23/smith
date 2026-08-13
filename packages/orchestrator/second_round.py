"""Second-Round-Modul — Agenten mit Round-1-Zusammenfassung.

Im Gegensatz zu Round 1 sehen die Agents hier:
  - Round-1-Zusammenfassung (NICHT einzelne Reports)
  - Kontrarian-Review
  - Cross-Market-Kontext

Dies verhindert Peer-Kontamination und begrenzt den Peer-Einfluss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from packages.consensus import ConsensusDecision, ConsensusResult, VoteDirection
from packages.orchestrator.graph import (
    OrchestratorGraph,
    PipelineStage,
    TradingGraphState,
)
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
)


@dataclass
class RoundSummary:
    """Zusammenfassung der First-Round-Ergebnisse.

    direction: Ueberwiegende Richtung (LONG_BIAS, SHORT_BIAS, RANGE, NO_TRADE)
    confidence: Durchschnittliche Konfidenz aller Reports
    dissent_score: Mass fuer Uneinigkeit (0 = alle gleich, 1 = total uneinig)
    agent_agreements: Liste der Agenten-IDs die uebereinstimmen
    agent_disagreements: Liste der Agenten-IDs die abweichen
    """

    direction: str = "NO_TRADE"
    confidence: float = 0.0
    dissent_score: float = 0.0
    agent_agreements: list[str] = field(default_factory=list)
    agent_disagreements: list[str] = field(default_factory=list)


def build_round_summary(
    reports: list[AgentReport],
) -> RoundSummary:
    """Erzeugt eine Zusammenfassung aus First-Round-Reports.

    Die Zusammenfassung ist AGGREGIERT — keine einzelnen Reports.

    Args:
        reports: Liste von AgentReports aus Round 1.

    Returns:
        RoundSummary mit aggregierten Werten.
    """
    if not reports:
        return RoundSummary()

    # Zaelle Votes nach Richtung
    up_count = 0
    down_count = 0
    range_count = 0
    total_confidence = 0.0
    max_confidence = 0.0

    for report in reports:
        probs = report.probabilities or {}
        up = probs.get("up", 0.0)
        down = probs.get("down", 0.0)
        range_v = probs.get("range", 0.0)

        if up > down and up > range_v:
            up_count += 1
        elif down > up and down > range_v:
            down_count += 1
        else:
            range_count += 1

        conf = report.raw_confidence or 0.0
        total_confidence += conf
        max_confidence = max(max_confidence, conf)

    avg_confidence = total_confidence / len(reports)

    # Bestimme Richtung
    if up_count > down_count and up_count > range_count:
        direction = "LONG_BIAS"
    elif down_count > up_count and down_count > range_count:
        direction = "SHORT_BIAS"
    elif range_count >= up_count and range_count >= down_count:
        direction = "RANGE"
    else:
        direction = "NO_TRADE"

    # Dissent: Masse fuer Uneinigkeit
    total = len(reports)
    if total <= 1:
        dissent_score = 0.0
    else:
        # Je gleichmaessiger verteilt, hoeher der Dissent
        max_count = max(up_count, down_count, range_count)
        dissent_score = 1.0 - (max_count / total)

    return RoundSummary(
        direction=direction,
        confidence=round(avg_confidence, 4),
        dissent_score=round(dissent_score, 4),
        agent_agreements=[],
        agent_disagreements=[],
    )


@dataclass
class RoundContext:
    """Kontext, den Second-Round-Agenten zur Verfuegung gestellt bekommen.

    Erstellt aus build_round_summary — aggregiert, NICHT rohe Reports.
    """

    first_round_summary: dict[str, Any]
    contrarian_review: dict[str, Any] | None = None
    cross_market_context: dict[str, Any] | None = None
    market_snapshot: dict[str, Any] = field(default_factory=dict)
    feature_snapshot_id: str | None = None
    regime_report: dict[str, Any] | None = None


def _build_round_context(
    state: TradingGraphState,
    summary: RoundSummary,
    contrarian_review: dict[str, Any] | None = None,
) -> RoundContext:
    """Erzeugt den RoundContext aus State und Summary.

    Args:
        state: Aktueller Graphzustand.
        summary: Zusammenfassung von Round 1.
        contrarian_review: Optionale Gegenreview.

    Returns:
        RoundContext mit allen verfuegbaren Informationen.
    """
    summary_dict: dict[str, Any] = {
        "direction": summary.direction,
        "confidence": summary.confidence,
        "dissent_score": summary.dissent_score,
        "agent_count": len(state.first_round_reports),
    }

    return RoundContext(
        first_round_summary=summary_dict,
        contrarian_review=contrarian_review,
        cross_market_context=None,
        market_snapshot=state.market_snapshot,
        feature_snapshot_id=state.feature_snapshot_id,
        regime_report=state.regime_report,
    )


def run_second_round(
    state: TradingGraphState,
    graph: OrchestratorGraph,
    agents: list[Any],
    market_data: dict[str, Any],
    contrarian_review: dict[str, Any] | None = None,
    cross_market_context: dict[str, Any] | None = None,
) -> tuple[TradingGraphState, OrchestratorGraph, list[AgentReport]]:
    """Fuehrt Round 2 aus: Agenten mit Round-1-Zusammenfassung.

    Wichtig: Agenten bekommen KEINE einzelnen Reports — nur die
    Zusammenfassung. Dies verhindert Peer-Kontamination.

    Args:
        state: Aktueller Graphzustand mit populated seal_records.
        graph: OrchestratorGraph fuer Stage-Management.
        agents: Liste von Agenten mit ``analyze_with_context`` Methode.
        market_data: Rohdaten.
        contrarian_review: Optionale Gegenreview der staerksten Hypothese.
        cross_market_context: Optionale Cross-Market-Daten.

    Returns:
        (TradingGraphState, OrchestratorGraph, list[AgentReport])
        mit populated second_round_reports.

    Raises:
        ValueError: Wenn first_round_reports oder seal_records leer sind.
    """
    if not state.first_round_reports:
        raise ValueError(
            "run_second_round: first_round_reports must be populated"
        )
    if not state.seal_records:
        raise ValueError(
            "run_second_round: seal_records must be populated"
        )
    if not agents:
        raise ValueError("run_second_round: at least one agent is required")

    # Baue Summary aus Round 1 — aggregiert, NICHT einzelne Reports
    summary = build_round_summary(state.first_round_reports)

    # Baue RoundContext
    context = _build_round_context(
        state,
        summary,
        contrarian_review=contrarian_review,
    )
    if cross_market_context:
        context.cross_market_context = cross_market_context

    # Jeder Agent liefert einen AgentReport basierend auf dem Context
    reports: list[AgentReport] = []
    for agent in agents:
        # Jede analyze-Methode muss den Context verarbeiten
        report = agent.analyze_with_context(context, market_data)
        reports.append(report)

    # RoundSummary im State speichern
    summary_dict = {
        "direction": summary.direction,
        "confidence": summary.confidence,
        "dissent_score": summary.dissent_score,
        "agent_count": len(state.first_round_reports),
    }

    new_state = state.__class__(
        run_id=state.run_id,
        instrument=state.instrument,
        market_snapshot=state.market_snapshot,
        first_round_reports=state.first_round_reports,
        first_round_hash=state.first_round_hash,
        seal_records=state.seal_records,
        second_round_reports=reports,
        round_summary=summary_dict,
        current_stage=PipelineStage.SECOND_ROUND.value,
        errors=[],
        warnings=[],
    )

    graph.transition(
        PipelineStage.SECOND_ROUND,
        inputs={
            "agent_count": len(agents),
            "round1_agents": len(state.first_round_reports),
        },
        outputs={"second_round_count": len(reports)},
    )

    return new_state, graph, reports


def compute_consensus(
    reports: list[AgentReport],
    high_dissent: bool = False,  # noqa: FBT001,FBT002
) -> ConsensusResult:
    """Berechnet Konsens aus Second-Round-Reports.

    Shadow-Agenten (status == SHADOW) werden komplett ignoriert
    (weight = 0.0).

    Bei high_dissent=True: wenn >60% Uneinigkeit, dann NO_TRADE.

    Args:
        reports: Liste von AgentReports (Second Round).
        high_dissent: True wenn Dissens-Prüfung aktiv ist.

    Returns:
        ConsensusResult mit Entscheidung und Gewichten.
    """
    from packages.consensus import WeightConfig, WeightedConsensusEngine

    # Filtere Shadow-Agenten heraus (weight = 0.0)
    active_reports = [
        r for r in reports
        if r.status != AgentStatus.SHADOW
    ]

    # Wenn keine aktiven Reports: NO_TRADE
    if not active_reports:
        return ConsensusResult(
            decision=ConsensusDecision.NO_TRADE,
            vote_distribution={
                    VoteDirection.LONG: 0.0, VoteDirection.SHORT: 0.0, VoteDirection.RANGE: 0.0, VoteDirection.ABSTAIN: 0.0,
                },
            agent_weights={},
            agent_agreements=[],
            agent_disagreements=[],
            confidence=0.0,
            reason="No active agents (all shadow)",
        )

    # WeightConfig mit shadow_weight = 0.0 (EPIC-08 Spezifikation)
    config = WeightConfig(
        status_multiplier={
            "active": 1.0,
            "shadow": 0.0,  # EPIC-08: Shadow agents weight 0.0
            "degraded": 0.3,
            "quarantined": 0.0,
            "disabled": 0.0,
        },
    )
    engine = WeightedConsensusEngine(config=config)
    result = engine.compute_consensus(active_reports)

    # Dissens-Prüfung: wenn >60% Uneinigkeit → NO_TRADE
    if high_dissent and result.confidence < 0.4:
        total_weight = sum(result.agent_weights.values())
        if total_weight > 0:
            disagreement_ratio = sum(
                w for a, w in result.agent_weights.items()
                if a in result.agent_disagreements
            ) / total_weight
            if disagreement_ratio > 0.6:
                return ConsensusResult(
                    decision=ConsensusDecision.NO_TRADE,
                    vote_distribution=result.vote_distribution,
                    agent_weights=result.agent_weights,
                    agent_agreements=result.agent_agreements,
                    agent_disagreements=result.agent_disagreements,
                    confidence=result.confidence,
                    reason="High dissent: >60% disagreement ratio",
                )

    return result
