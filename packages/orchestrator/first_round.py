"""First Round — parallele Agenten-Ausfuehrung OHNE Peer-Reports.

Jeder Agent bekommt Rohdaten (market_data + features) — NICHT die
Reports anderer Agenten. Jede Antwort wird als AgentReport zurueckgegeben.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from packages.orchestrator.graph import OrchestratorGraph, PipelineStage, TradingGraphState

if TYPE_CHECKING:
    from packages.schemas.agent_report import AgentReport

__all__ = [
    "run_first_round",
]


def run_first_round(
    state: TradingGraphState,
    graph: OrchestratorGraph,
    agents: list[Any],
    market_data: dict[str, Any],
) -> tuple[TradingGraphState, OrchestratorGraph]:
    """Fuehrt Round 1 aus: parallele Agenten-Ausfuehrung OHNE Peer-Reports.

    Args:
        state: Aktueller Graphzustand.
        graph: OrchestratorGraph fuer Stage-Management.
        agents: Liste von Agenten mit ``analyze`` Methode.
        market_data: Rohdaten (candles, orderbook, features, ...).

    Returns:
        (TradingGraphState, OrchestratorGraph) mit populated first_round_reports.

    Raises:
        ValueError: Wenn keine Agenten verfuegbar sind.
    """
    if not agents:
        raise ValueError("run_first_round: at least one agent is required")

    reports: list[AgentReport] = []
    for agent in agents:
        report = agent.analyze(market_data)
        reports.append(report)

    new_state = state.__class__(
        run_id=state.run_id,
        instrument=state.instrument,
        market_snapshot=state.market_snapshot,
        first_round_reports=reports,
        current_stage=PipelineStage.FIRST_ROUND.value,
        errors=[],
        warnings=[],
    )

    graph.transition(
        PipelineStage.FIRST_ROUND,
        inputs={"agent_count": len(agents)},
        outputs={"report_count": len(reports)},
    )

    return new_state, graph
