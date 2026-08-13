"""Orchestrator-Pipeline — chaining von first_round → seal → second_round → consensus.

Die OrchestratorPipeline ist die Haupt-Schnittstelle fuer die
Konsens-Pipeline. Sie kombiniert alle Phasen in einer nahtlosen Kette.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from packages.consensus import ConsensusDecision, ConsensusResult
from packages.orchestrator.graph import (
    OrchestratorGraph,
    PipelineStage,
    TradingGraphState,
    create_initial_state,
)
from packages.orchestrator.seal import seal_first_round
from packages.orchestrator.second_round import (
    build_round_summary,
    compute_consensus,
    run_second_round,
)
from packages.schemas.agent_report import AgentReport


@dataclass
class OrchestratorPipelineResult:
    """Ergebnis der Orchestrator-Pipeline.

    decision: Finale Handelsentscheidung
    consensus: Konsens-Ergebnis
    first_round_reports: Reports aus Round 1
    seal_records: Hash-Siegel
    second_round_reports: Reports aus Round 2
    errors: Liste von Fehlern
    """

    decision: str
    consensus: ConsensusResult | None
    first_round_reports: list[AgentReport]
    seal_records: list[dict]
    second_round_reports: list[AgentReport]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_no_trade(self) -> bool:
        """True wenn die Entscheidung NO_TRADE ist."""
        return self.decision == ConsensusDecision.NO_TRADE

    @property
    def direction(self) -> str:
        """Handelsrichtung der Entscheidung."""
        return self.decision


class OrchestratorPipeline:
    """Chained Orchestrator-Pipeline: first_round -> seal -> second_round -> consensus.

    Example:
        pipeline = OrchestratorPipeline()
        result = pipeline.run(
            run_id="run-001",
            instrument="BTC/USD",
            agents=agents,
            market_data=market_data,
        )
        print(result.decision)
    """

    def __init__(self, high_dissent_threshold: float = 0.6) -> None:
        """Initialisiert die Pipeline.

        Args:
            high_dissent_threshold: Schwellwert fuer Dissens (Default 0.6 = 60%).
                                    Wenn >60% der gewichteten Agenten disagree,
                                    dann NO_TRADE.
        """
        self.high_dissent_threshold = high_dissent_threshold

    def run(
        self,
        run_id: str,
        instrument: str,
        agents: list[Any],
        market_data: dict[str, Any],
    ) -> OrchestratorPipelineResult:
        """Fuehrt die komplette Pipeline aus.

        Sequenz:
          1. FIRST_ROUND — parallele Agenten-Ausfuehrung ohne Peer-Reports
          2. SEAL — SHA-256 Hash-Siegel pro Report
          3. SECOND_ROUND — Agenten mit Round-1-Zusammenfassung
          4. CONSENSUS — gewichteter Konsens

        Args:
            run_id: Eindeutige Run-ID.
            instrument: Handelsinstrument (z.B. "BTC/USD").
            agents: Liste von Agenten mit ``analyze`` und
                    ``analyze_with_context`` Methoden.
            market_data: Marktdaten fuer die Analyse.

        Returns:
            OrchestratorPipelineResult mit allen Ergebnissen.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Step 0: Create initial state
        state = create_initial_state(run_id, instrument, market_data)
        graph = OrchestratorGraph()
        graph.transition(
            PipelineStage.REQUEST,
            inputs={"instrument": instrument},
        )

        # Step 1: First Round
        try:
            state, graph = self._run_first_round(
                state, graph, agents, market_data
            )
        except Exception as exc:
            errors.append(f"First round failed: {exc}")
            return OrchestratorPipelineResult(
                decision=ConsensusDecision.NO_TRADE.value,
                consensus=None,
                first_round_reports=[],
                seal_records=[],
                second_round_reports=[],
                errors=errors,
                warnings=warnings,
            )

        # Step 2: Seal
        try:
            state, graph, seal_records = seal_first_round(state, graph)
        except Exception as exc:
            errors.append(f"Seal failed: {exc}")
            return OrchestratorPipelineResult(
                decision=ConsensusDecision.NO_TRADE.value,
                consensus=None,
                first_round_reports=state.first_round_reports,
                seal_records=[],
                second_round_reports=[],
                errors=errors,
                warnings=warnings,
            )

        # Step 3: Second Round
        try:
            state, graph, second_round_reports = self._run_second_round(
                state, graph, agents, market_data
            )
        except Exception as exc:
            errors.append(f"Second round failed: {exc}")
            return OrchestratorPipelineResult(
                decision=ConsensusDecision.NO_TRADE.value,
                consensus=None,
                first_round_reports=state.first_round_reports,
seal_records=[r.__dict__ for r in seal_records],
                second_round_reports=[],
                errors=errors,
                warnings=warnings,
            )

        # Step 4: Consensus
        try:
            consensus = self._run_consensus(
                second_round_reports, state
            )
            decision = consensus.decision.value
        except Exception as exc:
            errors.append(f"Consensus failed: {exc}")
            decision = ConsensusDecision.NO_TRADE.value
            consensus = None

        # Transition to decision
        graph.transition(
            PipelineStage.CONSENSUS,
            inputs={
                "decision": decision,
                "confidence": consensus.confidence if consensus else 0.0,
            },
        )

        return OrchestratorPipelineResult(
            decision=decision,
            consensus=consensus,
            first_round_reports=state.first_round_reports,
            seal_records=[r.__dict__ for r in seal_records],
            second_round_reports=second_round_reports,
            errors=errors,
            warnings=warnings,
        )

    # ── Internal helpers ──────────────────────────────────────────

    def _run_first_round(
        self,
        state: TradingGraphState,
        graph: OrchestratorGraph,
        agents: list[Any],
        market_data: dict[str, Any],
    ) -> tuple[TradingGraphState, OrchestratorGraph]:
        """Internal: First-Round-Ausfuehrung."""
        from packages.orchestrator.graph import run_first_round

        return run_first_round(state, graph, agents, market_data)

    def _run_second_round(
        self,
        state: TradingGraphState,
        graph: OrchestratorGraph,
        agents: list[Any],
        market_data: dict[str, Any],
    ) -> tuple[TradingGraphState, OrchestratorGraph, list[AgentReport]]:
        """Internal: Second-Round-Ausfuehrung mit Summary."""
        # Build summary from first round
        summary = build_round_summary(state.first_round_reports)

        return run_second_round(
            state,
            graph,
            agents,
            market_data,
            contrarian_review={
                "direction": summary.direction,
                "confidence": summary.confidence,
            },
        )

    def _run_consensus(
        self,
        reports: list[AgentReport],
        state: TradingGraphState,
    ) -> ConsensusResult:
        """Internal: Consensus mit Dissens-Prüfung."""
        summary = state.round_summary
        high_dissent = summary.get("dissent_score", 0) > self.high_dissent_threshold

        return compute_consensus(reports, high_dissent=high_dissent)
