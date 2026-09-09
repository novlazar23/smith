"""Shadow Mode Decision Engine.

Same orchestrator stack as production — same agents, same consensus,
same strategy. The only difference: decisions are NOT executed, only
recorded for later comparison with paper or live trading outcomes.

Audit trail for all shadow decisions via ``AuditTrail.log_decision()``.
Latency measured identically to production (no acceleration).

Public API:
    - ShadowEngine — runs full pipeline, records, never executes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from packages.consensus import ConsensusDecision, ConsensusResult
from packages.governance.audit import AuditTrail
from packages.orchestrator import (
    OrchestratorPipeline,
    OrchestratorPipelineResult,
)

if TYPE_CHECKING:
    from packages.schemas.agent_report import AgentReport


@dataclass(frozen=True, slots=True)
class ShadowDecision:
    """Ein in Shadow-Mode getroffene Entscheidung.

    Enthält alle Daten der Pipeline-Execution sowie
    Metriken (Brier Score, Latency, etc.).
    """

    run_id: str
    instrument: str
    timestamp: datetime
    decision: str
    consensus: ConsensusResult | None
    first_round_reports: list[AgentReport]
    second_round_reports: list[AgentReport]
    brier_score: float | None = None
    latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_no_trade(self) -> bool:
        """True wenn die Entscheidung NO_TRADE ist."""
        return self.decision == ConsensusDecision.NO_TRADE.value


class ShadowEngine:
    """Shadow-Mode Decision Engine.

    Führt die volle Orchestrator-Pipeline aus (identisch zu Production)
    und schreibt alle Entscheidungen in den Audit Trail — aber ohne
    Order-Ausführung oder Kapitalrisiko.

    Example:
        engine = ShadowEngine(
            instrument="BTC/USD",
            audit_trail=audit_trail,
        )
        decision = engine.run(
            run_id="shadow-001",
            agents=agents,
            market_data=market_data,
        )
        # decision is recorded, never executed
    """

    def __init__(
        self,
        instrument: str,
        audit_trail: AuditTrail | None = None,
        high_dissent_threshold: float = 0.6,
    ) -> None:
        """Initialisiert die Shadow Engine.

        Args:
            instrument: Handelsinstrument (z.B. "BTC/USD").
            audit_trail: Optionales Audit-Trail-Objekt. Wird intern
                         erzeugt, wenn nicht angegeben.
            high_dissent_threshold: Schwellwert für Dissens-Prüfung.
        """
        self.instrument = instrument
        self._audit_trail = audit_trail or AuditTrail()
        self._pipeline = OrchestratorPipeline(
            high_dissent_threshold=high_dissent_threshold,
        )

    @property
    def audit_trail(self) -> AuditTrail:
        """Das Audit-Trail-Objekt für Shadow-Entscheidungen."""
        return self._audit_trail

    def run(
        self,
        run_id: str,
        agents: list[Any],
        market_data: dict[str, Any],
    ) -> ShadowDecision:
        """Führt die volle Orchestrator-Pipeline im Shadow-Modus aus.

        Gleiche Sequenz wie Production:
          1. FIRST_ROUND — parallele Agenten-Ausführung
          2. SEAL — SHA-256 Hash-Siegel
          3. SECOND_ROUND — Agenten mit Round-1-Zusammenfassung
          4. CONSENSUS — gewichteter Konsens

        Keine Order-Ausführung — Entscheidung wird nur protokolliert.

        Args:
            run_id: Eindeutige Run-ID.
            agents: Liste von Agenten mit ``analyze`` und
                    ``analyze_with_context`` Methoden.
            market_data: Marktdaten für die Analyse.

        Returns:
            ShadowDecision mit allen Pipeline-Daten und Metriken.
        """
        start = datetime.now(UTC)

        # Pipeline ausführen (identisch zu Production)
        result = self._pipeline.run(
            run_id=run_id,
            instrument=self.instrument,
            agents=agents,
            market_data=market_data,
        )

        end = datetime.now(UTC)
        latency_ms = (end - start).total_seconds() * 1000

        # Entscheidung protokollieren
        self._audit_decision(run_id, result)

        return ShadowDecision(
            run_id=run_id,
            instrument=self.instrument,
            timestamp=end,
            decision=result.decision,
            consensus=result.consensus,
            first_round_reports=result.first_round_reports,
            second_round_reports=result.second_round_reports,
            latency_ms=round(latency_ms, 2),
            errors=result.errors,
            warnings=result.warnings,
        )

    def run_single_round(
        self,
        run_id: str,
        agents: list[Any],
        market_data: dict[str, Any],
    ) -> ShadowDecision:
        """Shadow-Modus für schnelle Iteration und Experimente.

        # ponytail: Aktuell Alias für ``run()`` — OrchestratorPipeline kennt
        noch keinen Single-Round-Modus (FIRST_ROUND + CONSENSUS ohne
        SEAL/SECOND_ROUND). Upgrade path: Pipeline-Flag oder eigene Stage.

        Args:
            run_id: Eindeutige Run-ID.
            agents: Liste von Agenten.
            market_data: Marktdaten.

        Returns:
            ShadowDecision mit Pipeline-Ergebnis.
        """
        start = datetime.now(UTC)

        result = self._pipeline.run(
            run_id=run_id,
            instrument=self.instrument,
            agents=agents,
            market_data=market_data,
        )

        end = datetime.now(UTC)
        latency_ms = (end - start).total_seconds() * 1000

        self._audit_decision(run_id, result)

        return ShadowDecision(
            run_id=run_id,
            instrument=self.instrument,
            timestamp=end,
            decision=result.decision,
            consensus=result.consensus,
            first_round_reports=result.first_round_reports,
            second_round_reports=result.second_round_reports,
            latency_ms=round(latency_ms, 2),
            errors=result.errors,
            warnings=result.warnings,
        )

    def _audit_decision(
        self,
        run_id: str,
        result: OrchestratorPipelineResult,
    ) -> None:
        """Schreibt die Entscheidung in den Audit Trail.

        Nutzt den vorhandenen AuditTrail.log_decision()-Mechanismus.

        Args:
            run_id: Eindeutige Run-ID des Shadow-Runs.
            result: Pipeline-Ergebnis mit Entscheidungsdaten.
        """
        details: dict[str, Any] = {
            "decision": result.decision,
            "run_id": run_id,
            "consensus": (
                {
                    "decision": result.consensus.decision.value
                    if result.consensus
                    else None,
                    "confidence": result.consensus.confidence
                    if result.consensus
                    else 0.0,
                    "reason": result.consensus.reason
                    if result.consensus
                    else "",
                }
                if result.consensus
                else None
            ),
            "first_round_count": len(result.first_round_reports),
            "second_round_count": len(result.second_round_reports),
            "errors": result.errors,
            "warnings": result.warnings,
        }

        self._audit_trail.log_decision(
            agent_id="shadow_engine",
            decision=result.decision,
            actor="shadow",
            details=details,
        )

    @property
    def decision_count(self) -> int:
        """Anzahl der protokollierten Shadow-Entscheidungen."""
        return self._audit_trail.total_entries
