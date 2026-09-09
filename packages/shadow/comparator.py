"""Shadow Decision Comparison vs Paper Execution.

Tracks shadow decisions alongside paper execution results and
computes quality metrics:
- Agreement rate (shadow vs paper)
- Divergence tracking (when decisions differ)
- Latency comparison (should be identical)
- Decision-level audit trail

Shadow mode never executes — this module provides the
comparison layer for quality measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from packages.consensus import ConsensusDecision
from packages.schemas.agent_report import AgentReport


@dataclass(frozen=True, slots=True)
class PaperDecision:
    """Eine Paper-Execution-Entscheidung.

    Paper trading: virtual orders, no real capital, but same
    execution logic as live trading.
    """

    run_id: str
    timestamp: datetime
    direction: str  # "LONG_BIAS", "SHORT_BIAS", "RANGE", "NO_TRADE"
    confidence: float
    instrument: str
    agent_reports: list[AgentReport] = field(default_factory=list)
    consensus_reason: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def is_no_trade(self) -> bool:
        """True wenn die Paper-Entscheidung NO_TRADE ist."""
        return self.direction == ConsensusDecision.NO_TRADE.value

    @property
    def is_trade(self) -> bool:
        """True wenn eine Paper-Order ausgeführt würde."""
        return not self.is_no_trade


@dataclass(frozen=True, slots=True)
class ShadowDecisionRecord:
    """Shadow decision with metadata for comparison.

    Stores all data from a shadow-mode pipeline run, ready for
    later comparison with paper or live trading.
    """

    run_id: str
    instrument: str
    timestamp: datetime
    direction: str
    confidence: float
    consensus: dict | None
    agent_reports: list[AgentReport]
    latency_ms: float
    brier_score: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_no_trade(self) -> bool:
        """True wenn die Shadow-Entscheidung NO_TRADE ist."""
        return self.direction == ConsensusDecision.NO_TRADE.value

    @property
    def is_trade(self) -> bool:
        """True wenn eine Order getroffen wurde."""
        return not self.is_no_trade


@dataclass
class ShadowPaperComparison:
    """Vergleicht Shadow-Entscheidungen mit Paper-Execution.

    Tracks agreement rate, divergence, and quality metrics.

    Example:
        comparator = ShadowPaperComparison()
        comparator.record_shadow(shadow_decision)
        comparator.record_paper(paper_decision)
        result = comparator.compare(run_id="run-001")
        print(result.agreement)
    """

    _shadow_records: dict[str, ShadowDecisionRecord] = field(default_factory=dict)
    _paper_records: dict[str, PaperDecision] = field(default_factory=dict)
    _comparison_log: list[ComparisonResult] = field(default_factory=list)

    def record_shadow(
        self,
        run_id: str,
        instrument: str,
        timestamp: datetime,
        direction: str,
        confidence: float,
        consensus: dict | None,
        agent_reports: list[AgentReport],
        latency_ms: float,
        brier_score: float | None = None,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """Recordet eine Shadow-Entscheidung.

        Args:
            run_id: Eindeutige Run-ID.
            instrument: Handelsinstrument.
            timestamp: Zeitstempel.
            direction: Entscheidung Richtung.
            confidence: Konsens-Konfidenz.
            consensus: Konsens-Details.
            agent_reports: Agent Reports.
            latency_ms: Pipeline-Latenz in ms.
            brier_score: Optional Brier Score.
            errors: Optionale Fehlerliste.
            warnings: Optionale Warnliste.
        """
        record = ShadowDecisionRecord(
            run_id=run_id,
            instrument=instrument,
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            consensus=consensus,
            agent_reports=agent_reports,
            latency_ms=latency_ms,
            brier_score=brier_score,
            errors=errors or [],
            warnings=warnings or [],
        )
        self._shadow_records[run_id] = record

    def record_paper(
        self,
        run_id: str,
        timestamp: datetime,
        direction: str,
        confidence: float,
        instrument: str,
        agent_reports: list[AgentReport] | None = None,
        consensus_reason: str = "",
        errors: list[str] | None = None,
    ) -> None:
        """Recordet eine Paper-Execution-Entscheidung.

        Args:
            run_id: Eindeutige Run-ID.
            timestamp: Zeitstempel.
            direction: Entscheidung Richtung.
            confidence: Konfidenz.
            instrument: Handelsinstrument.
            agent_reports: Optionale Agent Reports.
            consensus_reason: Begründung des Konsens.
            errors: Optionale Fehlerliste.
        """
        record = PaperDecision(
            run_id=run_id,
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            instrument=instrument,
            agent_reports=agent_reports or [],
            consensus_reason=consensus_reason,
            errors=errors or [],
        )
        self._paper_records[run_id] = record

    def compare(self, run_id: str) -> ComparisonResult:
        """Vergleicht Shadow und Paper für eine spezifische Run-ID.

        Args:
            run_id: Run-ID zum Vergleichen.

        Returns:
            ComparisonResult mit allen Metriken.

        Raises:
            KeyError: Wenn keine Shadow oder Paper-Entscheidung gefunden.
        """
        shadow = self._shadow_records.get(run_id)
        paper = self._paper_records.get(run_id)

        if shadow is None:
            raise KeyError(f"Shadow decision not found for run_id: {run_id}")
        if paper is None:
            raise KeyError(f"Paper decision not found for run_id: {run_id}")

        # Direction agreement
        direction_match = shadow.direction == paper.direction
        trade_match = shadow.is_trade == paper.is_trade

        # Latency comparison
        latency_ms = shadow.latency_ms

        # Confidence difference
        confidence_diff = abs(shadow.confidence - paper.confidence)

        # Detailed comparison
        details: dict[str, object] = {
            "shadow_direction": shadow.direction,
            "paper_direction": paper.direction,
            "shadow_confidence": shadow.confidence,
            "paper_confidence": paper.confidence,
            "latency_ms": latency_ms,
        }

        if shadow.consensus:
            details["shadow_consensus"] = shadow.consensus
        if paper.consensus_reason:
            details["paper_reason"] = paper.consensus_reason

        result = ComparisonResult(
            run_id=run_id,
            timestamp=datetime.now(UTC),
            direction_match=direction_match,
            trade_match=trade_match,
            overall_agreement=direction_match and trade_match,
            confidence_diff=round(confidence_diff, 6),
            latency_ms=latency_ms,
            brier_score=shadow.brier_score,
            details=details,
        )

        self._comparison_log.append(result)
        return result

    @property
    def agreement_rate(self) -> float:
        """Gesamt-Übereinstimmungsrate Shadow vs Paper.

        Fraction of comparisons where shadow and paper agree
        on both direction AND trade/no-trade classification.
        """
        if not self._comparison_log:
            return 0.0
        matching = sum(
            1 for c in self._comparison_log if c.overall_agreement
        )
        return matching / len(self._comparison_log)

    @property
    def direction_agreement_rate(self) -> float:
        """Nur Richtungs-Übereinstimmungsrate."""
        if not self._comparison_log:
            return 0.0
        matching = sum(
            1 for c in self._comparison_log if c.direction_match
        )
        return matching / len(self._comparison_log)

    @property
    def comparison_count(self) -> int:
        """Anzahl durchgeführter Vergleiche."""
        return len(self._comparison_log)

    @property
    def divergence_count(self) -> int:
        """Anzahl der Divergenzen (unterschiedliche Entscheidungen)."""
        return sum(
            1 for c in self._comparison_log if not c.overall_agreement
        )

    def get_divergences(self) -> list[ComparisonResult]:
        """Gibt alle Divergenzen zurück."""
        return [c for c in self._comparison_log if not c.overall_agreement]

    @property
    def avg_confidence_diff(self) -> float:
        """Durchschnittlicher Konfidenz-Unterschied."""
        if not self._comparison_log:
            return 0.0
        return sum(c.confidence_diff for c in self._comparison_log) / len(
            self._comparison_log
        )

    @property
    def avg_latency_ms(self) -> float:
        """Durchschnittliche Latenz (sollte identisch zu Production sein)."""
        if not self._comparison_log:
            return 0.0
        return sum(c.latency_ms for c in self._comparison_log) / len(
            self._comparison_log
        )

    @property
    def summary(self) -> dict[str, object]:
        """Kompakte Zusammenfassung aller Vergleiche."""
        return {
            "total_comparisons": self.comparison_count,
            "overall_agreement_rate": round(self.agreement_rate, 4),
            "direction_agreement_rate": round(self.direction_agreement_rate, 4),
            "divergence_count": self.divergence_count,
            "avg_confidence_diff": round(self.avg_confidence_diff, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "shadow_runs": len(self._shadow_records),
            "paper_runs": len(self._paper_records),
        }


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Ergebnis eines einzelnen Shadow-Paper Vergleichs.

    Attributes:
        run_id: Eindeutige Run-ID.
        timestamp: Zeitpunkt des Vergleichs.
        direction_match: True wenn Shadow und Paper gleiche Richtung.
        trade_match: True wenn beide entweder traden oder nicht.
        overall_agreement: True wenn direction AND trade match.
        confidence_diff: Absoluter Konfidenz-Unterschied.
        latency_ms: Pipeline-Latenz des Shadow-Runs.
        brier_score: Optionaler Brier Score für diese Entscheidung.
        details: Detaillierte Vergleichsdaten.
    """

    run_id: str
    timestamp: datetime
    direction_match: bool
    trade_match: bool
    overall_agreement: bool
    confidence_diff: float
    latency_ms: float
    brier_score: float | None = None
    details: dict[str, object] = field(default_factory=dict)
