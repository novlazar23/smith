"""Seal-Modul — SHA-256 Hash-Siegel fuer First-Round-Reports.

Jeder Report wird mit einem SHA-256 Hash gesiegelt.
Der Hash ist unveraenderbar — einmal geschrieben, kann er nicht
manipuliert werden.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from packages.orchestrator.graph import OrchestratorGraph, PipelineStage, TradingGraphState
from packages.schemas.agent_report import AgentReport


@dataclass(frozen=True)
class SealRecord:
    """Ein Siegelter Datensatz pro First-Round-Report.

    data_hash: SHA-256 Hash des Report-JSON (unveraenderbar)
    report_id: ID des gesiegelten Reports
    timestamp: Zeitpunkt des Siegelns
    """

    data_hash: str
    report_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def timestamp_iso(self) -> str:
        """ISO-Format des Zeitstempels."""
        return self.timestamp.isoformat()


def _hash_report(report: AgentReport) -> str:
    """Erzeugt SHA-256 Hash aus Report-JSON.

    Args:
        report: Zu siegelnder AgentReport.

    Returns:
        SHA-256 Hex-Digest des sortierten Report-JSONs.
    """
    data = json.dumps(
        report.model_dump(), sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_seal(report: AgentReport, expected_hash: str) -> bool:
    """Verifiziert einen zuvor erzeugten Seal-Hash.

    Args:
        report: Der Report, der verifiziert werden soll.
        expected_hash: Der erwartete SHA-256 Hash.

    Returns:
        True wenn der Hash uebereinstimmt.
    """
    computed = _hash_report(report)
    import hmac
    return hmac.compare_digest(computed, expected_hash)


def seal_first_round(
    state: TradingGraphState,
    graph: OrchestratorGraph,
) -> tuple[TradingGraphState, OrchestratorGraph, list[SealRecord]]:
    """Siegelt alle First-Round-Reports mit SHA-256.

    Jeder Report wird einzeln gehasht und in einer immutable
    SealRecord-Liste gespeichert. Der Hash ist unveraenderbar.

    Args:
        state: Aktueller Graphzustand mit populated first_round_reports.
        graph: OrchestratorGraph fuer Stage-Management.

    Returns:
        (TradingGraphState, OrchestratorGraph, list[SealRecord])
        mit populated seal_records und Stage SEAL.

    Raises:
        ValueError: Wenn first_round_reports leer ist.
    """
    reports = state.first_round_reports
    if not reports:
        raise ValueError(
            "seal_first_round: first_round_reports must not be empty"
        )

    # Hash jeden Report — unveraenderbar
    seal_records: list[SealRecord] = []
    for report in reports:
        data_hash = _hash_report(report)
        seal_records.append(SealRecord(
            data_hash=data_hash,
            report_id=report.report_id,
        ))

    # State aktualisieren — seal records sind immutable
    seal_data = [
        {
            "data_hash": r.data_hash,
            "report_id": r.report_id,
            "timestamp_iso": r.timestamp_iso,
        }
        for r in seal_records
    ]

    new_state = state.__class__(
        run_id=state.run_id,
        instrument=state.instrument,
        first_round_reports=state.first_round_reports,
        first_round_hash=seal_records[0].data_hash if seal_records else "",
        seal_records=seal_data,
        current_stage=PipelineStage.SEAL.value,
        errors=[],
        warnings=[],
    )

    graph.transition(
        PipelineStage.SEAL,
        inputs={"report_count": len(seal_records)},
        outputs={"first_round_hash": seal_records[0].data_hash[:16] if seal_records else ""},
    )

    return new_state, graph, seal_records
