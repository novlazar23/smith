"""Quarantine — Automatische Degradation bei Drift/Fehlern.

Quarantäne → Gewicht 0.0 im Consensus.
Auslöser: Ungültige Ausgaben, Kalibrierungsverschlechterung, Drift,
fehlende Evidenz, gestörte Quelle, unvert. Verteilung, Timeout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum


class QuarantineReason(StrEnum):
    """Gründe für die Quarantäne eines Agents."""

    INVALID_OUTPUT = "INVALID_OUTPUT"
    CALIBRATION_REGRESSION = "CALIBRATION_REGRESSION"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    DISRUPTED_SOURCE = "DISRUPTED_SOURCE"
    UNVERIFIED_DISTRIBUTION = "UNVERIFIED_DISTRIBUTION"
    TIMEOUT = "TIMEOUT"


@dataclass
class QuarantineEvent:
    """Einzelnes Quarantäne-Ereignis."""

    agent_id: str
    reason: QuarantineReason
    severity: str  # "warning", "critical"
    details: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved: bool = False
    resolved_at: datetime | None = None


@dataclass
class QuarantineConfig:
    """Konfiguration für automatische Quarantäne."""

    calibration_regression_threshold: float = 0.10
    drift_threshold: float = 0.20
    min_evidence_count: int = 3
    max_source_gap_hours: int = 24
    max_age_days: int = 30
    distribution_confidence: float = 0.90


class QuarantineEngine:
    """Prüft, ob ein Agent in Quarantäne gehen muss."""

    def __init__(self, config: QuarantineConfig | None = None) -> None:
        self.config = config or QuarantineConfig()
        self.events: list[QuarantineEvent] = []

    def check(
        self,
        agent_id: str,
        calibration_score: float | None = None,
        prev_calibration_score: float | None = None,
        drift_score: float | None = None,
        evidence_count: int | None = None,
        source_connected: bool | None = None,
        source_last_seen: datetime | None = None,
        distribution_verified: bool | None = None,
        age_days: float | None = None,
    ) -> tuple[bool, list[QuarantineEvent]]:
        """Evaluiert alle Quarantäne-Kriterien.

        Returns:
            (quarantine_needed, events)
        """
        events: list[QuarantineEvent] = []

        # CALIBRATION_REGRESSION
        if (
            calibration_score is not None
            and prev_calibration_score is not None
            and prev_calibration_score - calibration_score > self.config.calibration_regression_threshold
        ):
            events.append(
                QuarantineEvent(
                    agent_id=agent_id,
                    reason=QuarantineReason.CALIBRATION_REGRESSION,
                    severity="critical",
                    details=f"Calibration dropped {prev_calibration_score:.2f} → {calibration_score:.2f}",
                )
            )

        # DRIFT_DETECTED
        if drift_score is not None and drift_score > self.config.drift_threshold:
            events.append(
                QuarantineEvent(
                    agent_id=agent_id,
                    reason=QuarantineReason.DRIFT_DETECTED,
                    severity="critical",
                    details=f"Drift score {drift_score:.2f} > threshold {self.config.drift_threshold:.2f}",
                )
            )

        # MISSING_EVIDENCE
        if evidence_count is not None and evidence_count < self.config.min_evidence_count:
            events.append(
                QuarantineEvent(
                    agent_id=agent_id,
                    reason=QuarantineReason.MISSING_EVIDENCE,
                    severity="warning",
                    details=f"Evidence count {evidence_count} < {self.config.min_evidence_count}",
                )
            )

        # DISRUPTED_SOURCE
        if source_connected is not None and not source_connected:
            events.append(
                QuarantineEvent(
                    agent_id=agent_id,
                    reason=QuarantineReason.DISRUPTED_SOURCE,
                    severity="critical",
                    details="Source disconnected",
                )
            )
        elif source_last_seen is not None:
            gap = datetime.now(UTC) - source_last_seen
            if gap > timedelta(hours=self.config.max_source_gap_hours):
                events.append(
                    QuarantineEvent(
                        agent_id=agent_id,
                        reason=QuarantineReason.DISRUPTED_SOURCE,
                        severity="warning",
                        details=f"Source gap {gap.total_seconds()/3600:.1f}h > {self.config.max_source_gap_hours}h",
                    )
                )

        # UNVERIFIED_DISTRIBUTION
        if distribution_verified is False:
            events.append(
                QuarantineEvent(
                    agent_id=agent_id,
                    reason=QuarantineReason.UNVERIFIED_DISTRIBUTION,
                    severity="warning",
                    details="Distribution not verified",
                )
            )

        # TIMEOUT
        if age_days is not None and age_days > self.config.max_age_days:
            events.append(
                QuarantineEvent(
                    agent_id=agent_id,
                    reason=QuarantineReason.TIMEOUT,
                    severity="warning",
                    details=f"Agent age {age_days:.0f}d > {self.config.max_age_days}d",
                )
            )

        quarantine_needed = any(e.severity == "critical" for e in events)
        return quarantine_needed, events

    @property
    def active_quarantine_count(self) -> int:
        return sum(1 for e in self.events if not e.resolved)
