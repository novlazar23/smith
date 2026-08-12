"""Kill Criteria — Automatisiert auswertbare Kriterien für Agenten-Deaktivierung.

Kill Criteria (automatische Degradation/Quarantäne):
    1. Ungültige Ausgaben (schema violation, NaN, Inf)
    2. Kalibrierungsverschlechterung (> threshold)
    3. Drift-Erkennung (> threshold)
    4. Fehlende Evidenz (< min_count)
    5. Gestörte Datenquelle (> gap_hours)
    6. Unverifizierte Verteilung
    7. Timeout (> max_age_days)

Alle Kriterien sind konfigurierbar und automatisch auswertbar.
Bei Trigger → Quarantäne oder Degradation (je nach Schwere).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from ..state_machine import AgentState


class KillSeverity(StrEnum):
    """Schweregrad eines Kill-Kriteriums.

    Reihenfolge = Priorität: CRITICAL > WARNING > INFO.
    Index wird für `min()` mit key=index verwendet (niedrigerer Index = höherer Schweregrad).
    """

    CRITICAL = "critical"   # index 0 → höchster Schweregrad
    WARNING = "warning"     # index 1
    INFO = "info"           # index 2


@dataclass
class KillCriteriaResult:
    """Ergebnis der Kill-Criteria-Auswertung."""

    agent_id: str
    triggered: bool = False
    criteria_met: list[str] = field(default_factory=list)
    severity: KillSeverity = KillSeverity.INFO
    details: list[dict[str, Any]] = field(default_factory=list)
    recommended_action: str = ""
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class KillCriteriaConfig:
    """Konfiguration der Kill Criteria."""

    # Kalibrierung
    calibration_regression_threshold: float = 0.10

    # Drift
    drift_threshold: float = 0.20

    # Evidenz
    min_evidence_count: int = 3

    # Datenquelle
    max_source_gap_hours: int = 24

    # Alter
    max_age_days: int = 30

    # Verteilung
    require_distribution_verification: bool = True

    # Schema/Output
    reject_invalid_output: bool = True


class KillCriteriaEngine:
    """Evaluiert alle Kill Criteria für einen Agenten.

    Parameters:
        config: Optional KillCriteriaConfig. Bei None wird Standard verwendet.
    """

    def __init__(self, config: KillCriteriaConfig | None = None) -> None:
        self.config = config or KillCriteriaConfig()

    def evaluate(
        self,
        agent_id: str,
        *,
        has_invalid_output: bool = False,
        calibration_score: float | None = None,
        prev_calibration_score: float | None = None,
        drift_score: float | None = None,
        evidence_count: int | None = None,
        source_connected: bool | None = None,
        source_last_seen: datetime | None = None,
        distribution_verified: bool | None = None,
        age_days: float | None = None,
        current_state: AgentState | None = None,
    ) -> KillCriteriaResult:
        """Evaluiert alle Kill Criteria und gibt das Ergebnis zurück.

        Returns:
            KillCriteriaResult mit allen getriggerten Kriterien und
            empfohlener Aktion.
        """
        result = KillCriteriaResult(agent_id=agent_id)
        triggered = False
        max_severity = KillSeverity.INFO

        # 1. Ungültige Ausgaben
        if self.config.reject_invalid_output and has_invalid_output:
            result.criteria_met.append("INVALID_OUTPUT")
            result.details.append({
                "criterion": "INVALID_OUTPUT",
                "severity": KillSeverity.CRITICAL,
                "detail": "Agent produced invalid output",
            })
            triggered = True
            max_severity = max(max_severity, KillSeverity.CRITICAL, key=lambda s: list(KillSeverity).index(s))

        # 2. Kalibrierungsverschlechterung
        if (
            calibration_score is not None
            and prev_calibration_score is not None
            and (prev_calibration_score - calibration_score) > self.config.calibration_regression_threshold
        ):
            result.criteria_met.append("CALIBRATION_REGRESSION")
            result.details.append({
                "criterion": "CALIBRATION_REGRESSION",
                "severity": KillSeverity.CRITICAL,
                "detail": f"Calibration dropped {prev_calibration_score:.2f} → {calibration_score:.2f}",
            })
            triggered = True
            max_severity = max(max_severity, KillSeverity.CRITICAL, key=lambda s: list(KillSeverity).index(s))

        # 3. Drift-Erkennung
        if drift_score is not None and drift_score > self.config.drift_threshold:
            result.criteria_met.append("DRIFT_DETECTED")
            result.details.append({
                "criterion": "DRIFT_DETECTED",
                "severity": KillSeverity.CRITICAL,
                "detail": f"Drift score {drift_score:.2f} > threshold {self.config.drift_threshold:.2f}",
            })
            triggered = True
            max_severity = max(max_severity, KillSeverity.CRITICAL, key=lambda s: list(KillSeverity).index(s))

        # 4. Fehlende Evidenz
        if evidence_count is not None and evidence_count < self.config.min_evidence_count:
            result.criteria_met.append("MISSING_EVIDENCE")
            result.details.append({
                "criterion": "MISSING_EVIDENCE",
                "severity": KillSeverity.WARNING,
                "detail": f"Evidence count {evidence_count} < {self.config.min_evidence_count}",
            })
            triggered = True
            max_severity = max(max_severity, KillSeverity.WARNING, key=lambda s: list(KillSeverity).index(s))

        # 5. Gestörte Quelle
        if source_connected is not None and not source_connected:
            result.criteria_met.append("DISRUPTED_SOURCE")
            result.details.append({
                "criterion": "DISRUPTED_SOURCE",
                "severity": KillSeverity.CRITICAL,
                "detail": "Source disconnected",
            })
            triggered = True
            max_severity = max(max_severity, KillSeverity.CRITICAL, key=lambda s: list(KillSeverity).index(s))
        elif source_last_seen is not None:
            gap = datetime.now(UTC) - source_last_seen
            if gap > timedelta(hours=self.config.max_source_gap_hours):
                result.criteria_met.append("DISRUPTED_SOURCE")
                result.details.append({
                    "criterion": "DISRUPTED_SOURCE",
                    "severity": KillSeverity.WARNING,
                    "detail": f"Source gap {gap.total_seconds()/3600:.1f}h > {self.config.max_source_gap_hours}h",
                })
                triggered = True
                max_severity = max(max_severity, KillSeverity.WARNING, key=lambda s: list(KillSeverity).index(s))

        # 6. Unverifizierte Verteilung
        if self.config.require_distribution_verification and distribution_verified is False:
            result.criteria_met.append("UNVERIFIED_DISTRIBUTION")
            result.details.append({
                "criterion": "UNVERIFIED_DISTRIBUTION",
                "severity": KillSeverity.WARNING,
                "detail": "Distribution not verified",
            })
            triggered = True
            max_severity = max(max_severity, KillSeverity.WARNING, key=lambda s: list(KillSeverity).index(s))

        # 7. Timeout
        if age_days is not None and age_days > self.config.max_age_days:
            result.criteria_met.append("TIMEOUT")
            result.details.append({
                "criterion": "TIMEOUT",
                "severity": KillSeverity.WARNING,
                "detail": f"Agent age {age_days:.0f}d > {self.config.max_age_days}d",
            })
            triggered = True
            max_severity = max(max_severity, KillSeverity.WARNING, key=lambda s: list(KillSeverity).index(s))

        result.triggered = triggered
        result.severity = max_severity

        # Empfohlene Aktion basierend auf Schweregrad
        if max_severity == KillSeverity.CRITICAL:
            result.recommended_action = "QUARANTINE"
        elif max_severity == KillSeverity.WARNING:
            result.recommended_action = "DEGRADE"
        else:
            result.recommended_action = "LOG_ONLY"

        return result

    def get_recommendation(
        self,
        agent_id: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[AgentState, str]:
        """Kurze Empfehlung: (Zielstatus, Begründung).

        Parameters:
            **kwargs: Werden an evaluate() weitergegeben.

        Returns:
            (target_state, reason) — z.B. (AgentState.QUARANTINED, "CRITICAL: DRIFT_DETECTED")
        """
        result = self.evaluate(agent_id, **kwargs)

        if not result.triggered:
            return AgentState.ACTIVE, "No criteria met"

        if result.severity == KillSeverity.CRITICAL:
            reasons = "; ".join(
                d["detail"] for d in result.details if d["severity"] == KillSeverity.CRITICAL
            )
            return AgentState.QUARANTINED, f"CRITICAL: {reasons}"

        reasons = "; ".join(
            d["detail"] for d in result.details if d["severity"] == KillSeverity.WARNING
        )
        return AgentState.DEGRADED, f"WARNING: {reasons}"
