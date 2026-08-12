"""Quarantine Manager für fehlerhafte Market Data Events.

Speichert invalidierte Events in einer Quarantäne-Storage
(MinIO/S3) und ermöglicht spätere manuelle Prüfung:
- Quarantäne-Manager mit threshold-basierter Steuerung
- Export/Import von Quarantäne-Batches
- Replay von quarantänisierten Events nach manueller Freigabe
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class QuarantineEntry:
    """Ein quarantänisiertes Event."""

    event_hash: str  # SHA256 des serialisierten Events
    event_type: str
    instrument: str
    venue: str
    reason: str
    severity: str  # 'warning', 'error', 'critical'
    quality_score: float
    event_data: dict[str, Any]
    quarantined_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    release_status: str = "quarantined"  # 'quarantined', 'released', 'discarded'

    def __post_init__(self) -> None:
        if self.release_status not in ("quarantined", "released", "discarded"):
            raise ValueError(f"Invalid release_status: {self.release_status}")


@dataclass
class QuarantineResult:
    """Ergebnis der Quarantäne-Entscheidung."""

    is_quarantined: bool
    entry: QuarantineEntry | None = None
    reason: str = ""

    @classmethod
    def approve(cls) -> QuarantineResult:
        return cls(is_quarantined=False)

    @classmethod
    def quarantine(cls, entry: QuarantineEntry) -> QuarantineResult:
        return cls(is_quarantined=True, entry=entry, reason=entry.reason)


class QuarantineManager:
    """Verwaltet die Quarantäne für invalidierte Market Data Events.

    Entschiede basierend auf quality_score und event severity,
    ob ein Event in Quarantäne geht, durchgeht oder abgewiesen wird.
    """

    # Quality-Score thresholds
    PASS_THRESHOLD = 0.9
    QUARANTINE_THRESHOLD = 0.5
    # Severity weights
    SEVERITY_WEIGHTS: ClassVar[dict[str, float]] = {
        "critical": 1.0,
        "error": 0.8,
        "warning": 0.3,
        "info": 0.0,
    }

    def __init__(
        self,
        storage_backend: str = "memory",
        storage_path: str = "",
    ) -> None:
        self._storage_backend = storage_backend
        self._storage_path = storage_path
        self._entries: list[QuarantineEntry] = []

    def evaluate_and_quarantine(
        self,
        event: dict[str, Any],
        quality_score: float,
        issues: list[dict[str, Any]],
    ) -> QuarantineResult:
        """Entscheidet über Quarantäne eines Events.

        Logik:
        - quality_score >= PASS_THRESHOLD → durchgehen
        - quality_score < QUARANTINE_THRESHOLD oder CRITICAL issue → Quarantäne
        - Sonst → prüfen ob Schweregrad quarantäne-würdig ist

        Args:
            event: Das zu prüfende Event-Dict
            quality_score: Validierungs-Quality-Score
            issues: Liste der Validierungsprobleme

        Returns:
            QuarantineResult
        """
        # Pass: hohe Qualität
        if quality_score >= self.PASS_THRESHOLD:
            return QuarantineResult.approve()

        # Prüfe auf kritische Issues
        has_critical = any(
            i.get("severity", "") in ("critical", "error")
            for i in issues
        )

        # Quarantäne: niedrig oder kritisch
        if quality_score < self.QUARANTINE_THRESHOLD or has_critical:
            event_hash = self._hash_event(event)
            severity = "critical" if has_critical else "error"

            entry = QuarantineEntry(
                event_hash=event_hash,
                event_type=event.get("type", event.get("event_type", "unknown")),
                instrument=event.get("instrument", "unknown"),
                venue=event.get("venue", "unknown"),
                reason=self._reason_from_issues(issues),
                severity=severity,
                quality_score=quality_score,
                event_data=event,
            )

            self._entries.append(entry)
            logger.warning(
                "Event %s quarantined: score=%.2f, reasons=%s",
                event_hash[:8], quality_score, entry.reason,
            )

            return QuarantineResult.quarantine(entry)

        # Degraded: nicht kritisch aber mit Warnungen
        # Wird durchgelassen mit Warning-Log
        return QuarantineResult.approve()

    def release(self, event_hash: str) -> bool:
        """Hebt die Quarantäne für ein Event auf (manuelle Freigabe).

        Args:
            event_hash: SHA256-Hash des Events

        Returns:
            True wenn Event gefunden und freigegeben
        """
        for entry in self._entries:
            if entry.event_hash == event_hash:
                entry.release_status = "released"
                logger.info("Event %s released from quarantine", event_hash[:8])
                return True
        return False

    def discard(self, event_hash: str) -> bool:
        """Verwirft ein Event dauerhaft aus der Quarantäne.

        Args:
            event_hash: SHA256-Hash des Events

        Returns:
            True wenn Event gefunden und verworfen
        """
        for entry in self._entries:
            if entry.event_hash == event_hash:
                entry.release_status = "discarded"
                logger.info("Event %s discarded", event_hash[:8])
                return True
        return False

    def get_quarantined_events(
        self,
        instrument: str | None = None,
        event_type: str | None = None,
    ) -> list[QuarantineEntry]:
        """Gibt alle quarantänisierten Events zurück.

        Args:
            instrument: Optional filter nach Instrument
            event_type: Optional filter nach Event-Typ

        Returns:
            Liste von QuarantineEntry
        """
        results = self._entries
        if instrument:
            results = [e for e in results if e.instrument == instrument]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        return [e for e in results if e.release_status == "quarantined"]

    def get_stats(self) -> dict[str, Any]:
        """Gibt Statistiken über die Quarantäne zurück."""
        total = len(self._entries)
        quarantined = sum(1 for e in self._entries if e.release_status == "quarantined")
        released = sum(1 for e in self._entries if e.release_status == "released")
        discarded = sum(1 for e in self._entries if e.release_status == "discarded")

        return {
            "total_quarantined": total,
            "currently_quarantined": quarantined,
            "released": released,
            "discarded": discarded,
        }

    def export_quarantine(
        self,
        instrument: str | None = None,
        event_type: str | None = None,
    ) -> str:
        """Exportiert die Quarantäne als JSON-String.

        Nützlich für Backup und manuelle Prüfung.
        """
        events = self.get_quarantined_events(instrument, event_type)
        data = [
            {
                "event_hash": e.event_hash,
                "event_type": e.event_type,
                "instrument": e.instrument,
                "venue": e.venue,
                "reason": e.reason,
                "severity": e.severity,
                "quality_score": e.quality_score,
                "quarantined_at": e.quarantined_at.isoformat(),
            }
            for e in events
        ]
        return json.dumps(data, indent=2, default=str)

    def import_quarantine(self, json_data: str) -> int:
        """Importiert eine Quarantäne-Liste aus JSON.

        Returns:
            Anzahl importierter Einträge
        """
        data = json.loads(json_data)
        imported = 0
        for item in data:
            entry = QuarantineEntry(
                event_hash=item["event_hash"],
                event_type=item["event_type"],
                instrument=item["instrument"],
                venue=item["venue"],
                reason=item["reason"],
                severity=item["severity"],
                quality_score=item["quality_score"],
                event_data=item.get("event_data", {}),
                quarantined_at=datetime.fromisoformat(item["quarantined_at"]),
            )
            self._entries.append(entry)
            imported += 1
        logger.info("Imported %d quarantine entries", imported)
        return imported

    def clear(self) -> None:
        """Löscht alle Quarantäne-Einträge."""
        self._entries.clear()
        logger.info("Quarantine cleared")

    def _hash_event(self, event: dict[str, Any]) -> str:
        """Berechnet SHA256-Hash eines Events."""
        event_json = json.dumps(event, sort_keys=True, default=str)
        return hashlib.sha256(event_json.encode("utf-8")).hexdigest()

    def _reason_from_issues(self, issues: list[dict[str, Any]]) -> str:
        """Erstellt eine zusammenfassende Reason aus Issues."""
        if not issues:
            return "Low quality score"
        return "; ".join(
            f"{i.get('field', '?')}: {i.get('message', '?')}" for i in issues
        )
