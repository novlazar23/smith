"""Quant Evidence Aggregator (Phase 9).

Kombiniert alle Quant-Evidence zu einem einheitlichen Dict für den
Shadow Trading Loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar


@dataclass
class EvidenceEntry:
    """Einzelner Evidence-Eintrag."""
    source: str  # z.B. "features", "anomalies", "regime"
    timestamp: str
    data: dict[str, Any]
    confidence: float = 1.0
    priority: int = 0  # höher = wichtiger


@dataclass
class AggregatedEvidence:
    """Aggregierte Evidence für einen Tick."""
    symbol: str
    timeframe: str
    timestamp: str
    entries: dict[str, EvidenceEntry]
    summary: dict[str, Any]
    total_confidence: float
    high_priority_count: int


class EvidenceAggregator:
    """Aggregiert Quant-Evidence für den Shadow Trading Loop."""

    # Priorität pro Quelle
    SOURCE_PRIORITIES: ClassVar[dict[str, int]] = {
        "anomalies": 10,
        "regime": 8,
        "forward_outcomes": 7,
        "ml_features": 6,
        "similarity": 5,
        "features": 4,
        "backtest": 3,
    }

    def __init__(self) -> None:
        self._entries: dict[str, EvidenceEntry] = {}

    def add_entry(
        self,
        source: str,
        data: dict[str, Any],
        timestamp: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Fügt einen Evidence-Eintrag hinzu."""
        ts = timestamp or datetime.now(UTC).isoformat()
        priority = self.SOURCE_PRIORITIES.get(source, 0)
        self._entries[source] = EvidenceEntry(
            source=source, timestamp=ts, data=data,
            confidence=confidence, priority=priority,
        )

    def aggregate(
        self,
        symbol: str,
        timeframe: str,
    ) -> AggregatedEvidence:
        """Aggregiert alle Evidence-Einträge."""
        now = datetime.now(UTC).isoformat()
        if not self._entries:
            return AggregatedEvidence(
                symbol=symbol, timeframe=timeframe, timestamp=now,
                entries={}, summary={}, total_confidence=0.0,
                high_priority_count=0,
            )

        summary = self._build_summary()
        total_conf = sum(e.confidence for e in self._entries.values()) / len(self._entries)
        high_pri = sum(1 for e in self._entries.values() if e.priority >= 7)

        return AggregatedEvidence(
            symbol=symbol, timeframe=timeframe, timestamp=now,
            entries=dict(self._entries), summary=summary,
            total_confidence=total_conf, high_priority_count=high_pri,
        )

    def _build_summary(self) -> dict[str, Any]:
        """Baut eine Zusammenfassung aller Evidence."""
        summary: dict[str, Any] = {
            "sources": list(self._entries.keys()),
            "total_entries": len(self._entries),
        }
        # Extrahiere wichtigste Informationen
        if "regime" in self._entries:
            summary["current_regime"] = self._entries["regime"].data.get("regime", "unknown")
        if "anomalies" in self._entries:
            anomaly_data = self._entries["anomalies"].data
            summary["anomaly_count"] = anomaly_data.get("count", 0)
        if "features" in self._entries:
            feat_data = self._entries["features"].data
            summary["feature_count"] = len(feat_data)
        return summary

    def clear(self) -> None:
        """Löscht alle Einträge."""
        self._entries.clear()

    def get_entry(self, source: str) -> EvidenceEntry | None:
        """Gibt einen Eintrag zurück."""
        return self._entries.get(source)

    def has_source(self, source: str) -> bool:
        """Prüft ob eine Quelle vorhanden ist."""
        return source in self._entries

    @property
    def sources(self) -> list[str]:
        return list(self._entries.keys())

    @property
    def entry_count(self) -> int:
        return len(self._entries)
