"""In-Memory-Observability für das Quant-Modul (Quant-Plattform, P1-9).

``QuantMetrics`` aggregiert Prozess-interne Counter für geschriebene und
downsamplte Candles, Query-Latenz und Fehler. Bewusst ohne externes
Metrics-System (kein Prometheus/StatsD) — nur interne Sichtbarkeit für
die Shadow-Trading-Integration.

``QuantHealthChecker`` kombiniert den Verbindungszustand des
``InfluxDBStore`` mit den Metrik-Summen zu einem einzigen Snapshot für
Health-Endpoints und Logs.

Konventionen aus ``influxdb_client.py``: RLock für Zustandszugriffe,
Lese-Pfade werfen keine Exceptions — ein ausgefallener Store meldet sich
einfach als disconnected.
"""

from __future__ import annotations

import threading
from typing import Any

from trading_harness.quant.influxdb_client import InfluxDBStore


class QuantMetrics:
    """Thread-sichere In-Memory-Counter für die Quant-Daten-Pipeline."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._candles_written: dict[str, int] = {}
        self._candles_downsampled: dict[tuple[str, str], int] = {}
        self._queries: dict[str, dict[str, float]] = {}
        self._errors: dict[str, int] = {}
        self._errors_by_component: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Aufzeichnung
    # ------------------------------------------------------------------

    def record_candles_written(self, measurement: str, count: int) -> None:
        """Addiert geschriebene Candles zum Counter des Measurements."""
        if count < 0:
            raise ValueError("count must be >= 0")
        with self._lock:
            self._candles_written[measurement] = self._candles_written.get(measurement, 0) + count

    def record_candles_downsampled(self, source_tf: str, target_tf: str, count: int) -> None:
        """Addiert downsamplte Candles zum Counter des (Quell-, Ziel-)Timeframe-Paars."""
        if count < 0:
            raise ValueError("count must be >= 0")
        with self._lock:
            key = (source_tf, target_tf)
            self._candles_downsampled[key] = self._candles_downsampled.get(key, 0) + count

    def record_query(self, measurement: str, duration_ms: float) -> None:
        """Zählt eine Query und erfasst deren Dauer in Millisekunden."""
        if duration_ms < 0:
            raise ValueError("duration_ms must be >= 0")
        with self._lock:
            entry = self._queries.setdefault(
                measurement, {"count": 0.0, "total_ms": 0.0, "max_ms": 0.0}
            )
            entry["count"] += 1.0
            entry["total_ms"] += duration_ms
            entry["max_ms"] = max(entry["max_ms"], duration_ms)

    def record_error(self, error_type: str, component: str) -> None:
        """Zählt einen Fehler nach Typ und Komponente."""
        with self._lock:
            self._errors[error_type] = self._errors.get(error_type, 0) + 1
            self._errors_by_component[component] = self._errors_by_component.get(component, 0) + 1

    # ------------------------------------------------------------------
    # Lese-Zugriffe
    # ------------------------------------------------------------------

    @property
    def total_writes(self) -> int:
        """Gesamtanzahl geschriebener Candles über alle Measurements."""
        with self._lock:
            return sum(self._candles_written.values())

    @property
    def total_queries(self) -> int:
        """Gesamtanzahl aufzeichneter Queries."""
        with self._lock:
            return int(sum(e["count"] for e in self._queries.values()))

    @property
    def total_errors(self) -> int:
        """Gesamtanzahl aufzeichneter Fehler."""
        with self._lock:
            return sum(self._errors.values())

    def get_summary(self) -> dict[str, Any]:
        """Gibt einen konsistenten Snapshot aller Counter zurück."""
        with self._lock:
            total_writes = sum(self._candles_written.values())
            total_downsampled = sum(self._candles_downsampled.values())
            total_queries = int(sum(e["count"] for e in self._queries.values()))
            total_ms = sum(e["total_ms"] for e in self._queries.values())
            max_ms = max((e["max_ms"] for e in self._queries.values()), default=0.0)
            total_errors = sum(self._errors.values())
            return {
                "total_writes": total_writes,
                "candles_written": dict(self._candles_written),
                "total_downsampled": total_downsampled,
                "candles_downsampled": {
                    f"{source}->{target}": count
                    for (source, target), count in self._candles_downsampled.items()
                },
                "total_queries": total_queries,
                "queries": {m: int(e["count"]) for m, e in self._queries.items()},
                "query_duration_total_ms": total_ms,
                "query_duration_avg_ms": (total_ms / total_queries) if total_queries else 0.0,
                "query_duration_max_ms": max_ms,
                "total_errors": total_errors,
                "errors": dict(self._errors),
                "errors_by_component": dict(self._errors_by_component),
            }


class QuantHealthChecker:
    """Kombiniert InfluxDBStore-Zustand und QuantMetrics zu einem Health-Snapshot.

    ``url`` und ``enabled`` werden typischerweise aus den Settings
    (``influxdb_url`` / ``influxdb_enabled``) injiziert; ohne ``url`` wird
    auf die Store-Config zurückgegriffen.
    """

    def __init__(
        self,
        store: InfluxDBStore,
        *,
        metrics: QuantMetrics | None = None,
        url: str = "",
        enabled: bool = False,
    ) -> None:
        self._store = store
        self._metrics = metrics if metrics is not None else QuantMetrics()
        self._url = url
        self._enabled = enabled

    @property
    def metrics(self) -> QuantMetrics:
        """Die diesem Checker zugeordneten Metriken (eigene Instanz, falls keine injiziert)."""
        return self._metrics

    def check_all(self) -> dict[str, Any]:
        """Liefert den aktuellen Health-Snapshot (wirft nie)."""
        return {
            "influxdb": {
                "connected": self._store.is_available,
                "url": self._url or getattr(self._store, "_url", ""),
                "buffer_size": self._store.buffer_size(),
            },
            "metrics": {
                "total_writes": self._metrics.total_writes,
                "total_queries": self._metrics.total_queries,
                "total_errors": self._metrics.total_errors,
            },
            "enabled": self._enabled,
        }
