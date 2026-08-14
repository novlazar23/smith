from __future__ import annotations

from threading import RLock

from trading_harness.models import PerformanceRecord


class PerformanceStore:
    """Thread-safe in-memory store for performance records."""

    def __init__(self) -> None:
        self._records: dict[str, PerformanceRecord] = {}
        self._lock = RLock()

    def add(self, record: PerformanceRecord) -> PerformanceRecord:
        with self._lock:
            self._records[record.id] = record
            return record

    def get(self, record_id: str) -> PerformanceRecord | None:
        with self._lock:
            return self._records.get(record_id)

    def by_run(self, run_id: str) -> list[PerformanceRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.run_id == run_id]

    def by_agent(self, agent_id: str) -> list[PerformanceRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.agent_id == agent_id]

    def all(self) -> list[PerformanceRecord]:
        with self._lock:
            return list(self._records.values())

    def by_snapshot(self, snapshot_id: str) -> list[PerformanceRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.snapshot_id == snapshot_id]