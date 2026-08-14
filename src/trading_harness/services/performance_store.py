from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from trading_harness.models import PerformanceRecord
from trading_harness.services.db import Database


def _record_to_row(record: PerformanceRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "agent_id": record.agent_id,
        "snapshot_id": record.snapshot_id,
        "direction": record.direction,
        "confidence": record.confidence,
        "outcome": record.outcome,
        "realized_pnl": record.realized_pnl,
        "mfe": record.mfe,
        "mae": record.mae,
        "timestamp": record.timestamp.isoformat(),
    }


def _row_to_record(row: dict[str, Any]) -> PerformanceRecord:
    return PerformanceRecord(
        id=row["id"],
        run_id=row["run_id"],
        agent_id=row["agent_id"],
        snapshot_id=row["snapshot_id"],
        direction=row["direction"],
        confidence=row["confidence"],
        outcome=row.get("outcome"),
        realized_pnl=row.get("realized_pnl", 0.0),
        mfe=row.get("mfe", 0.0),
        mae=row.get("mae", 0.0),
        timestamp=datetime.fromisoformat(str(row["timestamp"])).replace(tzinfo=UTC),
    )


class PersistedPerformanceStore:
    """PostgreSQL-backed store for performance records.

    Falls back to in-memory store when PostgreSQL is unavailable.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._fallback: dict[str, PerformanceRecord] = {}

    def add(self, record: PerformanceRecord) -> PerformanceRecord:
        if self._db and self._db.is_available:
            row = _record_to_row(record)
            cols = list(row.keys())
            placeholders = ",".join(["%s"] * len(cols))
            self._db.execute_write(
                f"INSERT INTO performance_records ({','.join(cols)}) "
                f"VALUES ({placeholders}) "
                "ON CONFLICT (id) DO UPDATE SET "
                + ",".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id"),
                *[row[c] for c in cols],
            )
        else:
            self._fallback[record.id] = record
        return record

    def get(self, record_id: str) -> PerformanceRecord | None:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM performance_records WHERE id = %s", (record_id,)
            )
            if rows:
                return _row_to_record(rows[0])
            return None
        return self._fallback.get(record_id)

    def by_run(self, run_id: str) -> list[PerformanceRecord]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM performance_records WHERE run_id = %s ORDER BY timestamp",
                (run_id,),
            )
            return [_row_to_record(r) for r in rows]
        return [r for r in self._fallback.values() if r.run_id == run_id]

    def by_agent(self, agent_id: str) -> list[PerformanceRecord]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM performance_records WHERE agent_id = %s ORDER BY timestamp",
                (agent_id,),
            )
            return [_row_to_record(r) for r in rows]
        return [r for r in self._fallback.values() if r.agent_id == agent_id]

    def all(self) -> list[PerformanceRecord]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM performance_records ORDER BY timestamp"
            )
            return [_row_to_record(r) for r in rows]
        return list(self._fallback.values())

    def by_snapshot(self, snapshot_id: str) -> list[PerformanceRecord]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM performance_records WHERE snapshot_id = %s ORDER BY timestamp",
                (snapshot_id,),
            )
            return [_row_to_record(r) for r in rows]
        return [r for r in self._fallback.values() if r.snapshot_id == snapshot_id]