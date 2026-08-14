from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from trading_harness.models import EvaluationResult
from trading_harness.services.db import Database


def _result_to_row(result: EvaluationResult) -> dict[str, Any]:
    return {
        "id": result.id,
        "run_id": result.run_id,
        "agent_id": result.agent_id,
        "metric_name": result.metric_name,
        "metric_value": result.metric_value,
        "observations": result.observations,
        "details": result.details,
        "timestamp": result.timestamp.isoformat(),
    }


def _row_to_result(row: dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(
        id=row["id"],
        run_id=row["run_id"],
        agent_id=row["agent_id"],
        metric_name=row["metric_name"],
        metric_value=row["metric_value"],
        observations=row["observations"],
        details=row.get("details", {}),
        timestamp=datetime.fromisoformat(str(row["timestamp"])).replace(tzinfo=UTC),
    )


class PersistedEvaluationResultStore:
    """PostgreSQL-backed store for evaluation results.

    Falls back to in-memory store when PostgreSQL is unavailable.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._fallback: dict[str, EvaluationResult] = {}

    def add(self, result: EvaluationResult) -> EvaluationResult:
        if self._db and self._db.is_available:
            row = _result_to_row(result)
            cols = list(row.keys())
            placeholders = ",".join(["%s"] * len(cols))
            self._db.execute_write(
                f"INSERT INTO evaluation_results ({','.join(cols)}) "
                f"VALUES ({placeholders}) "
                "ON CONFLICT (id) DO UPDATE SET "
                + ",".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id"),
                *[row[c] for c in cols],
            )
        else:
            self._fallback[result.id] = result
        return result

    def get(self, result_id: str) -> EvaluationResult | None:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM evaluation_results WHERE id = %s", (result_id,)
            )
            if rows:
                return _row_to_result(rows[0])
            return None
        return self._fallback.get(result_id)

    def by_agent(self, agent_id: str) -> list[EvaluationResult]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM evaluation_results WHERE agent_id = %s ORDER BY timestamp",
                (agent_id,),
            )
            return [_row_to_result(r) for r in rows]
        return [r for r in self._fallback.values() if r.agent_id == agent_id]

    def by_run(self, run_id: str) -> list[EvaluationResult]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM evaluation_results WHERE run_id = %s ORDER BY timestamp",
                (run_id,),
            )
            return [_row_to_result(r) for r in rows]
        return [r for r in self._fallback.values() if r.run_id == run_id]

    def all(self) -> list[EvaluationResult]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM evaluation_results ORDER BY timestamp"
            )
            return [_row_to_result(r) for r in rows]
        return list(self._fallback.values())