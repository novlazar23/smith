from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from trading_harness.models import AgentAnalysisResult
from trading_harness.services.db import Database


def _result_to_row(result: AgentAnalysisResult) -> dict[str, Any]:
    signal = result.signal
    return {
        "id": signal.id,
        "run_id": signal.run_id,
        "agent_id": signal.agent_id,
        "snapshot_id": signal.snapshot_id,
        "category": signal.category,
        "direction": signal.direction,
        "confidence": signal.confidence,
        "reasoning": signal.reasoning,
        "signals": signal.signals,
        "risks": signal.risks,
        "prompt_version": result.prompt_version,
        "model_profile": result.model_profile,
        "raw_response": result.raw_response,
        "timestamp": signal.timestamp.isoformat(),
    }


def _row_to_result(row: dict[str, Any]) -> AgentAnalysisResult:
    from trading_harness.models import AgentSignal

    return AgentAnalysisResult(
        run_id=row["run_id"],
        agent_id=row["agent_id"],
        signal=AgentSignal(
            id=row["id"],
            run_id=row["run_id"],
            agent_id=row["agent_id"],
            snapshot_id=row["snapshot_id"],
            category=row["category"],
            direction=row["direction"],
            confidence=row["confidence"],
            reasoning=row.get("reasoning", ""),
            signals=row.get("signals", []),
            risks=row.get("risks", []),
            timestamp=datetime.fromisoformat(str(row["timestamp"])).replace(tzinfo=UTC),
        ),
        prompt_version=row["prompt_version"],
        model_profile=row["model_profile"],
        raw_response=row.get("raw_response", {}),
    )


class PersistedAgentAnalysisStore:
    """PostgreSQL-backed store for agent analysis results.

    Falls back to in-memory storage when PostgreSQL is unavailable.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._fallback: dict[str, AgentAnalysisResult] = {}

    def add(self, result: AgentAnalysisResult) -> AgentAnalysisResult:
        if self._db and self._db.is_available:
            row = _result_to_row(result)
            cols = list(row.keys())
            placeholders = ",".join(["%s"] * len(cols))
            self._db.execute_write(
                f"INSERT INTO agent_analysis_results ({','.join(cols)}) "
                f"VALUES ({placeholders}) "
                "ON CONFLICT (id) DO UPDATE SET "
                + ",".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id"),
                *[row[c] for c in cols],
            )
        else:
            self._fallback[result.signal.id] = result
        return result

    def get(self, signal_id: str) -> AgentAnalysisResult | None:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM agent_analysis_results WHERE id = %s", (signal_id,)
            )
            if rows:
                return _row_to_result(rows[0])
            return None
        return self._fallback.get(signal_id)

    def by_run(self, run_id: str) -> list[AgentAnalysisResult]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM agent_analysis_results WHERE run_id = %s ORDER BY timestamp",
                (run_id,),
            )
            return [_row_to_result(r) for r in rows]
        return [r for r in self._fallback.values() if r.run_id == run_id]

    def by_agent(self, agent_id: str) -> list[AgentAnalysisResult]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM agent_analysis_results WHERE agent_id = %s ORDER BY timestamp",
                (agent_id,),
            )
            return [_row_to_result(r) for r in rows]
        return [r for r in self._fallback.values() if r.agent_id == agent_id]

    def by_snapshot(self, snapshot_id: str) -> list[AgentAnalysisResult]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM agent_analysis_results WHERE snapshot_id = %s ORDER BY timestamp",
                (snapshot_id,),
            )
            return [_row_to_result(r) for r in rows]
        return [r for r in self._fallback.values() if r.signal.snapshot_id == snapshot_id]

    def all(self) -> list[AgentAnalysisResult]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM agent_analysis_results ORDER BY timestamp"
            )
            return [_row_to_result(r) for r in rows]
        return list(self._fallback.values())