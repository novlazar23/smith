from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from trading_harness.models import AgentGenome, AgentStatus
from trading_harness.services.agent_registry import AgentRegistry
from trading_harness.services.db import Database

_SCHEMA_VERSION = "1"


def _genome_to_row(agent: AgentGenome) -> dict[str, Any]:
    return {
        "id": agent.id,
        "generation": agent.generation,
        "parent_agents": agent.parent_agents,
        "category": agent.category,
        "status": agent.status,
        "prompt_version": agent.prompt_version,
        "reasoning_style": agent.reasoning_style,
        "indicators": agent.indicators,
        "timeframes": agent.timeframes,
        "feature_preferences": agent.feature_preferences,
        "statistical_methods": agent.statistical_methods,
        "weighting_strategy": agent.weighting_strategy,
        "confidence_calibration": agent.confidence_calibration,
        "risk_attitude": agent.risk_attitude,
        "context_window_strategy": agent.context_window_strategy,
        "output_schema": agent.output_schema,
        "model_profile": agent.model_profile,
        "temperature": agent.temperature,
        "created_at": agent.created_at.isoformat(),
    }


def _row_to_genome(row: dict[str, Any]) -> AgentGenome:
    return AgentGenome(
        id=row["id"],
        generation=row["generation"],
        parent_agents=row["parent_agents"],
        category=row["category"],
        status=AgentStatus(row["status"]),
        prompt_version=row["prompt_version"],
        reasoning_style=row["reasoning_style"],
        indicators=row["indicators"],
        timeframes=row["timeframes"],
        feature_preferences=row["feature_preferences"],
        statistical_methods=row["statistical_methods"],
        weighting_strategy=row["weighting_strategy"],
        confidence_calibration=row["confidence_calibration"],
        risk_attitude=row["risk_attitude"],
        context_window_strategy=row["context_window_strategy"],
        output_schema=row["output_schema"],
        model_profile=row["model_profile"],
        temperature=row["temperature"],
        created_at=_parse_ts(row["created_at"]),
    )


def _parse_ts(value: str | Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


class PersistedAgentRegistry:
    """PostgreSQL-backed agent registry.

    Falls back to in-memory AgentRegistry when PostgreSQL is unavailable.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._fallback = AgentRegistry()

    def list(self) -> list[AgentGenome]:
        if self._db and self._db.is_available:
            rows = self._db.execute("SELECT * FROM agents ORDER BY id")
            return [_row_to_genome(r) for r in rows]
        return self._fallback.list()

    def get(self, agent_id: str) -> AgentGenome | None:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM agents WHERE id = %s", (agent_id,)
            )
            if rows:
                return _row_to_genome(rows[0])
            return None
        return self._fallback.get(agent_id)

    def add(self, agent: AgentGenome) -> AgentGenome:
        if self._db and self._db.is_available:
            row = _genome_to_row(agent)
            cols = list(row.keys())
            placeholders = ",".join(["%s"] * len(cols))
            self._db.execute_write(
                f"INSERT INTO agents ({','.join(cols)}) VALUES ({placeholders}) "
                "ON CONFLICT (id) DO UPDATE SET "
                + ",".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id"),
                *[row[c] for c in cols],
            )
        else:
            self._fallback.add(agent)
        return agent

    def get_version(self, agent_id: str) -> str:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT COUNT(*) AS c FROM agents WHERE id = %s", (agent_id,)
            )
            if rows:
                return str(rows[0]["c"])
            return "0"
        return "1" if self._fallback.get(agent_id) is not None else "0"