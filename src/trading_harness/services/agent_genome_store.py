from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from trading_harness.models import AgentGenome
from trading_harness.services.db import Database

logger = logging.getLogger(__name__)


def _genome_hash(agent: AgentGenome) -> str:
    payload = json.dumps(
        {k: v for k, v in agent.model_dump().items() if k not in ("id", "created_at")},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class AgentGenomeStore:
    """In-memory agent genome store for evolution."""

    def __init__(self) -> None:
        self._store: dict[str, AgentGenome] = {}

    def get(self, agent_id: str) -> AgentGenome | None:
        return self._store.get(agent_id)

    def list_all(self) -> list[AgentGenome]:
        return list(self._store.values())

    def list_by_category(self, category: str) -> list[AgentGenome]:
        return [a for a in self._store.values() if a.category == category]

    def list_by_status(self, status: str) -> list[AgentGenome]:
        return [a for a in self._store.values() if a.status == status]

    def list_active(self, category: str) -> list[AgentGenome]:
        return [
            a for a in self._store.values()
            if a.category == category and a.status in ("ACTIVE", "CHAMPION")
        ]

    def list_challengers(self, category: str) -> list[AgentGenome]:
        return [
            a for a in self._store.values()
            if a.category == category and a.status == "CHALLENGER"
        ]

    def list_by_generation(self, generation: int) -> list[AgentGenome]:
        return [a for a in self._store.values() if a.generation == generation]

    def add(self, agent: AgentGenome) -> AgentGenome:
        self._store[agent.id] = agent
        return agent

    def update(self, agent: AgentGenome) -> AgentGenome:
        self._store[agent.id] = agent
        return agent

    def get_or_create(self, agent_id: str) -> AgentGenome:
        agent = self._store.get(agent_id)
        if agent is None:
            agent = AgentGenome(id=agent_id, category="generic")
            self._store[agent_id] = agent
        return agent


class PersistedAgentGenomeStore(AgentGenomeStore):
    """PostgreSQL-backed agent genome store with in-memory fallback."""

    def __init__(self, db: Database | None = None) -> None:
        super().__init__()
        self._db = db

    def _ensure_schema(self) -> None:
        if self._db and self._db.is_available:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS agent_genomes (
                    id TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL DEFAULT 0,
                    parent_agents TEXT[] NOT NULL DEFAULT '{}',
                    category TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'GENERATED',
                    prompt_version TEXT NOT NULL DEFAULT '1',
                    reasoning_style TEXT NOT NULL DEFAULT 'systematic',
                    indicators TEXT[] NOT NULL DEFAULT '{}',
                    timeframes TEXT[] NOT NULL DEFAULT '{}',
                    feature_preferences TEXT[] NOT NULL DEFAULT '{}',
                    statistical_methods TEXT[] NOT NULL DEFAULT '{}',
                    weighting_strategy TEXT NOT NULL DEFAULT 'default',
                    confidence_calibration TEXT NOT NULL DEFAULT 'default',
                    risk_attitude TEXT NOT NULL DEFAULT 'conservative',
                    context_window_strategy TEXT NOT NULL DEFAULT 'bounded',
                    output_schema TEXT NOT NULL DEFAULT 'signal-v1',
                    model_profile TEXT NOT NULL DEFAULT 'local-main',
                    temperature REAL NOT NULL DEFAULT 0.2,
                    content_hash TEXT,
                    created_at TIMESTAMPTZ NOT NULL
                )"""
            )

    def add(self, agent: AgentGenome) -> AgentGenome:
        self._ensure_schema()
        if self._db and self._db.is_available:
            row = {
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
                "content_hash": _genome_hash(agent),
                "created_at": agent.created_at.isoformat(),
            }
            cols = list(row.keys())
            placeholders = ",".join(["%s"] * len(cols))
            self._db.execute_write(
                f"INSERT INTO agent_genomes ({','.join(cols)}) VALUES ({placeholders}) "
                "ON CONFLICT (id) DO UPDATE SET "
                + ",".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id"),
                *[row[c] for c in cols],
            )
        super().add(agent)
        return agent

    def get(self, agent_id: str) -> AgentGenome | None:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM agent_genomes WHERE id = %s", (agent_id,)
            )
            if rows:
                r = rows[0]
                return AgentGenome(
                    id=r["id"],
                    generation=r["generation"],
                    parent_agents=r["parent_agents"],
                    category=r["category"],
                    status=r["status"],
                    prompt_version=r["prompt_version"],
                    reasoning_style=r["reasoning_style"],
                    indicators=r["indicators"],
                    timeframes=r["timeframes"],
                    feature_preferences=r["feature_preferences"],
                    statistical_methods=r["statistical_methods"],
                    weighting_strategy=r["weighting_strategy"],
                    confidence_calibration=r["confidence_calibration"],
                    risk_attitude=r["risk_attitude"],
                    context_window_strategy=r["context_window_strategy"],
                    output_schema=r["output_schema"],
                    model_profile=r["model_profile"],
                    temperature=r["temperature"],
                    created_at=_parse_ts(r["created_at"]),
                )
            return None
        return super().get(agent_id)

    def list_all(self) -> list[AgentGenome]:
        if self._db and self._db.is_available:
            rows = self._db.execute("SELECT * FROM agent_genomes ORDER BY id")
            if rows:
                result = []
                for r in rows:
                    a = AgentGenome(
                        id=r["id"],
                        generation=r["generation"],
                        parent_agents=r["parent_agents"],
                        category=r["category"],
                        status=r["status"],
                        prompt_version=r["prompt_version"],
                        reasoning_style=r["reasoning_style"],
                        indicators=r["indicators"],
                        timeframes=r["timeframes"],
                        feature_preferences=r["feature_preferences"],
                        statistical_methods=r["statistical_methods"],
                        weighting_strategy=r["weighting_strategy"],
                        confidence_calibration=r["confidence_calibration"],
                        risk_attitude=r["risk_attitude"],
                        context_window_strategy=r["context_window_strategy"],
                        output_schema=r["output_schema"],
                        model_profile=r["model_profile"],
                        temperature=r["temperature"],
                        created_at=_parse_ts(r["created_at"]),
                    )
                    result.append(a)
                    self._store[a.id] = a
                return result
        return super().list_all()

    def list_by_category(self, category: str) -> list[AgentGenome]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM agent_genomes WHERE category = %s ORDER BY id", (category,)
            )
            if rows:
                result = []
                for r in rows:
                    a = AgentGenome(
                        id=r["id"],
                        generation=r["generation"],
                        parent_agents=r["parent_agents"],
                        category=r["category"],
                        status=r["status"],
                        prompt_version=r["prompt_version"],
                        reasoning_style=r["reasoning_style"],
                        indicators=r["indicators"],
                        timeframes=r["timeframes"],
                        feature_preferences=r["feature_preferences"],
                        statistical_methods=r["statistical_methods"],
                        weighting_strategy=r["weighting_strategy"],
                        confidence_calibration=r["confidence_calibration"],
                        risk_attitude=r["risk_attitude"],
                        context_window_strategy=r["context_window_strategy"],
                        output_schema=r["output_schema"],
                        model_profile=r["model_profile"],
                        temperature=r["temperature"],
                        created_at=_parse_ts(r["created_at"]),
                    )
                    result.append(a)
                return result
        return super().list_by_category(category)


def _parse_ts(value: str | Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)