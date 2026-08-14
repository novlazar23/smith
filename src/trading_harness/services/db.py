from __future__ import annotations

import logging
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from psycopg_pool import PoolTimeout as PsycopgPoolTimeout

logger = logging.getLogger(__name__)

INIT_SQL = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL DEFAULT 0,
    parent_agents TEXT[],
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
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    data JSONB NOT NULL,
    content_hash TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    previous_state TEXT,
    new_state TEXT,
    details JSONB NOT NULL DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS trading_runs (
    id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'CREATED',
    decisions JSONB NOT NULL DEFAULT '[]',
    outcome TEXT,
    outcome_reason TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
    id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction_predicted TEXT NOT NULL,
    direction_actual TEXT NOT NULL,
    confidence_predicted REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    mfe REAL NOT NULL DEFAULT 0,
    mae REAL NOT NULL DEFAULT 0,
    holding_period_bars INTEGER NOT NULL DEFAULT 0,
    realized_pnl REAL NOT NULL DEFAULT 0,
    regime TEXT NOT NULL DEFAULT 'unknown',
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    observations INTEGER NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL
);
"""


class Database:
    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._pool: ConnectionPool | None = None
        self._ready = False

    @property
    def is_available(self) -> bool:
        return self._pool is not None and self._ready

    def _ensure_pool(self) -> None:
        """Lazily initialize the connection pool on first use."""
        if self._ready:
            return
        try:
            self._pool = ConnectionPool(
                self._url,
                open=True,
                timeout=1,
            )
            with self._pool.connection() as conn:
                conn.execute(INIT_SQL)
            self._ready = True
            logger.info("PostgreSQL connection pool initialized and schema verified")
        except (
            OSError,
            RuntimeError,
            ValueError,
            PsycopgPoolTimeout,
        ):
            self._pool = None
            self._ready = False
            logger.warning("PostgreSQL not available, stores will use in-memory fallback")

    @property
    def is_connected(self) -> bool:
        return self._ready

    def execute(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self._ensure_pool()
        if not self._pool or not self._ready:
            return []
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, args)
            if cur.description:
                return cur.fetchall()
            return []

    def execute_write(self, sql: str, *args: Any) -> None:
        self._ensure_pool()
        if not self._pool or not self._ready:
            return
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, args)
            conn.commit()