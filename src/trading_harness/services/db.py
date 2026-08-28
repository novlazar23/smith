from __future__ import annotations

import json
import logging
import time
from typing import Any

from psycopg import adapters
from psycopg.rows import dict_row
from psycopg.types.json import JsonbDumper
from psycopg.types.json import set_json_dumps as _psycopg_set_json_dumps
from psycopg_pool import ConnectionPool
from psycopg_pool import PoolTimeout as PsycopgPoolTimeout

logger = logging.getLogger(__name__)

# JSONB-Spalten werden direkt mit Python-dict-Werten aus den Stores gefüttert
# (z.B. agent_analysis_results.raw_response, portfolio_states.positions).
# psycopg3 adaptiert dict nicht automatisch; der globale Dumper serialisiert
# dict-Werte als JSONB. Listen werden NICHT global registriert, weil TEXT[]-
# Spalten (z.B. agents.indicators) auf Listen setzen.
_psycopg_set_json_dumps(lambda obj: json.dumps(obj, default=str))
adapters.register_dumper(dict, JsonbDumper)

# Cooldown after a failed connection attempt: stores gate on is_available, so a
# dead database would otherwise trigger a pool-open attempt (up to timeout=1s)
# on every request.
RETRY_COOLDOWN_SECONDS = 30.0

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

CREATE TABLE IF NOT EXISTS agent_analysis_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    category TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT NOT NULL DEFAULT '',
    signals JSONB NOT NULL DEFAULT '[]',
    risks JSONB NOT NULL DEFAULT '[]',
    prompt_version TEXT NOT NULL DEFAULT '1',
    model_profile TEXT NOT NULL DEFAULT 'local-main',
    raw_response JSONB NOT NULL DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS performance_records (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    outcome TEXT,
    realized_pnl REAL NOT NULL DEFAULT 0,
    mfe REAL NOT NULL DEFAULT 0,
    mae REAL NOT NULL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL
);

-- Phase 4: Paper Trading tables
CREATE TABLE IF NOT EXISTS paper_trades (
    id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    equity REAL NOT NULL,
    entry_price REAL NOT NULL,
    requested_leverage REAL NOT NULL DEFAULT 1.0,
    requested_quantity REAL NOT NULL DEFAULT 1.0,
    actual_quantity REAL NOT NULL DEFAULT 0,
    actual_price REAL NOT NULL DEFAULT 0,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL DEFAULT 0,
    fill_rate REAL NOT NULL DEFAULT 0.8,
    slippage_bps REAL NOT NULL DEFAULT 0,
    fees REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'PENDING',
    partial_fills JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL,
    filled_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    reject_reason TEXT
);

CREATE TABLE IF NOT EXISTS paper_positions (
    id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    fees REAL NOT NULL DEFAULT 0,
    current_price REAL NOT NULL DEFAULT 0,
    unrealized_pnl REAL NOT NULL DEFAULT 0,
    realized_pnl REAL NOT NULL DEFAULT 0,
    stop_price REAL NOT NULL DEFAULT 0,
    target_price REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'OPEN',
    open_timestamp TIMESTAMPTZ NOT NULL,
    close_timestamp TIMESTAMPTZ,
    close_price REAL,
    close_reason TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_states (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    start_equity REAL NOT NULL DEFAULT 100000,
    current_equity REAL NOT NULL DEFAULT 100000,
    total_realized_pnl REAL NOT NULL DEFAULT 0,
    total_unrealized_pnl REAL NOT NULL DEFAULT 0,
    max_drawdown REAL NOT NULL DEFAULT 0,
    current_drawdown REAL NOT NULL DEFAULT 0,
    peak_equity REAL NOT NULL DEFAULT 100000,
    positions JSONB NOT NULL DEFAULT '{}',
    symbols TEXT[] NOT NULL DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL
);
"""


class Database:
    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._pool: ConnectionPool | None = None
        self._ready = False
        self._retry_after = 0.0

    def initialize(self) -> bool:
        """Eagerly ensure pool + schema. Returns True when PostgreSQL is usable.

        Called at application startup so the schema is created and stores never
        silently run in their in-memory fallback while Postgres is reachable.
        """
        self._ensure_pool()
        return self._ready

    @property
    def is_available(self) -> bool:
        # Lazy init: stores gate on is_available BEFORE calling execute()/write(),
        # so is_available itself must trigger the pool attempt. Otherwise the pool
        # is never created and every persisted store silently falls back to memory.
        self._ensure_pool()
        return self._pool is not None and self._ready

    def _ensure_pool(self) -> None:
        """Lazily initialize the connection pool on first use."""
        if self._ready:
            return
        if time.monotonic() < self._retry_after:
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
            self._retry_after = time.monotonic() + RETRY_COOLDOWN_SECONDS
            logger.warning("PostgreSQL not available, stores will use in-memory fallback")

    @property
    def is_connected(self) -> bool:
        self._ensure_pool()
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