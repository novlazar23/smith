from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_harness.models import AgentGenome, AgentStatus, MarketSnapshot
from trading_harness.services.db import Database
from trading_harness.services.persisted_agent_registry import PersistedAgentRegistry
from trading_harness.services.persisted_snapshot_store import PersistedSnapshotStore


def _make_agent(**overrides):
    defaults = {
        "id": "agent-test-1",
        "generation": 1,
        "parent_agents": [],
        "category": "technical",
        "status": AgentStatus.GENERATED,
        "prompt_version": "1",
        "reasoning_style": "systematic",
        "indicators": ["rsi", "macd"],
        "timeframes": ["1h", "4h"],
        "feature_preferences": ["momentum", "mean_reversion"],
        "statistical_methods": ["bayesian"],
        "weighting_strategy": "performance_weighted",
        "confidence_calibration": "isotonic",
        "risk_attitude": "conservative",
        "context_window_strategy": "bounded",
        "output_schema": "signal-v1",
        "model_profile": "local-main",
        "temperature": 0.2,
        "created_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return AgentGenome(**defaults)


def _make_snapshot(**overrides):
    defaults = {
        "id": "snap-test-1",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(UTC),
        "data": {"price": 50000, "volume": 1000},
    }
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


# ---------------------------------------------------------------------------
# Database module
# ---------------------------------------------------------------------------


def test_db_not_available_without_pool():
    db = Database("postgresql://nonexistent:5432/test")
    assert db.is_available is False
    rows = db.execute("SELECT 1")
    assert rows == []
    db.execute_write("INSERT INTO x VALUES (1)")  # Should not raise


def test_db_fallback_gracefully():
    """When DB is unavailable, operations should return empty / no-ops."""
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()  # Should catch exception and set _ready = False
    assert db.is_available is False
    assert db.is_connected is False


# ---------------------------------------------------------------------------
# PersistedAgentRegistry (no-DB fallback)
# ---------------------------------------------------------------------------


def test_persisted_agent_registry_fallback_add():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    reg = PersistedAgentRegistry(db)

    agent = _make_agent(id="agent-fb-1")
    result = reg.add(agent)
    assert result.id == "agent-fb-1"

    retrieved = reg.get("agent-fb-1")
    assert retrieved is not None
    assert retrieved.id == "agent-fb-1"
    assert retrieved.category == "technical"


def test_persisted_agent_registry_fallback_list():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    reg = PersistedAgentRegistry(db)

    reg.add(_make_agent(id="agent-list-1"))
    reg.add(_make_agent(id="agent-list-2"))
    agents = reg.list()
    assert len(agents) == 2
    assert {a.id for a in agents} == {"agent-list-1", "agent-list-2"}


def test_persisted_agent_registry_fallback_not_found():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    reg = PersistedAgentRegistry(db)

    assert reg.get("nonexistent") is None


def test_persisted_agent_registry_fallback_version():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    reg = PersistedAgentRegistry(db)

    reg.add(_make_agent(id="agent-ver-1"))
    version = reg.get_version("agent-ver-1")
    assert int(version) >= 1


# ---------------------------------------------------------------------------
# PersistedSnapshotStore (no-DB fallback)
# ---------------------------------------------------------------------------


def test_persisted_snapshot_store_fallback_add():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedSnapshotStore(db)

    snap = _make_snapshot()
    result = store.add(snap)
    assert result.id == "snap-test-1"
    assert result.content_hash is not None


def test_persisted_snapshot_store_fallback_get():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedSnapshotStore(db)

    snap = _make_snapshot()
    store.add(snap)
    retrieved = store.get("snap-test-1")
    assert retrieved is not None
    assert retrieved.symbol == "BTCUSDT"
    assert retrieved.content_hash is not None


def test_persisted_snapshot_store_fallback_not_found():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedSnapshotStore(db)

    assert store.get("nonexistent") is None


def test_persisted_snapshot_store_hash_consistency():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedSnapshotStore(db)

    snap = _make_snapshot()
    store.add(snap)

    assert snap.content_hash is not None
    assert len(snap.content_hash) == 64  # SHA-256 hex length


def test_persisted_agent_registry_fallback_duplicate():
    """Adding a duplicate agent should raise in fallback mode (thread-safe registry)."""
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    reg = PersistedAgentRegistry(db)

    agent = _make_agent(id="agent-dup-1")
    reg.add(agent)
    # Fallback uses AgentRegistry which raises on duplicates (thread-safe)
    agent2 = _make_agent(id="agent-dup-1", generation=2)
    with pytest.raises(ValueError, match="already exists"):
        reg.add(agent2)
    result = reg.get("agent-dup-1")
    assert result is not None
    assert result.generation == 1  # First agent preserved (default generation=1)