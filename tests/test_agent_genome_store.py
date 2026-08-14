from __future__ import annotations

from datetime import UTC, datetime

from trading_harness.models import AgentGenome, AgentStatus
from trading_harness.services.agent_genome_store import AgentGenomeStore, PersistedAgentGenomeStore
from trading_harness.services.db import Database


def _make_agent(**overrides):
    defaults = {
        "id": "agent-genome-test-1",
        "generation": 1,
        "parent_agents": [],
        "category": "technical",
        "status": AgentStatus.GENERATED,
        "prompt_version": "1",
        "reasoning_style": "systematic",
        "indicators": ["rsi", "macd"],
        "timeframes": ["1h", "4h"],
        "feature_preferences": ["momentum"],
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


# ---------------------------------------------------------------------------
# AgentGenomeStore (in-memory)
# ---------------------------------------------------------------------------


def test_genome_store_add_and_get():
    store = AgentGenomeStore()
    agent = _make_agent(id="gs-1")
    store.add(agent)
    result = store.get("gs-1")
    assert result is not None
    assert result.id == "gs-1"
    assert result.category == "technical"


def test_genome_store_get_not_found():
    store = AgentGenomeStore()
    assert store.get("nonexistent") is None


def test_genome_store_list_all():
    store = AgentGenomeStore()
    store.add(_make_agent(id="gs-list-1", category="technical"))
    store.add(_make_agent(id="gs-list-2", category="macro"))
    all_agents = store.list_all()
    assert len(all_agents) == 2


def test_genome_store_list_by_category():
    store = AgentGenomeStore()
    store.add(_make_agent(id="gs-cat-1", category="technical"))
    store.add(_make_agent(id="gs-cat-2", category="technical"))
    store.add(_make_agent(id="gs-cat-3", category="macro"))
    technical = store.list_by_category("technical")
    assert len(technical) == 2
    assert all(a.category == "technical" for a in technical)


def test_genome_store_list_by_status():
    store = AgentGenomeStore()
    store.add(_make_agent(id="gs-st-1", status=AgentStatus.ACTIVE))
    store.add(_make_agent(id="gs-st-2", status=AgentStatus.GENERATED))
    generated = store.list_by_status(AgentStatus.GENERATED)
    assert len(generated) == 1


def test_genome_store_list_active():
    store = AgentGenomeStore()
    store.add(_make_agent(id="gs-act-1", category="technical", status=AgentStatus.ACTIVE))
    store.add(_make_agent(id="gs-act-2", category="technical", status=AgentStatus.CHAMPION))
    store.add(_make_agent(id="gs-act-3", category="technical", status=AgentStatus.GENERATED))
    active = store.list_active("technical")
    assert len(active) == 2


def test_genome_store_list_challengers():
    store = AgentGenomeStore()
    store.add(_make_agent(id="gs-ch-1", category="technical", status=AgentStatus.CHALLENGER))
    store.add(_make_agent(id="gs-ch-2", category="technical", status=AgentStatus.CHALLENGER))
    store.add(_make_agent(id="gs-ch-3", category="technical", status=AgentStatus.ACTIVE))
    challengers = store.list_challengers("technical")
    assert len(challengers) == 2


def test_genome_store_list_by_generation():
    store = AgentGenomeStore()
    store.add(_make_agent(id="gs-gen-1", generation=1))
    store.add(_make_agent(id="gs-gen-2", generation=1))
    store.add(_make_agent(id="gs-gen-3", generation=2))
    gen1 = store.list_by_generation(1)
    assert len(gen1) == 2


def test_genome_store_update():
    store = AgentGenomeStore()
    agent = _make_agent(id="gs-upd-1", status=AgentStatus.GENERATED)
    store.add(agent)
    agent.status = AgentStatus.ACTIVE
    store.update(agent)
    result = store.get("gs-upd-1")
    assert result.status == AgentStatus.ACTIVE


def test_genome_store_get_or_create():
    store = AgentGenomeStore()
    agent = store.get_or_create("gs-create-1")
    assert agent is not None
    assert agent.id == "gs-create-1"
    assert agent.category == "generic"


def test_genome_store_get_or_create_existing():
    store = AgentGenomeStore()
    agent = _make_agent(id="gs-oc-1", category="technical")
    store.add(agent)
    retrieved = store.get_or_create("gs-oc-1")
    assert retrieved.id == "gs-oc-1"
    assert retrieved.category == "technical"


# ---------------------------------------------------------------------------
# PersistedAgentGenomeStore (fallback only)
# ---------------------------------------------------------------------------


def test_persisted_genome_store_fallback_add():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedAgentGenomeStore(db)
    agent = _make_agent(id="pgs-1")
    result = store.add(agent)
    assert result.id == "pgs-1"


def test_persisted_genome_store_fallback_get():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedAgentGenomeStore(db)
    agent = _make_agent(id="pgs-2")
    store.add(agent)
    result = store.get("pgs-2")
    assert result is not None
    assert result.id == "pgs-2"


def test_persisted_genome_store_fallback_not_found():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedAgentGenomeStore(db)
    assert store.get("nonexistent") is None


def test_persisted_genome_store_fallback_list_all():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedAgentGenomeStore(db)
    store.add(_make_agent(id="pgs-l1"))
    store.add(_make_agent(id="pgs-l2"))
    all_agents = store.list_all()
    assert len(all_agents) == 2


def test_persisted_genome_store_fallback_list_by_category():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedAgentGenomeStore(db)
    store.add(_make_agent(id="pgs-c1", category="technical"))
    store.add(_make_agent(id="pgs-c2", category="macro"))
    technical = store.list_by_category("technical")
    assert len(technical) == 1
    assert technical[0].id == "pgs-c1"