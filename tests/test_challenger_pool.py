from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_harness.models import (
    AgentGenome,
    AgentStatus,
    ChampionChallenger,
    EvolutionRun,
)
from trading_harness.services.agent_genome_store import AgentGenomeStore
from trading_harness.services.challenger_pool import ChallengerPool
from trading_harness.services.evolution import PromotionPolicy


def _make_agent(**overrides):
    defaults = {
        "id": "agent-cp-1",
        "generation": 1,
        "parent_agents": [],
        "category": "technical",
        "status": AgentStatus.ACTIVE,
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


def _make_policy():
    return PromotionPolicy({"promotion": {
        "minimum_observations": 10,
        "relative_improvement_min": 0.03,
        "require_out_of_sample": True,
        "require_walk_forward": True,
        "require_shadow_mode": True,
        "require_positive_ensemble_contribution": True,
        "require_security_pass": True,
    }})


def _make_pool():
    store = AgentGenomeStore()
    policy = _make_policy()
    pop_policy = {"categories": {"technical": {"active": 5, "challengers": 10}}}
    return ChallengerPool(store, policy, pop_policy)


# ---------------------------------------------------------------------------
# Challenger management
# ---------------------------------------------------------------------------


def test_add_challenger():
    pool = _make_pool()
    agent = _make_agent(id="cp-add-1")
    result = pool.add_challenger(agent)
    assert result.status == AgentStatus.CHALLENGER


def test_get_challengers():
    pool = _make_pool()
    pool.add_challenger(_make_agent(id="cp-ch-1"))
    pool.add_challenger(_make_agent(id="cp-ch-2"))
    challengers = pool.get_challengers("technical")
    assert len(challengers) == 2
    assert all(c.status == AgentStatus.CHALLENGER for c in challengers)


def test_get_challengers_empty():
    pool = _make_pool()
    assert pool.get_challengers("technical") == []


def test_get_active_agents():
    pool = _make_pool()
    pool._store.add(_make_agent(id="cp-act-1", status=AgentStatus.ACTIVE))
    pool._store.add(_make_agent(id="cp-act-2", status=AgentStatus.CHAMPION))
    pool._store.add(_make_agent(id="cp-gen-1", status=AgentStatus.GENERATED))
    active = pool.get_active_agents("technical")
    assert len(active) == 2


# ---------------------------------------------------------------------------
# Champion/Challenger pairing
# ---------------------------------------------------------------------------


def test_pair_for_evaluation():
    pool = _make_pool()
    pool.add_challenger(_make_agent(id="cp-pair-ch", category="technical"))
    pool._store.add(_make_agent(id="cp-pair-champ", status=AgentStatus.ACTIVE, category="technical"))
    pair = pool.pair_for_evaluation("technical")
    assert pair is not None
    assert pair.champion_id == "cp-pair-champ"
    assert pair.challenger_id == "cp-pair-ch"
    assert pair.category == "technical"


def test_pair_for_evaluation_no_challenger():
    pool = _make_pool()
    pool._store.add(_make_agent(id="cp-noc-1", status=AgentStatus.ACTIVE, category="technical"))
    assert pool.pair_for_evaluation("technical") is None


def test_pair_for_evaluation_no_active():
    pool = _make_pool()
    pool.add_challenger(_make_agent(id="cp-noa-1", category="technical"))
    assert pool.pair_for_evaluation("technical") is None


# ---------------------------------------------------------------------------
# Promotion evaluation
# ---------------------------------------------------------------------------


def test_promotion_approved():
    pool = _make_pool()
    decision = pool.evaluate_promotion(
        champion_id="champ-1",
        challenger_id="chal-1",
        category="technical",
        challenger_score=0.40,
        incumbent_score=0.30,
        observations=150,
        out_of_sample_pass=True,
        walk_forward_pass=True,
        shadow_pass=True,
        ensemble_contribution=0.05,
        security_pass=True,
    )
    assert decision.promote is True
    assert decision.reason == "PROMOTION_APPROVED"


def test_promotion_insufficient_observations():
    pool = _make_pool()
    decision = pool.evaluate_promotion(
        champion_id="champ-1",
        challenger_id="chal-1",
        category="technical",
        challenger_score=0.50,
        incumbent_score=0.30,
        observations=5,
    )
    assert decision.promote is False
    assert decision.reason == "INSUFFICIENT_OBSERVATIONS"


def test_promotion_oos_fail():
    pool = _make_pool()
    decision = pool.evaluate_promotion(
        champion_id="champ-1",
        challenger_id="chal-1",
        category="technical",
        challenger_score=0.50,
        incumbent_score=0.30,
        observations=150,
        out_of_sample_pass=False,
    )
    assert decision.promote is False
    assert decision.reason == "OUT_OF_SAMPLE_FAILED"


def test_promotion_margin_not_met():
    pool = _make_pool()
    decision = pool.evaluate_promotion(
        champion_id="champ-1",
        challenger_id="chal-1",
        category="technical",
        challenger_score=0.302,
        incumbent_score=0.30,
        observations=150,
    )
    assert decision.promote is False
    assert decision.reason == "PROMOTION_MARGIN_NOT_MET"


# ---------------------------------------------------------------------------
# Promotion execution
# ---------------------------------------------------------------------------


def test_promote_challenger():
    pool = _make_pool()
    pool.add_challenger(_make_agent(id="cp-prom-chal"))
    pool._store.add(_make_agent(id="cp-prom-champ", status=AgentStatus.ACTIVE))
    run = pool.promote_challenger("cp-prom-chal", "cp-prom-champ", "technical")
    chal = pool._store.get("cp-prom-chal")
    champ = pool._store.get("cp-prom-champ")
    assert chal.status == AgentStatus.ACTIVE
    assert champ.status == AgentStatus.PROBATION
    assert isinstance(run, EvolutionRun)
    assert run.method == "champion_challenger_promotion"


def test_promote_challenger_not_found():
    pool = _make_pool()
    with pytest.raises(ValueError, match="not found"):
        pool.promote_challenger("nonexistent", "also-nonexistent", "technical")


# ---------------------------------------------------------------------------
# Demotion
# ---------------------------------------------------------------------------


def test_demote_to_probation():
    pool = _make_pool()
    pool._store.add(_make_agent(id="cp-demote-1", status=AgentStatus.ACTIVE))
    pool.demote_to_probation("cp-demote-1")
    agent = pool._store.get("cp-demote-1")
    assert agent.status == AgentStatus.PROBATION


def test_demote_to_probation_not_found():
    pool = _make_pool()
    with pytest.raises(ValueError, match="not found"):
        pool.demote_to_probation("nonexistent")


# ---------------------------------------------------------------------------
# Population stats
# ---------------------------------------------------------------------------


def test_get_category_size():
    pool = _make_pool()
    stats = pool.get_category_size("technical")
    assert stats["category"] == "technical"
    assert "active" in stats
    assert "challengers" in stats
    assert stats["active_limit"] == 5
    assert stats["challenger_limit"] == 10


def test_get_challenger_pairs():
    pool = _make_pool()
    pool.add_challenger(_make_agent(id="cp-pair-chal"))
    pool._store.add(_make_agent(id="cp-pair-champ", status=AgentStatus.ACTIVE))
    pool.pair_for_evaluation("technical")
    pairs = pool.get_challenger_pairs("technical")
    assert len(pairs) == 1
    assert isinstance(pairs[0], ChampionChallenger)