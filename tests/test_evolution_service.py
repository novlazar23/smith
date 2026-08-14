from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_harness.models import (
    AgentGenome,
    AgentStatus,
    EvolutionRun,
    MutationType,
    RollbackEntry,
)
from trading_harness.services.agent_genome_store import AgentGenomeStore
from trading_harness.services.evolution import PromotionPolicy
from trading_harness.services.evolution_service import EvolutionService


def _make_agent(**overrides):
    defaults = {
        "id": "agent-es-1",
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


def _make_service():
    store = AgentGenomeStore()
    policy = PromotionPolicy({
        "promotion": {
            "minimum_observations": 10,
            "relative_improvement_min": 0.03,
            "require_out_of_sample": True,
            "require_walk_forward": True,
            "require_shadow_mode": True,
            "require_positive_ensemble_contribution": True,
            "require_security_pass": True,
        }
    })
    pop_policy = {"categories": {"technical": {"active": 5, "challengers": 10}}}
    return EvolutionService(store, policy, pop_policy)


# ---------------------------------------------------------------------------
# generate_mutant
# ---------------------------------------------------------------------------


def test_generate_mutant():
    service = _make_service()
    parent = _make_service()._store.get("agent-es-1") or _make_agent(id="parent-mutant")
    service._store.add(parent)
    child, record = service.generate_mutant(
        parent,
        mutation_type="INDICATOR_ADD",
        hypothesized_advantage="Better trend detection",
        expected_failure_modes=["whipsaw in range"],
    )
    assert child.status == AgentStatus.CANDIDATE
    assert child.generation == parent.generation + 1
    assert child.parent_agents == [parent.id]
    assert record.mutation_type == MutationType.INDICATOR_ADD
    assert record.hypothesized_advantage == "Better trend detection"
    assert "whipsaw in range" in record.expected_failure_modes
    assert service._store.get(child.id) is not None


def test_generate_mutant_recombination():
    service = _make_service()
    parent_a = _make_agent(id="parent-rec-a")
    parent_b = _make_agent(id="parent-rec-b")
    service._store.add(parent_a)
    service._store.add(parent_b)
    child, _record = service.recombine(parent_a, parent_b)
    assert child.status == AgentStatus.CANDIDATE
    assert child.generation == max(parent_a.generation, parent_b.generation) + 1


# ---------------------------------------------------------------------------
# recombine
# ---------------------------------------------------------------------------


def test_recombine():
    service = _make_service()
    pa = _make_agent(id="pa-recomb", category="technical")
    pb = _make_agent(id="pb-recomb", category="technical")
    service._store.add(pa)
    service._store.add(pb)
    child, record = service.recombine(pa, pb)
    assert child.status == AgentStatus.CANDIDATE
    assert child.generation == max(pa.generation, pb.generation) + 1
    assert set(child.parent_agents) == {"pa-recomb", "pb-recomb"}
    assert record.mutation_type == MutationType.RECOMBINATION


# ---------------------------------------------------------------------------
# add_challenger
# ---------------------------------------------------------------------------


def test_add_challenger():
    service = _make_service()
    agent = _make_agent(id="chal-1")
    result = service.add_challenger(agent)
    assert result.status == AgentStatus.CHALLENGER


# ---------------------------------------------------------------------------
# evaluate_challenger — promotion approved
# ---------------------------------------------------------------------------


def test_evaluate_promotion_approved():
    service = _make_service()
    decision = service.evaluate_challenger(
        challenger_id="chal-1",
        champion_id="champ-1",
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


def test_evaluate_promotion_insufficient_observations():
    service = _make_service()
    decision = service.evaluate_challenger(
        challenger_id="chal-1",
        champion_id="champ-1",
        category="technical",
        challenger_score=0.50,
        incumbent_score=0.30,
        observations=5,
    )
    assert decision.promote is False
    assert decision.reason == "INSUFFICIENT_OBSERVATIONS"


def test_evaluate_promotion_oos_fail():
    service = _make_service()
    decision = service.evaluate_challenger(
        challenger_id="chal-1",
        champion_id="champ-1",
        category="technical",
        challenger_score=0.50,
        incumbent_score=0.30,
        observations=150,
        out_of_sample_pass=False,
    )
    assert decision.promote is False
    assert decision.reason == "OUT_OF_SAMPLE_FAILED"


def test_evaluate_promotion_margin_not_met():
    service = _make_service()
    decision = service.evaluate_challenger(
        challenger_id="chal-1",
        champion_id="champ-1",
        category="technical",
        challenger_score=0.302,
        incumbent_score=0.30,
        observations=150,
    )
    assert decision.promote is False
    assert decision.reason == "PROMOTION_MARGIN_NOT_MET"


# ---------------------------------------------------------------------------
# promote_challenger
# ---------------------------------------------------------------------------


def test_promote_challenger():
    service = _make_service()
    challenger = _make_agent(id="prom-chal", status=AgentStatus.CHALLENGER)
    incumbent = _make_agent(id="prom-champ", status=AgentStatus.ACTIVE)
    service._store.add(challenger)
    service._store.add(incumbent)
    run = service.promote_challenger("prom-chal", "prom-champ", "technical")
    chal = service._store.get("prom-chal")
    champ = service._store.get("prom-champ")
    assert chal.status == AgentStatus.ACTIVE
    assert champ.status == AgentStatus.PROBATION
    assert isinstance(run, EvolutionRun)
    assert run.method == "champion_challenger_promotion"


# ---------------------------------------------------------------------------
# retire_agent
# ---------------------------------------------------------------------------


def test_retire_agent():
    service = _make_service()
    agent = _make_agent(id="retire-1", status=AgentStatus.ACTIVE)
    service._store.add(agent)
    service.retire_agent(agent, reason="Poor performance", final_score=0.15)
    updated = service._store.get("retire-1")
    assert updated.status == AgentStatus.RETIRED
    rec = service.graveyard.get_by_agent("retire-1")
    assert rec is not None
    assert rec.final_score == 0.15
    assert rec.reason == "Poor performance"
    rollbacks = service.get_rollbacks_for_agent("retire-1")
    assert len(rollbacks) == 1
    assert rollbacks[0].new_status == AgentStatus.RETIRED


# ---------------------------------------------------------------------------
# reject_challenger
# ---------------------------------------------------------------------------


def test_reject_challenger():
    service = _make_service()
    agent = _make_agent(id="reject-1", status=AgentStatus.CHALLENGER)
    service._store.add(agent)
    service.reject_challenger(agent, reason="Failed OOS")
    updated = service._store.get("reject-1")
    assert updated.status == AgentStatus.REJECTED
    rec = service.graveyard.get_by_agent("reject-1")
    assert rec is not None
    assert rec.final_score == 0.0


# ---------------------------------------------------------------------------
# demote_to_probation
# ---------------------------------------------------------------------------


def test_demote_to_probation():
    service = _make_service()
    agent = _make_agent(id="demo-1", status=AgentStatus.ACTIVE)
    service._store.add(agent)
    service.demote_to_probation("demo-1", reason="Shadow mode failed")
    updated = service._store.get("demo-1")
    assert updated.status == AgentStatus.PROBATION
    rollbacks = service.get_rollbacks_for_agent("demo-1")
    assert len(rollbacks) == 1
    assert rollbacks[0].reason == "Shadow mode failed"


def test_demote_to_probation_not_found():
    service = _make_service()
    with pytest.raises(ValueError, match="not found"):
        service.demote_to_probation("nonexistent")


# ---------------------------------------------------------------------------
# rollback_agent_status
# ---------------------------------------------------------------------------


def test_rollback_agent_status():
    service = _make_service()
    agent = _make_agent(id="rb-1", status=AgentStatus.PROBATION)
    service._store.add(agent)
    entry = service.rollback_agent_status("rb-1", AgentStatus.ACTIVE, reason="Restored after error")
    updated = service._store.get("rb-1")
    assert updated.status == AgentStatus.ACTIVE
    assert isinstance(entry, RollbackEntry)
    assert entry.reason == "Restored after error"


def test_rollback_agent_status_not_found():
    service = _make_service()
    with pytest.raises(ValueError, match="not found"):
        service.rollback_agent_status("nonexistent", AgentStatus.ACTIVE, reason="test")


# ---------------------------------------------------------------------------
# get_challenger_pairs
# ---------------------------------------------------------------------------


def test_get_challenger_pairs():
    service = _make_service()
    service._store.add(_make_agent(id="pair-chal", status=AgentStatus.CHALLENGER))
    service._store.add(_make_agent(id="pair-champ", status=AgentStatus.ACTIVE))
    service._challenger_pool.pair_for_evaluation("technical")
    pairs = service.get_challenger_pairs("technical")
    assert len(pairs) >= 1


# ---------------------------------------------------------------------------
# get_population_stats
# ---------------------------------------------------------------------------


def test_get_population_stats():
    service = _make_service()
    stats = service.get_population_stats("technical")
    assert stats["category"] == "technical"
    assert "active" in stats
    assert "challengers" in stats
    assert stats["active_limit"] == 5


# ---------------------------------------------------------------------------
# get_rollbacks
# ---------------------------------------------------------------------------


def test_get_rollbacks_empty():
    service = _make_service()
    assert service.get_rollbacks() == []


def test_get_rollbacks_populated():
    service = _make_service()
    service._rollback("agent-x", AgentStatus.ACTIVE, AgentStatus.PROBATION, "test reason")
    rollbacks = service.get_rollbacks()
    assert len(rollbacks) == 1
    assert rollbacks[0].agent_id == "agent-x"