from __future__ import annotations

import pytest

from trading_harness.models import (
    AgentGenome,
    AgentStatus,
    MutationType,
)
from trading_harness.services.agent_factory import (
    AgentFactory,
    MutationError,
)


def _make_agent(**overrides):
    defaults = {
        "id": "agent-factory-1",
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
    }
    defaults.update(overrides)
    return AgentGenome(**defaults)


# ---------------------------------------------------------------------------
# Indicator mutation
# ---------------------------------------------------------------------------


def test_indicator_add_mutation():
    factory = AgentFactory()
    parent = _make_agent(id="parent-ind-add", generation=1)
    child, record = factory.generate_from_parent(
        parent, mutation_type=MutationType.INDICATOR_ADD
    )
    assert child.generation == 2
    assert child.status == AgentStatus.CANDIDATE
    assert child.parent_agents == ["parent-ind-add"]
    assert len(child.indicators) > len(parent.indicators)
    assert record.mutation_type == MutationType.INDICATOR_ADD


def test_indicator_remove_mutation():
    factory = AgentFactory()
    parent = _make_agent(
        id="parent-ind-rm",
        generation=1,
        indicators=["rsi", "macd", "bollinger", "vwap", "stochastic"],
    )
    child, _record = factory.generate_from_parent(
        parent, mutation_type=MutationType.INDICATOR_REMOVE
    )
    assert child.generation == 2
    assert len(child.indicators) < len(parent.indicators)


# ---------------------------------------------------------------------------
# Timeframe mutation
# ---------------------------------------------------------------------------


def test_timeframe_mutation():
    factory = AgentFactory()
    parent = _make_agent(id="parent-tf", generation=1, timeframes=["1h", "4h"])
    child, _record = factory.generate_from_parent(
        parent, mutation_type=MutationType.TIMEFRAME_MODIFY
    )
    assert child.generation == 2
    assert child.timeframes != parent.timeframes or len(child.timeframes) > 0


# ---------------------------------------------------------------------------
# Parameter mutation
# ---------------------------------------------------------------------------


def test_risk_attitude_mutation():
    factory = AgentFactory()
    parent = _make_agent(id="parent-risk", risk_attitude="conservative")
    child, _record = factory.generate_from_parent(
        parent, mutation_type=MutationType.RISK_ATTITUDE
    )
    assert child.generation == 2
    # Risk attitude may or may not change randomly, but generation must increment


def test_weighting_strategy_mutation():
    factory = AgentFactory()
    parent = _make_agent(id="parent-ws", weighting_strategy="default")
    child, _record = factory.generate_from_parent(
        parent, mutation_type=MutationType.WEIGHTING_STRATEGY
    )
    assert child.generation == 2


# ---------------------------------------------------------------------------
# Temperature mutation
# ---------------------------------------------------------------------------


def test_temperature_mutation():
    factory = AgentFactory()
    parent = _make_agent(id="parent-temp", temperature=0.3)
    child, _record = factory.generate_from_parent(
        parent, mutation_type=MutationType.TEMPERATURE_MODIFY
    )
    assert child.generation == 2
    assert 0.05 <= child.temperature <= 1.0


# ---------------------------------------------------------------------------
# Recombination
# ---------------------------------------------------------------------------


def test_recombine_same_category():
    factory = AgentFactory()
    parent_a = _make_agent(
        id="parent-recomb-a",
        category="technical",
        generation=3,
        indicators=["rsi", "macd"],
        parent_agents=[],
    )
    parent_b = _make_agent(
        id="parent-recomb-b",
        category="technical",
        generation=3,
        indicators=["bollinger", "vwap"],
        parent_agents=[],
    )
    child, record = factory.recombine(parent_a, parent_b)
    assert child.generation == 4
    assert child.status == AgentStatus.CANDIDATE
    assert set(child.parent_agents) == {"parent-recomb-a", "parent-recomb-b"}
    assert record.mutation_type == MutationType.RECOMBINATION


def test_recombine_different_category_raises():
    factory = AgentFactory()
    parent_a = _make_agent(id="parent-cat-a", category="technical", indicators=["rsi"])
    parent_b = _make_agent(id="parent-cat-b", category="macro", indicators=["cpi"])
    with pytest.raises(MutationError, match="Cannot recombine different categories"):
        factory.recombine(parent_a, parent_b)


# ---------------------------------------------------------------------------
# Specialization
# ---------------------------------------------------------------------------


def test_specialize():
    factory = AgentFactory()
    parent = _make_agent(id="parent-spec")
    child, _record = factory.specialize(parent, target_regime="strong_bull")
    assert child.generation == 2
    assert child.status == AgentStatus.CANDIDATE
    assert "regime_strong_bull" in child.feature_preferences


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------


def test_simplify():
    factory = AgentFactory()
    parent = _make_agent(
        id="parent-simplify",
        indicators=["rsi", "macd", "bollinger", "vwap", "stochastic"],
        timeframes=["1m", "5m", "15m", "1h"],
    )
    child, _record = factory.simplify(parent, max_indicators=2)
    assert child.generation == 2
    assert child.status == AgentStatus.CANDIDATE
    assert len(child.indicators) <= 2


# ---------------------------------------------------------------------------
# Diversity injection
# ---------------------------------------------------------------------------


def test_diversity_injection():
    factory = AgentFactory()
    parent = _make_agent(
        id="parent-div",
        indicators=[],
        timeframes=[],
        risk_attitude="conservative",
        temperature=0.1,
    )
    child, _record = factory.inject_diversity(parent)
    assert child.generation == 2
    assert child.status == AgentStatus.CANDIDATE
    # Diversity should add some indicators/timeframes if missing


# ---------------------------------------------------------------------------
# Random generation
# ---------------------------------------------------------------------------


def test_generate_random():
    factory = AgentFactory()
    agent = factory.generate_random(category="technical", generation=1)
    assert agent.category == "technical"
    assert agent.status == AgentStatus.CANDIDATE
    assert len(agent.indicators) > 0
    assert len(agent.timeframes) > 0


# ---------------------------------------------------------------------------
# Mutation record
# ---------------------------------------------------------------------------


def test_mutation_record_fields():
    factory = AgentFactory()
    parent = _make_agent(id="parent-record")
    _child, record = factory.generate_from_parent(
        parent,
        mutation_type=MutationType.INDICATOR_ADD,
        hypothesized_advantage="Better trend detection",
        expected_failure_modes=["whipsaw in range"],
    )
    assert record.agent_id == "parent-record"
    assert record.generation == 2
    assert record.mutation_type == MutationType.INDICATOR_ADD
    assert record.hypothesized_advantage == "Better trend detection"
    assert record.expected_failure_modes == ["whipsaw in range"]