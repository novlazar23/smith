from __future__ import annotations

import pytest

from trading_harness.models import MarketRegime, OutcomeRecord
from trading_harness.services.outcome_generator import (
    InMemoryOutcomeStore,
    OutcomeGenerator,
)

# ---------------------------------------------------------------------------
# OutcomeGenerator — basic generation
# ---------------------------------------------------------------------------


def test_generate_long_position():
    gen = OutcomeGenerator()
    o = gen.generate(
        prediction_id="p1",
        agent_id="a1",
        run_id="r1",
        snapshot_id="s1",
        symbol="BTCUSDT",
        direction_predicted="LONG",
        direction_actual="LONG",
        confidence_predicted=0.8,
        entry_price=100.0,
        exit_price=110.0,
        realized_pnl=100.0,
    )
    assert o.id.startswith("outcome-")
    assert o.entry_price == 100.0
    assert o.exit_price == 110.0
    assert o.direction_predicted == "LONG"
    assert o.mfe > 0


def test_generate_short_position():
    gen = OutcomeGenerator()
    o = gen.generate(
        prediction_id="p2",
        agent_id="a1",
        run_id="r1",
        snapshot_id="s1",
        symbol="ETHUSDT",
        direction_predicted="SHORT",
        direction_actual="SHORT",
        confidence_predicted=0.6,
        entry_price=2000.0,
        exit_price=1900.0,
    )
    assert o.direction_predicted == "SHORT"
    assert o.mfe > 0


def test_generate_negative_return():
    gen = OutcomeGenerator()
    o = gen.generate(
        prediction_id="p3",
        agent_id="a1",
        run_id="r1",
        snapshot_id="s1",
        symbol="BTCUSDT",
        direction_predicted="LONG",
        direction_actual="SHORT",
        confidence_predicted=0.5,
        entry_price=100.0,
        exit_price=90.0,
    )
    assert o.mae > 0
    assert o.mfe == 0.0


def test_generate_no_trade():
    gen = OutcomeGenerator()
    o = gen.generate(
        prediction_id="p4",
        agent_id="a1",
        run_id="r1",
        snapshot_id="s1",
        symbol="BTCUSDT",
        direction_predicted="NO_TRADE",
        direction_actual="LONG",
        confidence_predicted=0.1,
        entry_price=100.0,
        exit_price=110.0,
    )
    assert o.direction_predicted == "NO_TRADE"
    assert o.mfe == 0.0
    assert o.mae == 0.0


# ---------------------------------------------------------------------------
# OutcomeGenerator — validation
# ---------------------------------------------------------------------------


def test_generate_negative_entry_price_raises():
    gen = OutcomeGenerator()
    with pytest.raises(ValueError, match="positive"):
        gen.generate(
            prediction_id="p1",
            agent_id="a1",
            run_id="r1",
            snapshot_id="s1",
            symbol="BTCUSDT",
            direction_predicted="LONG",
            direction_actual="LONG",
            confidence_predicted=0.8,
            entry_price=-1.0,
            exit_price=10.0,
        )


def test_generate_zero_entry_price_raises():
    gen = OutcomeGenerator()
    with pytest.raises(ValueError, match="positive"):
        gen.generate(
            prediction_id="p1",
            agent_id="a1",
            run_id="r1",
            snapshot_id="s1",
            symbol="BTCUSDT",
            direction_predicted="LONG",
            direction_actual="LONG",
            confidence_predicted=0.8,
            entry_price=0.0,
            exit_price=10.0,
        )


# ---------------------------------------------------------------------------
# OutcomeGenerator — store operations
# ---------------------------------------------------------------------------


def test_add_outcome():
    gen = OutcomeGenerator()
    o = gen.generate(
        prediction_id="p1",
        agent_id="a1",
        run_id="r1",
        snapshot_id="s1",
        symbol="BTCUSDT",
        direction_predicted="LONG",
        direction_actual="LONG",
        confidence_predicted=0.8,
        entry_price=100.0,
        exit_price=110.0,
    )
    added = gen.add(o)
    assert added.id == o.id


def test_get_outcome():
    gen = OutcomeGenerator()
    gen.generate(
        prediction_id="p1",
        agent_id="a1",
        run_id="r1",
        snapshot_id="s1",
        symbol="BTCUSDT",
        direction_predicted="LONG",
        direction_actual="LONG",
        confidence_predicted=0.8,
        entry_price=100.0,
        exit_price=110.0,
    )
    stored = gen.get(gen.store.all()[0].id)
    assert stored is not None
    assert stored.prediction_id == "p1"


def test_get_unknown_outcome():
    gen = OutcomeGenerator()
    assert gen.get("nonexistent") is None


def test_by_agent():
    gen = OutcomeGenerator()
    gen.generate(
        prediction_id="p1", agent_id="a1", run_id="r1", snapshot_id="s1",
        symbol="BTCUSDT", direction_predicted="LONG", direction_actual="LONG",
        confidence_predicted=0.8, entry_price=100.0, exit_price=110.0,
    )
    gen.generate(
        prediction_id="p2", agent_id="a2", run_id="r1", snapshot_id="s1",
        symbol="ETHUSDT", direction_predicted="SHORT", direction_actual="SHORT",
        confidence_predicted=0.6, entry_price=2000.0, exit_price=1900.0,
    )
    results = gen.by_agent("a1")
    assert len(results) == 1
    assert results[0].prediction_id == "p1"


def test_by_run():
    gen = OutcomeGenerator()
    gen.generate(
        prediction_id="p1", agent_id="a1", run_id="r1", snapshot_id="s1",
        symbol="BTCUSDT", direction_predicted="LONG", direction_actual="LONG",
        confidence_predicted=0.8, entry_price=100.0, exit_price=110.0,
    )
    gen.generate(
        prediction_id="p2", agent_id="a2", run_id="r2", snapshot_id="s1",
        symbol="ETHUSDT", direction_predicted="SHORT", direction_actual="SHORT",
        confidence_predicted=0.6, entry_price=2000.0, exit_price=1900.0,
    )
    results = gen.by_run("r1")
    assert len(results) == 1


def test_by_regime():
    gen = OutcomeGenerator()
    gen.generate(
        prediction_id="p1", agent_id="a1", run_id="r1", snapshot_id="s1",
        symbol="BTCUSDT", direction_predicted="LONG", direction_actual="LONG",
        confidence_predicted=0.8, entry_price=100.0, exit_price=110.0,
        regime=MarketRegime.STRONG_BULL,
    )
    gen.generate(
        prediction_id="p2", agent_id="a2", run_id="r1", snapshot_id="s1",
        symbol="ETHUSDT", direction_predicted="SHORT", direction_actual="SHORT",
        confidence_predicted=0.6, entry_price=2000.0, exit_price=1900.0,
        regime=MarketRegime.WEAK_BULL,
    )
    results = gen.by_regime(MarketRegime.STRONG_BULL)
    assert len(results) == 1


def test_all_outcomes():
    gen = OutcomeGenerator()
    gen.generate(
        prediction_id="p1", agent_id="a1", run_id="r1", snapshot_id="s1",
        symbol="BTCUSDT", direction_predicted="LONG", direction_actual="LONG",
        confidence_predicted=0.8, entry_price=100.0, exit_price=110.0,
    )
    gen.generate(
        prediction_id="p2", agent_id="a2", run_id="r1", snapshot_id="s1",
        symbol="ETHUSDT", direction_predicted="SHORT", direction_actual="SHORT",
        confidence_predicted=0.6, entry_price=2000.0, exit_price=1900.0,
    )
    assert len(gen.all()) == 2


def test_get_by_prediction_id():
    gen = OutcomeGenerator()
    gen.generate(
        prediction_id="target-pred", agent_id="a1", run_id="r1", snapshot_id="s1",
        symbol="BTCUSDT", direction_predicted="LONG", direction_actual="LONG",
        confidence_predicted=0.8, entry_price=100.0, exit_price=110.0,
    )
    result = gen.get_by_prediction_id("target-pred")
    assert result is not None
    assert result.prediction_id == "target-pred"


def test_get_by_prediction_id_not_found():
    gen = OutcomeGenerator()
    gen.generate(
        prediction_id="p1", agent_id="a1", run_id="r1", snapshot_id="s1",
        symbol="BTCUSDT", direction_predicted="LONG", direction_actual="LONG",
        confidence_predicted=0.8, entry_price=100.0, exit_price=110.0,
    )
    assert gen.get_by_prediction_id("nonexistent") is None


# ---------------------------------------------------------------------------
# InMemoryOutcomeStore — standalone
# ---------------------------------------------------------------------------


def test_in_memory_store_add_and_get():
    store = InMemoryOutcomeStore()
    record = OutcomeRecord(
        prediction_id="p1",
        agent_id="a1",
        run_id="r1",
        snapshot_id="s1",
        symbol="BTCUSDT",
        direction_predicted="LONG",
        direction_actual="LONG",
        confidence_predicted=0.8,
        entry_price=100.0,
        exit_price=110.0,
    )
    stored = store.add(record)
    assert stored.id == record.id
    assert store.get(record.id).id == record.id


def test_in_memory_store_by_agent():
    store = InMemoryOutcomeStore()
    store.add(OutcomeRecord(
        prediction_id="p1", agent_id="a1", run_id="r1", snapshot_id="s1",
        symbol="BTCUSDT", direction_predicted="LONG", direction_actual="LONG",
        confidence_predicted=0.8, entry_price=100.0, exit_price=110.0,
    ))
    store.add(OutcomeRecord(
        prediction_id="p2", agent_id="a2", run_id="r1", snapshot_id="s1",
        symbol="ETHUSDT", direction_predicted="SHORT", direction_actual="SHORT",
        confidence_predicted=0.6, entry_price=2000.0, exit_price=1900.0,
    ))
    results = store.by_agent("a1")
    assert len(results) == 1
    assert results[0].prediction_id == "p1"


# ---------------------------------------------------------------------------
# OutcomeGenerator — custom store
# ---------------------------------------------------------------------------


def test_custom_store_integration():
    custom_store = InMemoryOutcomeStore()
    gen = OutcomeGenerator(store=custom_store)
    gen.generate(
        prediction_id="p1",
        agent_id="a1",
        run_id="r1",
        snapshot_id="s1",
        symbol="BTCUSDT",
        direction_predicted="LONG",
        direction_actual="LONG",
        confidence_predicted=0.8,
        entry_price=100.0,
        exit_price=110.0,
    )
    # Verify the custom store received the outcome
    assert len(custom_store.all()) == 1