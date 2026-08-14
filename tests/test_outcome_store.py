from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_harness.models import MarketRegime, OutcomeRecord
from trading_harness.services.db import Database
from trading_harness.services.outcome_store import PersistedOutcomeStore


def _make_outcome(**overrides):
    defaults = {
        "id": "outcome-test-1",
        "prediction_id": "pred-test-1",
        "agent_id": "agent-test-1",
        "run_id": "run-test-1",
        "snapshot_id": "snap-test-1",
        "symbol": "BTCUSDT",
        "direction_predicted": "LONG",
        "direction_actual": "LONG",
        "confidence_predicted": 0.75,
        "entry_price": 50000.0,
        "exit_price": 51000.0,
        "mfe": 0.02,
        "mae": 0.0,
        "holding_period_bars": 12,
        "realized_pnl": 1000.0,
        "regime": MarketRegime.STRONG_BULL,
        "timestamp": datetime.now(UTC),
    }
    defaults.update(overrides)
    return OutcomeRecord(**defaults)


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------


def test_outcome_store_fallback_add():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    record = _make_outcome(id="outcome-fb-1")
    result = store.add(record)
    assert result.id == "outcome-fb-1"


def test_outcome_store_fallback_get():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    record = _make_outcome(id="outcome-get-1")
    store.add(record)
    retrieved = store.get("outcome-get-1")
    assert retrieved is not None
    assert retrieved.agent_id == "agent-test-1"
    assert retrieved.symbol == "BTCUSDT"


def test_outcome_store_fallback_get_not_found():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    assert store.get("nonexistent") is None


def test_outcome_store_fallback_by_agent():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    store.add(_make_outcome(id="outcome-aa-1", agent_id="agent-alpha"))
    store.add(_make_outcome(id="outcome-aa-2", agent_id="agent-beta"))
    store.add(_make_outcome(id="outcome-aa-3", agent_id="agent-alpha"))

    results = store.by_agent("agent-alpha")
    assert len(results) == 2
    assert {r.id for r in results} == {"outcome-aa-1", "outcome-aa-3"}


def test_outcome_store_fallback_by_run():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    store.add(_make_outcome(id="outcome-br-1", run_id="run-x"))
    store.add(_make_outcome(id="outcome-br-2", run_id="run-y"))
    store.add(_make_outcome(id="outcome-br-3", run_id="run-x"))

    results = store.by_run("run-x")
    assert len(results) == 2
    assert {r.id for r in results} == {"outcome-br-1", "outcome-br-3"}


def test_outcome_store_fallback_by_regime():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    store.add(_make_outcome(id="outcome-breg-1", regime=MarketRegime.STRONG_BULL))
    store.add(_make_outcome(id="outcome-breg-2", regime=MarketRegime.STRONG_BEAR))
    store.add(_make_outcome(id="outcome-breg-3", regime=MarketRegime.STRONG_BULL))

    results = store.by_regime(MarketRegime.STRONG_BULL)
    assert len(results) == 2
    assert {r.id for r in results} == {"outcome-breg-1", "outcome-breg-3"}


def test_outcome_store_fallback_all():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    store.add(_make_outcome(id="outcome-all-1"))
    store.add(_make_outcome(id="outcome-all-2"))
    store.add(_make_outcome(id="outcome-all-3"))

    results = store.all()
    assert len(results) == 3


def test_outcome_store_fallback_generate_long():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    record = store.generate(
        prediction_id="pred-gen-1",
        agent_id="agent-gen-1",
        run_id="run-gen-1",
        snapshot_id="snap-gen-1",
        symbol="ETHUSDT",
        direction_predicted="LONG",
        direction_actual="LONG",
        confidence_predicted=0.8,
        entry_price=3000.0,
        exit_price=3150.0,
        regime=MarketRegime.WEAK_BULL,
    )
    assert record.id == "outcome-pred-gen-1"
    assert record.prediction_id == "pred-gen-1"
    assert record.mfe == pytest.approx(0.05, abs=0.001)
    assert record.mae == 0.0


def test_outcome_store_fallback_generate_short():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    record = store.generate(
        prediction_id="pred-gen-2",
        agent_id="agent-gen-2",
        run_id="run-gen-2",
        snapshot_id="snap-gen-2",
        symbol="SOLUSDT",
        direction_predicted="SHORT",
        direction_actual="SHORT",
        confidence_predicted=0.6,
        entry_price=200.0,
        exit_price=180.0,
        regime=MarketRegime.WEAK_BEAR,
    )
    assert record.id == "outcome-pred-gen-2"
    assert record.direction_predicted == "SHORT"
    assert record.mfe == pytest.approx(0.1, abs=0.001)
    assert record.mae == 0.0


def test_outcome_store_fallback_generate_wrong_direction():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    record = store.generate(
        prediction_id="pred-gen-3",
        agent_id="agent-gen-3",
        run_id="run-gen-3",
        snapshot_id="snap-gen-3",
        symbol="BTCUSDT",
        direction_predicted="LONG",
        direction_actual="SHORT",
        confidence_predicted=0.5,
        entry_price=50000.0,
        exit_price=48000.0,
        regime=MarketRegime.CRASH,
    )
    assert record.mfe == 0.0
    assert record.mae == pytest.approx(0.04, abs=0.001)


def test_outcome_store_fallback_generate_invalid_prices():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    with pytest.raises(ValueError, match="must be positive"):
        store.generate(
            prediction_id="pred-bad",
            agent_id="agent-bad",
            run_id="run-bad",
            snapshot_id="snap-bad",
            symbol="TEST",
            direction_predicted="LONG",
            direction_actual="LONG",
            confidence_predicted=0.5,
            entry_price=-100.0,
            exit_price=50.0,
        )

    with pytest.raises(ValueError, match="must be positive"):
        store.generate(
            prediction_id="pred-bad2",
            agent_id="agent-bad",
            run_id="run-bad",
            snapshot_id="snap-bad",
            symbol="TEST",
            direction_predicted="LONG",
            direction_actual="LONG",
            confidence_predicted=0.5,
            entry_price=100.0,
            exit_price=0.0,
        )


def test_outcome_store_fallback_by_prediction_id():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    store.add(_make_outcome(id="outcome-bp-1", prediction_id="pred-target"))
    store.add(_make_outcome(id="outcome-bp-2", prediction_id="pred-other"))

    result = store.get_by_prediction_id("pred-target")
    assert result is not None
    assert result.id == "outcome-bp-1"
    assert store.get_by_prediction_id("pred-nonexistent") is None


def test_outcome_store_fallback_normalize_direction():
    db = Database("postgresql://nonexistent:5432/test")
    db._ensure_pool()
    store = PersistedOutcomeStore(db)

    record = store.generate(
        prediction_id="pred-norm-1",
        agent_id="agent-norm-1",
        run_id="run-norm-1",
        snapshot_id="snap-norm-1",
        symbol="BTCUSDT",
        direction_predicted="long",
        direction_actual="buy",
        confidence_predicted=0.7,
        entry_price=50000.0,
        exit_price=51000.0,
    )
    assert record.direction_predicted == "LONG"
    assert record.direction_actual == "BUY"