from trading_harness.models import PerformanceRecord
from trading_harness.services.performance_store import PersistedPerformanceStore


def _make_store():
    """Create a PersistedPerformanceStore with no DB (uses in-memory fallback)."""
    return PersistedPerformanceStore(db=None)


def test_add_and_get():
    store = _make_store()
    record = PerformanceRecord(
        run_id="run-1",
        agent_id="tech-1",
        snapshot_id="snap-1",
        direction="LONG",
        confidence=0.75,
    )
    result = store.add(record)
    assert result.id == record.id
    assert store.get(record.id) is result


def test_get_nonexistent_returns_none():
    store = _make_store()
    assert store.get("nonexistent") is None


def test_by_run():
    store = _make_store()
    r1 = store.add(PerformanceRecord(
        run_id="run-1", agent_id="tech-1", snapshot_id="snap-1",
        direction="LONG", confidence=0.75,
    ))
    store.add(PerformanceRecord(
        run_id="run-2", agent_id="tech-2", snapshot_id="snap-1",
        direction="SHORT", confidence=0.60,
    ))
    r3 = store.add(PerformanceRecord(
        run_id="run-1", agent_id="tech-2", snapshot_id="snap-1",
        direction="LONG", confidence=0.80,
    ))
    by_run1 = store.by_run("run-1")
    assert len(by_run1) == 2
    assert {r.id for r in by_run1} == {r1.id, r3.id}


def test_by_run_empty():
    store = _make_store()
    assert store.by_run("nonexistent") == []


def test_by_agent():
    store = _make_store()
    store.add(PerformanceRecord(
        run_id="run-1", agent_id="tech-1", snapshot_id="snap-1",
        direction="LONG", confidence=0.75,
    ))
    store.add(PerformanceRecord(
        run_id="run-2", agent_id="tech-2", snapshot_id="snap-1",
        direction="SHORT", confidence=0.60,
    ))
    store.add(PerformanceRecord(
        run_id="run-1", agent_id="tech-1", snapshot_id="snap-2",
        direction="SHORT", confidence=0.50,
    ))
    by_agent = store.by_agent("tech-1")
    assert len(by_agent) == 2


def test_by_agent_empty():
    store = _make_store()
    assert store.by_agent("nonexistent") == []


def test_by_snapshot():
    store = _make_store()
    store.add(PerformanceRecord(
        run_id="run-1", agent_id="tech-1", snapshot_id="snap-1",
        direction="LONG", confidence=0.75,
    ))
    store.add(PerformanceRecord(
        run_id="run-2", agent_id="tech-2", snapshot_id="snap-2",
        direction="SHORT", confidence=0.60,
    ))
    by_snap = store.by_snapshot("snap-1")
    assert len(by_snap) == 1
    assert by_snap[0].direction == "LONG"


def test_by_snapshot_empty():
    store = _make_store()
    assert store.by_snapshot("nonexistent") == []


def test_list_returns_all():
    store = _make_store()
    store.add(PerformanceRecord(
        run_id="run-1", agent_id="tech-1", snapshot_id="snap-1",
        direction="LONG", confidence=0.75,
    ))
    store.add(PerformanceRecord(
        run_id="run-2", agent_id="tech-2", snapshot_id="snap-1",
        direction="SHORT", confidence=0.60,
    ))
    assert len(store.all()) == 2


def test_defaults():
    record = PerformanceRecord(
        run_id="run-1", agent_id="tech-1", snapshot_id="snap-1",
        direction="LONG", confidence=0.5,
    )
    store = _make_store()
    result = store.add(record)
    assert result.outcome is None
    assert result.realized_pnl == 0.0
    assert result.mfe == 0.0
    assert result.mae == 0.0


def test_upsert_overwrites():
    store = _make_store()
    record1 = PerformanceRecord(
        run_id="run-1", agent_id="tech-1", snapshot_id="snap-1",
        direction="LONG", confidence=0.75,
        realized_pnl=100.0,
    )
    store.add(record1)
    # Read back to get the ID for the update
    stored = store.get(record1.id)
    assert stored is not None

    record2 = PerformanceRecord(
        id=stored.id,
        run_id="run-1", agent_id="tech-1", snapshot_id="snap-1",
        direction="SHORT", confidence=0.90,
        realized_pnl=200.0,
    )
    store.add(record2)
    updated = store.get(record2.id)
    assert updated is not None
    assert updated.direction == "SHORT"
    assert updated.confidence == 0.90
    assert updated.realized_pnl == 200.0
    # Total count should still be 1
    assert len(store.all()) == 1