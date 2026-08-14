from trading_harness.models import PerformanceRecord
from trading_harness.services.performance import PerformanceStore


def test_add_and_get():
    store = PerformanceStore()
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


def test_by_run():
    store = PerformanceStore()
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


def test_by_agent():
    store = PerformanceStore()
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


def test_by_snapshot():
    store = PerformanceStore()
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


def test_list_returns_all():
    store = PerformanceStore()
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
    store = PerformanceStore()
    result = store.add(record)
    assert result.outcome is None
    assert result.realized_pnl == 0.0
    assert result.mfe == 0.0
    assert result.mae == 0.0