from trading_harness.models import EvaluationResult
from trading_harness.services.evaluation_result_store import (
    PersistedEvaluationResultStore,
)


def _make_store():
    """Create a PersistedEvaluationResultStore with no DB (uses in-memory fallback)."""
    return PersistedEvaluationResultStore(db=None)


def test_add_and_get():
    store = _make_store()
    result = EvaluationResult(
        run_id="run-1",
        agent_id="tech-1",
        metric_name="brier_score",
        metric_value=0.25,
        observations=100,
        details={"precision": 0.8, "recall": 0.75},
    )
    stored = store.add(result)
    assert stored.id == result.id
    assert store.get(result.id) is stored


def test_get_nonexistent_returns_none():
    store = _make_store()
    assert store.get("nonexistent") is None


def test_by_agent():
    store = _make_store()
    r1 = store.add(EvaluationResult(
        run_id="run-1", agent_id="tech-1",
        metric_name="brier_score", metric_value=0.25, observations=100,
    ))
    store.add(EvaluationResult(
        run_id="run-1", agent_id="tech-2",
        metric_name="brier_score", metric_value=0.30, observations=100,
    ))
    r3 = store.add(EvaluationResult(
        run_id="run-2", agent_id="tech-1",
        metric_name="calibration_error", metric_value=0.05, observations=50,
    ))
    by_agent = store.by_agent("tech-1")
    assert len(by_agent) == 2
    assert {r.id for r in by_agent} == {r1.id, r3.id}


def test_by_agent_empty():
    store = _make_store()
    assert store.by_agent("nonexistent") == []


def test_by_run():
    store = _make_store()
    store.add(EvaluationResult(
        run_id="run-1", agent_id="tech-1",
        metric_name="brier_score", metric_value=0.25, observations=100,
    ))
    store.add(EvaluationResult(
        run_id="run-2", agent_id="tech-2",
        metric_name="brier_score", metric_value=0.30, observations=100,
    ))
    by_run = store.by_run("run-1")
    assert len(by_run) == 1


def test_by_run_empty():
    store = _make_store()
    assert store.by_run("nonexistent") == []


def test_list_returns_all():
    store = _make_store()
    store.add(EvaluationResult(
        run_id="run-1", agent_id="tech-1",
        metric_name="brier_score", metric_value=0.25, observations=100,
    ))
    store.add(EvaluationResult(
        run_id="run-2", agent_id="tech-2",
        metric_name="calibration_error", metric_value=0.05, observations=50,
    ))
    assert len(store.all()) == 2


def test_defaults():
    result = EvaluationResult(
        run_id="run-1", agent_id="tech-1",
        metric_name="brier_score", metric_value=0.25, observations=100,
    )
    store = _make_store()
    stored = store.add(result)
    assert stored.details == {}