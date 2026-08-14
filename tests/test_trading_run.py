from trading_harness.models import RunOutcome, RunState
from trading_harness.services.orchestrator import TradingRunService


def _make_run(snapshot_id: str = "snap-1") -> TradingRunService:
    svc = TradingRunService()
    return svc.create(snapshot_id)


def test_create_run():
    svc = TradingRunService()
    run = svc.create("snap-1")
    assert run.snapshot_id == "snap-1"
    assert run.state == RunState.CREATED
    assert run.decisions == []
    assert run.outcome is None


def test_run_lifecycle():
    svc = TradingRunService()
    run = svc.create("snap-1")

    run = svc.transition(run.id, RunState.DATA_READY)
    assert run.state == RunState.DATA_READY

    run = svc.transition(run.id, RunState.ANALYSIS_RUNNING)
    assert run.state == RunState.ANALYSIS_RUNNING

    run = svc.transition(run.id, RunState.ADVERSARIAL_REVIEW)
    assert run.state == RunState.ADVERSARIAL_REVIEW

    run = svc.transition(run.id, RunState.CONSENSUS)
    assert run.state == RunState.CONSENSUS

    run = svc.transition(run.id, RunState.RISK_REVIEW)
    assert run.state == RunState.RISK_REVIEW

    run = svc.transition(run.id, RunState.DECISION)
    assert run.state == RunState.DECISION

    run = svc.complete(run.id, RunOutcome.NO_TRADE, "insufficient_consensus")
    assert run.state == RunState.COMPLETE
    assert run.outcome == RunOutcome.NO_TRADE


def test_failed_state_is_terminal():
    svc = TradingRunService()
    run = svc.create("snap-1")
    run = svc.fail(run.id, "snapshot expired")
    assert run.state == RunState.FAILED
    assert run.error == "snapshot expired"

    with svc._lock:
        audit = svc.get_audit_log(run.id)
    assert len(audit) == 2  # create + fail
    assert audit[1].action == "fail"


def test_add_decision():
    svc = TradingRunService()
    run = svc.create("snap-1")
    decision = {"agent_id": "tech-1", "direction": "LONG", "confidence": 0.75}
    run = svc.add_decision(run.id, decision)
    assert len(run.decisions) == 1
    assert run.decisions[0] == decision


def test_get_returns_none_for_missing_run():
    svc = TradingRunService()
    assert svc.get("nonexistent") is None


def test_transition_unknown_run_raises():
    svc = TradingRunService()
    try:
        svc.transition("unknown", RunState.DATA_READY)
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError")


def test_complete_unknown_run_raises():
    svc = TradingRunService()
    try:
        svc.complete("unknown", RunOutcome.LONG, "test")
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError")


def test_list_returns_all_runs():
    svc = TradingRunService()
    r1 = svc.create("snap-1")
    r2 = svc.create("snap-2")
    runs = svc.all()
    assert len(runs) == 2
    assert {r.id for r in runs} == {r1.id, r2.id}


def test_audit_log_captures_transitions():
    svc = TradingRunService()
    run = svc.create("snap-1")
    svc.transition(run.id, RunState.DATA_READY, actor="test-actor")

    log = svc.get_audit_log(run.id)
    assert len(log) == 2  # create + transition
    assert log[1].action == "transition"
    assert log[1].actor == "test-actor"
    assert log[1].previous_state == "CREATED"
    assert log[1].new_state == "DATA_READY"


def test_full_long_outcome():
    svc = TradingRunService()
    run = svc.create("snap-1")
    run = svc.transition(run.id, RunState.DATA_READY)
    run = svc.transition(run.id, RunState.ANALYSIS_RUNNING)
    run = svc.transition(run.id, RunState.ADVERSARIAL_REVIEW)
    run = svc.transition(run.id, RunState.CONSENSUS)
    run = svc.transition(run.id, RunState.RISK_REVIEW)
    svc.add_decision(run.id, {"direction": "LONG"})
    run = svc.transition(run.id, RunState.DECISION)
    run = svc.complete(run.id, RunOutcome.LONG, "consensus_with_risk_pass")

    assert run.outcome == RunOutcome.LONG
    assert run.outcome_reason == "consensus_with_risk_pass"