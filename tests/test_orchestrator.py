import pytest

from trading_harness.models import RunState, transition


def test_valid_transition():
    assert transition(RunState.CREATED, RunState.DATA_READY) == RunState.DATA_READY


def test_invalid_transition():
    with pytest.raises(ValueError):
        transition(RunState.CREATED, RunState.CONSENSUS)


def test_complete_requires_decisions_path():
    assert transition(RunState.DECISION, RunState.COMPLETE) == RunState.COMPLETE


def test_failed_state_is_terminal():
    with pytest.raises(ValueError):
        transition(RunState.FAILED, RunState.COMPLETE)
