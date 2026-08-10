import pytest

from trading_harness.services.orchestrator import RunState, transition


def test_valid_transition():
    assert transition(RunState.CREATED, RunState.DATA_READY) == RunState.DATA_READY


def test_invalid_transition():
    with pytest.raises(ValueError):
        transition(RunState.CREATED, RunState.CONSENSUS)
