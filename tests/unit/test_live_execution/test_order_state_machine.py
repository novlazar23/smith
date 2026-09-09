"""Unit tests for packages.live_execution.order_state_machine.

Covers the 8-state lifecycle: initial state, valid/illegal transitions,
absorbing (terminal) semantics, fill handling, snapshots, and history.
"""

from __future__ import annotations

from typing import Any

import pytest
from packages.live_execution.order_state_machine import (
    OrderSnapshot,
    OrderState,
    OrderStateMachine,
    StateTransitionError,
)

TERMINAL_STATES = [
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.REJECTED,
    OrderState.EXPIRED,
    OrderState.ERROR,
]

ALL_STATES = list(OrderState)


def _machine(**overrides: Any) -> OrderStateMachine:
    kwargs = {
        "order_id": "ord-1",
        "symbol": "BTC/USDT",
        "venue": "binance",
        "side": "buy",
        "quantity": 1.0,
        "price": 45000.0,
    }
    kwargs.update(overrides)
    return OrderStateMachine(**kwargs)


class TestInitialState:
    def test_initial_state_is_new(self) -> None:
        assert _machine().state is OrderState.NEW

    def test_initial_snapshot_fields(self) -> None:
        snap = _machine().to_snapshot()
        assert isinstance(snap, OrderSnapshot)
        assert snap.order_id == "ord-1"
        assert snap.symbol == "BTC/USDT"
        assert snap.venue == "binance"
        assert snap.side == "buy"
        assert snap.quantity == 1.0
        assert snap.price == 45000.0
        assert snap.filled_quantity == 0.0
        assert snap.state is OrderState.NEW

    def test_initially_not_terminal(self) -> None:
        assert _machine().is_terminal is False

    def test_history_empty_initially(self) -> None:
        assert _machine().get_history() == []


class TestValidTransitions:
    @pytest.mark.parametrize("target", [OrderState.PENDING])
    def test_new_to_pending(self, target: OrderState) -> None:
        sm = _machine()
        sm.transition_to(target, event="submitted")
        assert sm.state is target

    @pytest.mark.parametrize(
        "target",
        [
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.ERROR,
        ],
    )
    def test_pending_to_all_non_new(self, target: OrderState) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        sm.transition_to(target)
        assert sm.state is target

    @pytest.mark.parametrize(
        "target",
        [OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED, OrderState.ERROR],
    )
    def test_partially_filled_targets(self, target: OrderState) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        sm.transition_to(OrderState.PARTIALLY_FILLED)
        sm.transition_to(target)
        assert sm.state is target

    def test_is_transition_allowed_reflects_table(self) -> None:
        sm = _machine()
        assert sm.is_transition_allowed(OrderState.NEW, OrderState.PENDING)
        assert not sm.is_transition_allowed(OrderState.NEW, OrderState.FILLED)
        assert not sm.is_transition_allowed(OrderState.FILLED, OrderState.PENDING)


class TestIllegalTransitions:
    @pytest.mark.parametrize("target", ALL_STATES)
    def test_terminal_states_absorbing(self, target: OrderState) -> None:
        for terminal in TERMINAL_STATES:
            sm = _machine()
            sm.transition_to(OrderState.PENDING)
            sm.transition_to(terminal)
            with pytest.raises(StateTransitionError):
                sm.transition_to(target)
            assert sm.state is terminal

    def test_new_cannot_jump_to_filled(self) -> None:
        sm = _machine()
        with pytest.raises(StateTransitionError):
            sm.transition_to(OrderState.FILLED)
        assert sm.state is OrderState.NEW

    def test_rejected_target_rejected_from_pending(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        with pytest.raises(StateTransitionError):
            sm.transition_to(OrderState.NEW)

    def test_error_message_names_states(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        with pytest.raises(StateTransitionError) as exc_info:
            sm.transition_to(OrderState.NEW)
        assert "PENDING" in str(exc_info.value)
        assert exc_info.value.from_state is OrderState.PENDING
        assert exc_info.value.to_state is OrderState.NEW


class TestTransitionFromError:
    def test_from_pending_reaches_error(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        sm.transition_from_error(event="boom")
        assert sm.state is OrderState.ERROR

    @pytest.mark.parametrize("terminal", TERMINAL_STATES)
    def test_from_terminal_is_safe_noop(self, terminal: OrderState) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        sm.transition_to(terminal)
        history_before = sm.get_history()
        snap = sm.transition_from_error(event="late error")
        assert sm.state is terminal
        assert snap.state is terminal
        assert sm.get_history() == history_before


class TestHistoryAndSnapshot:
    def test_history_records_each_transition(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING, event="submitted")
        sm.transition_to(OrderState.FILLED, event="filled")
        history = sm.get_history()
        assert [(s, e) for s, e, _ in history] == [
            (OrderState.PENDING, "submitted"),
            (OrderState.FILLED, "filled"),
        ]

    def test_get_history_returns_copy(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        history = sm.get_history()
        history.clear()
        assert len(sm.get_history()) == 1

    def test_transition_metadata_merged_into_snapshot(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING, event="submitted", metadata={"k": "v"})
        assert sm.to_snapshot().metadata == {"k": "v"}

    def test_to_snapshot_does_not_mutate_state_changed_at(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        first = sm.to_snapshot()
        first_ts = first.state_changed_at
        second = sm.to_snapshot()
        assert second.state_changed_at == first_ts

    def test_transition_to_logs_previous_state_as_source(self, caplog: pytest.LogCaptureFixture) -> None:
        sm = _machine(order_id="log-ord")
        with caplog.at_level("INFO", logger="packages.live_execution.order_state_machine"):
            sm.transition_to(OrderState.PENDING, event="submitted")
        messages = [r.getMessage() for r in caplog.records]
        assert any("NEW → PENDING" in m and "log-ord" in m for m in messages)


class TestUpdateFill:
    def test_partial_fill_from_pending(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        snap = sm.update_fill(0.4, fill_price=45100.0)
        assert snap.state is OrderState.PARTIALLY_FILLED
        assert sm.filled_quantity == 0.4

    def test_full_fill_from_pending(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        snap = sm.update_fill(1.0, fill_price=45200.0)
        assert snap.state is OrderState.FILLED
        assert sm.filled_quantity == 1.0

    def test_overfill_reaches_filled(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        snap = sm.update_fill(1.5, fill_price=45200.0)
        assert snap.state is OrderState.FILLED

    def test_second_partial_fill_updates_in_place(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        sm.update_fill(0.3, fill_price=45000.0)
        snap = sm.update_fill(0.7, fill_price=45050.0)
        assert snap.state is OrderState.PARTIALLY_FILLED
        assert sm.filled_quantity == 0.7
        assert snap.metadata["fill_price"] == 45050.0

    @pytest.mark.parametrize("terminal", TERMINAL_STATES)
    def test_update_fill_from_terminal_is_safe_noop(self, terminal: OrderState) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        sm.transition_to(terminal)
        filled_before = sm.filled_quantity
        history_before = sm.get_history()
        snap = sm.update_fill(9.9, fill_price=1.0)
        assert sm.state is terminal
        assert sm.filled_quantity == filled_before
        assert snap.state is terminal
        assert sm.get_history() == history_before

    def test_update_fill_from_new_raises(self) -> None:
        sm = _machine()
        with pytest.raises(StateTransitionError):
            sm.update_fill(0.5, fill_price=45000.0)


class TestReset:
    def test_reset_to_new_clears_state(self) -> None:
        sm = _machine()
        sm.transition_to(OrderState.PENDING)
        sm.update_fill(1.0, fill_price=45000.0)
        sm.reset_to_new()
        assert sm.state is OrderState.NEW
        assert sm.filled_quantity == 0.0
        assert sm.get_history() == []
