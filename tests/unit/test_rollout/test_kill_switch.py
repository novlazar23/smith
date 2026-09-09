"""Tests for packages.rollout.kill_switch — manual/automatic triggers, idempotency, callbacks."""

from __future__ import annotations

import pytest
from packages.rollout import KillSwitch, KillSwitchState


class FakeClock:
    """Mutable fake for packages.rollout.kill_switch._monotonic_now."""

    def __init__(self, value: float = 1000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("packages.rollout.kill_switch._monotonic_now", fake)
    return fake


class TestInitialState:
    def test_initial_state_is_disabled(self) -> None:
        ks = KillSwitch()
        assert ks.state == KillSwitchState.DISABLED
        assert ks.reason == ""
        assert ks.activated_at == 0.0

    def test_status_snapshot_when_disabled(self) -> None:
        ks = KillSwitch()
        assert ks.status() == {"state": "disabled", "reason": "", "activated_at": 0.0}


class TestManualActivation:
    def test_activation_sets_state_reason_time(self, clock: FakeClock) -> None:
        ks = KillSwitch()
        ks.activate("operator stop")
        assert ks.state == KillSwitchState.ACTIVATED
        assert ks.reason == "operator stop"
        assert ks.activated_at == 1000.0

    def test_status_snapshot_when_activated(self, clock: FakeClock) -> None:
        ks = KillSwitch()
        ks.activate("dd breach")
        assert ks.status() == {
            "state": "activated",
            "reason": "dd breach",
            "activated_at": 1000.0,
        }

    def test_activation_is_idempotent(self, clock: FakeClock) -> None:
        ks = KillSwitch()
        ks.activate("first reason")
        clock.value = 2000.0
        ks.activate("second reason")
        assert ks.reason == "first reason"
        assert ks.activated_at == 1000.0


class TestAutomaticTriggers:
    def test_drawdown_at_limit_activates(self) -> None:
        ks = KillSwitch(max_drawdown_pct=0.05)
        assert ks.check_and_activate(current_drawdown_pct=0.05) is True
        assert ks.state == KillSwitchState.ACTIVATED
        assert "drawdown" in ks.reason

    def test_drawdown_below_limit_does_not_activate(self) -> None:
        ks = KillSwitch(max_drawdown_pct=0.05)
        assert ks.check_and_activate(current_drawdown_pct=0.04) is False
        assert ks.state == KillSwitchState.DISABLED

    def test_spread_anomaly_activates(self) -> None:
        ks = KillSwitch(max_spread_anomaly_ratio=2.5)
        assert ks.check_and_activate(current_spread_ratio=2.5) is True
        assert "spread" in ks.reason

    def test_error_rate_activates(self) -> None:
        ks = KillSwitch(max_exchange_error_rate=0.10)
        assert ks.check_and_activate(current_error_rate=0.10) is True
        assert "error rate" in ks.reason

    def test_drawdown_gate_evaluated_first(self) -> None:
        ks = KillSwitch()
        ks.check_and_activate(
            current_drawdown_pct=0.06,
            current_spread_ratio=9.0,
            current_error_rate=0.5,
        )
        assert ks.state == KillSwitchState.ACTIVATED
        assert "drawdown" in ks.reason

    def test_within_limits_returns_false(self) -> None:
        ks = KillSwitch()
        assert (
            ks.check_and_activate(
                current_drawdown_pct=0.01,
                current_spread_ratio=1.2,
                current_error_rate=0.02,
            )
            is False
        )
        assert ks.state == KillSwitchState.DISABLED


class TestKillPriorityAndDeactivation:
    def test_already_activated_check_returns_true(self) -> None:
        ks = KillSwitch()
        ks.activate("manual")
        assert ks.check_and_activate(current_drawdown_pct=0.0) is True
        assert ks.reason == "manual"

    def test_deactivate_resets_state(self) -> None:
        ks = KillSwitch()
        ks.activate("manual")
        ks.deactivate()
        assert ks.state == KillSwitchState.DISABLED
        assert ks.reason == ""
        assert ks.activated_at == 0.0

    def test_deactivate_when_disabled_is_noop(self) -> None:
        ks = KillSwitch()
        ks.deactivate()
        assert ks.state == KillSwitchState.DISABLED


class TestCancelCallback:
    def test_callable_callback_invoked_once(self) -> None:
        calls: list[str] = []
        ks = KillSwitch()
        ks.register_cancel_callback(lambda: calls.append("cancelled"))
        ks.activate("boom")
        assert calls == ["cancelled"]

    def test_object_callback_invoked(self) -> None:
        class OrderService:
            def __init__(self) -> None:
                self.cancelled = False

            def cancel_open_orders(self) -> None:
                self.cancelled = True

        svc = OrderService()
        ks = KillSwitch()
        ks.register_cancel_callback(svc)
        ks.activate("boom")
        assert svc.cancelled is True

    def test_callback_exception_does_not_break_activation(self) -> None:
        def broken() -> None:
            raise RuntimeError("exchange unreachable")

        ks = KillSwitch()
        ks.register_cancel_callback(broken)
        ks.activate("boom")
        assert ks.state == KillSwitchState.ACTIVATED
