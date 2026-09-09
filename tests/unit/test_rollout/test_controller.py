"""Tests for packages.rollout.controller — PhasedRolloutController decisions and state."""

from __future__ import annotations

from typing import Any

import pytest
from packages.rollout import CircuitState, PhasedRolloutController, RolloutPhase


class FakeClock:
    """Mutable fake for packages.rollout.controller._now."""

    def __init__(self, value: float = 1000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("packages.rollout.controller._now", fake)
    return fake


@pytest.fixture
def ctrl(clock: FakeClock) -> PhasedRolloutController:
    controller = PhasedRolloutController()
    controller.start()
    return controller


def healthy(**overrides: Any) -> dict[str, Any]:
    """Promotion-grade evaluation metrics; individual gates can be overridden."""
    metrics: dict[str, Any] = {
        "brier_score": 0.22,
        "drawdown_pct": 0.01,
        "spread_ratio": 1.1,
        "exchange_error_rate": 0.02,
        "positive_trend": True,
    }
    metrics.update(overrides)
    return metrics


def promote_to_paper(ctrl: PhasedRolloutController, clock: FakeClock) -> None:
    clock.advance(601.0)
    decision = ctrl.evaluate(**healthy())
    assert decision.action == "promote"
    assert ctrl.current_phase == "PAPER"


class TestRolloutPhase:
    def test_levels_are_ordered(self) -> None:
        assert [p.level for p in RolloutPhase] == [0, 1, 2, 3, 4]

    def test_is_live_flags(self) -> None:
        assert RolloutPhase.SHADOW.is_live is False
        assert RolloutPhase.PAPER.is_live is False
        assert RolloutPhase.LIVE_SMALL.is_live is True
        assert RolloutPhase.LIVE_MEDIUM.is_live is True
        assert RolloutPhase.LIVE_FULL.is_live is True

    def test_next_phase_walk(self) -> None:
        assert RolloutPhase.SHADOW.next_phase is RolloutPhase.PAPER
        assert RolloutPhase.PAPER.next_phase is RolloutPhase.LIVE_SMALL
        assert RolloutPhase.LIVE_SMALL.next_phase is RolloutPhase.LIVE_MEDIUM
        assert RolloutPhase.LIVE_MEDIUM.next_phase is RolloutPhase.LIVE_FULL
        assert RolloutPhase.LIVE_FULL.next_phase is None

    def test_prev_phase_walk(self) -> None:
        assert RolloutPhase.SHADOW.prev_phase is None
        assert RolloutPhase.PAPER.prev_phase is RolloutPhase.SHADOW
        assert RolloutPhase.LIVE_SMALL.prev_phase is RolloutPhase.PAPER
        assert RolloutPhase.LIVE_FULL.prev_phase is RolloutPhase.LIVE_MEDIUM


class TestLifecycle:
    def test_initial_phase_is_shadow(self) -> None:
        assert PhasedRolloutController().current_phase == "SHADOW"

    def test_start_sets_timestamps_in_state(self, ctrl: PhasedRolloutController) -> None:
        s = ctrl.status()["controller"]
        assert s["current_phase"] == "SHADOW"
        assert s["started_at"] == 1000.0
        assert s["phase_started_at"] == 1000.0

    def test_status_snapshot_shape(self, ctrl: PhasedRolloutController) -> None:
        s = ctrl.status()
        assert set(s) == {"controller", "kill_switch", "circuit_breaker", "thresholds"}
        assert s["kill_switch"]["state"] == "disabled"
        assert s["circuit_breaker"]["state"] == "closed"
        assert s["thresholds"]["capital_ramp_pct"] == [0.01, 0.05, 0.25, 0.5, 1.0]

    def test_stop_deactivates_safety_mechanisms(self, ctrl: PhasedRolloutController) -> None:
        ctrl.force_kill("test")
        ctrl.circuit_breaker.force_open("test")
        ctrl.stop()
        assert ctrl.kill_switch.state == "disabled"
        assert ctrl.circuit_breaker.state == CircuitState.CLOSED


class TestPromotion:
    def test_promotion_reports_from_and_to_phase(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        clock.advance(601.0)
        d = ctrl.evaluate(**healthy())
        assert d.action == "promote"
        assert d.from_phase == "SHADOW"
        assert d.to_phase == "PAPER"
        assert ctrl.current_phase == "PAPER"

    def test_no_promotion_before_min_duration(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        clock.advance(599.0)
        d = ctrl.evaluate(**healthy())
        assert d.action == "hold"
        assert ctrl.current_phase == "SHADOW"

    def test_promotion_requires_positive_trend(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        clock.advance(601.0)
        d = ctrl.evaluate(**healthy(positive_trend=False))
        assert d.action == "hold"
        assert ctrl.current_phase == "SHADOW"

    def test_promotion_requires_brier_gate(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        clock.advance(601.0)
        d = ctrl.evaluate(**healthy(brier_score=0.40))
        assert d.action == "hold"
        assert ctrl.current_phase == "SHADOW"

    def test_promotion_criteria_gates(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        clock.advance(601.0)
        base = healthy()

        def criteria(**overrides: Any) -> bool:
            return ctrl._meets_promotion_criteria(
                current_phase=RolloutPhase.SHADOW, **{**base, **overrides}
            )

        assert criteria() is True
        assert criteria(positive_trend=False) is False
        assert criteria(brier_score=0.30) is True
        assert criteria(brier_score=0.31) is False
        assert criteria(drawdown_pct=0.05) is False
        assert criteria(spread_ratio=2.5) is False
        assert criteria(exchange_error_rate=0.10) is False

    def test_full_promotion_ladder_to_live_full(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        transitions: list[tuple[str, str]] = []
        for target in ("PAPER", "LIVE_SMALL", "LIVE_MEDIUM", "LIVE_FULL"):
            clock.advance(601.0)
            d = ctrl.evaluate(**healthy())
            assert d.action == "promote"
            assert d.to_phase == target
            transitions.append((d.from_phase, d.to_phase))
        assert ctrl.current_phase == "LIVE_FULL"
        assert transitions[0] == ("SHADOW", "PAPER")
        assert transitions[-1] == ("LIVE_MEDIUM", "LIVE_FULL")
        assert ctrl.status()["controller"]["promotions"] == 4

    def test_no_promotion_at_live_full(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        for _ in range(4):
            clock.advance(601.0)
            ctrl.evaluate(**healthy())
        assert ctrl.current_phase == "LIVE_FULL"
        clock.advance(601.0)
        d = ctrl.evaluate(**healthy())
        assert d.action == "hold"
        assert ctrl.current_phase == "LIVE_FULL"


class TestDemotion:
    def test_demotion_on_brier_regression(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        promote_to_paper(ctrl, clock)
        d = ctrl.evaluate(**healthy(brier_score=0.50))
        assert d.action == "demote"
        assert d.from_phase == "PAPER"
        assert d.to_phase == "SHADOW"
        assert ctrl.current_phase == "SHADOW"

    def test_no_demotion_when_brier_is_good(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        promote_to_paper(ctrl, clock)
        d = ctrl.evaluate(**healthy(brier_score=0.20))
        assert d.action == "hold"
        assert ctrl.current_phase == "PAPER"

    def test_demotion_on_drawdown(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        ctrl._kill_switch.max_drawdown_pct = 0.99
        promote_to_paper(ctrl, clock)
        d = ctrl.evaluate(**healthy(drawdown_pct=0.06))
        assert d.action == "demote"
        assert d.from_phase == "PAPER"
        assert d.to_phase == "SHADOW"
        assert ctrl.current_phase == "SHADOW"

    def test_needs_demotion_direct_triggers(self, ctrl: PhasedRolloutController) -> None:
        base = healthy()
        base.pop("positive_trend")

        def needs(**overrides: Any) -> bool:
            return ctrl._needs_demotion(current_phase=RolloutPhase.PAPER, **{**base, **overrides})

        assert needs(spread_ratio=2.5) is True
        assert needs(exchange_error_rate=0.10) is True
        assert needs(brier_score=0.46) is True
        assert needs() is False

    def test_no_demotion_from_shadow(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        clock.advance(601.0)
        d = ctrl.evaluate(**healthy(brier_score=0.90))
        assert d.action == "hold"
        assert ctrl.current_phase == "SHADOW"

    def test_demotion_updates_phase_timestamps(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        promote_to_paper(ctrl, clock)
        clock.advance(100.0)
        ctrl.evaluate(**healthy(brier_score=0.50))
        s = ctrl.status()["controller"]
        assert s["phase_started_at"] == 1701.0
        assert s["started_at"] == 1000.0


class TestSafetyPriority:
    def test_kill_switch_priority_over_promotion(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        clock.advance(601.0)
        d = ctrl.evaluate(**healthy(drawdown_pct=0.06))
        assert d.action == "kill"
        assert d.from_phase == "SHADOW"
        assert d.to_phase == "SHADOW"
        assert ctrl.current_phase == "SHADOW"
        assert ctrl.kill_switch.state == "activated"

    def test_activated_kill_switch_blocks_promotion(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        ctrl.force_kill("operator")
        clock.advance(601.0)
        d = ctrl.evaluate(**healthy())
        assert d.action == "kill"
        assert d.reason == "operator"
        assert ctrl.current_phase == "SHADOW"

    def test_circuit_breaker_open_leads_to_kill(self, ctrl: PhasedRolloutController) -> None:
        ctrl.circuit_breaker.force_open("manual")
        d = ctrl.evaluate(**healthy())
        assert d.action == "kill"
        assert "circuit breaker" in d.reason
        assert ctrl.current_phase == "SHADOW"

    def test_circuit_reset_allows_recovery(self, ctrl: PhasedRolloutController) -> None:
        ctrl.circuit_breaker.force_open("manual")
        ctrl.evaluate(**healthy())
        ctrl.reset_circuit_breaker()
        d = ctrl.evaluate(**healthy())
        assert d.action == "hold"


class TestForceOperations:
    def test_force_promote_multi_step(self, ctrl: PhasedRolloutController) -> None:
        d = ctrl.force_promote("LIVE_MEDIUM")
        assert d.action == "promote"
        assert d.from_phase == "SHADOW"
        assert d.to_phase == "LIVE_MEDIUM"
        assert ctrl.current_phase == "LIVE_MEDIUM"

    def test_force_promote_not_ahead_is_hold(self, ctrl: PhasedRolloutController) -> None:
        d = ctrl.force_promote("SHADOW")
        assert d.action == "hold"
        assert d.from_phase == "SHADOW"
        assert d.to_phase == "SHADOW"
        ctrl.force_promote("PAPER")
        assert ctrl.force_promote("PAPER").action == "hold"
        assert ctrl.current_phase == "PAPER"

    def test_force_demote(self, ctrl: PhasedRolloutController) -> None:
        ctrl.force_promote("LIVE_MEDIUM")
        d = ctrl.force_demote("PAPER")
        assert d.action == "demote"
        assert d.from_phase == "LIVE_MEDIUM"
        assert d.to_phase == "PAPER"
        assert ctrl.current_phase == "PAPER"

    def test_force_demote_not_behind_is_hold(self, ctrl: PhasedRolloutController) -> None:
        d = ctrl.force_demote("LIVE_FULL")
        assert d.action == "hold"
        assert ctrl.current_phase == "SHADOW"

    def test_force_kill_activates_switch_and_callback(self) -> None:
        cancelled: list[str] = []
        controller = PhasedRolloutController(on_cancel_orders=lambda: cancelled.append("x"))
        d = controller.force_kill("manual stop")
        assert d.action == "kill"
        assert d.reason == "manual kill: manual stop"
        assert controller.kill_switch.state == "activated"
        assert controller.kill_switch.reason == "manual stop"
        assert cancelled == ["x"]


class TestStateBookkeeping:
    def test_evaluation_cycles_and_last_decision(
        self, ctrl: PhasedRolloutController, clock: FakeClock
    ) -> None:
        clock.advance(601.0)
        d = ctrl.evaluate(**healthy())
        s = ctrl.status()["controller"]
        assert s["total_evaluation_cycles"] == 1
        assert s["last_decision"] == "promote"
        assert s["last_reason"] == d.reason
        assert s["promotions"] == 1
        assert s["demotions"] == 0

    def test_hold_records_reason(self, ctrl: PhasedRolloutController) -> None:
        d = ctrl.evaluate(**healthy())
        assert d.action == "hold"
        s = ctrl.status()["controller"]
        assert s["last_decision"] == "hold"
        assert s["last_reason"] == d.reason
