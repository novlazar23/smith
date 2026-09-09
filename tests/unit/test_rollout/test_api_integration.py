"""Tests for rollout integration in apps/api routers — shared controller singleton."""

from __future__ import annotations

import pytest
from packages.governance.feature_flags import feature_flags
from packages.rollout import get_rollout_controller, reset_rollout_controller


@pytest.fixture(autouse=True)
def clean_shared_controller() -> None:
    reset_rollout_controller()
    yield
    reset_rollout_controller()


@pytest.fixture
def live_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(feature_flags, "is_enabled", lambda flag, environment=None: True)


class TestSharedControllerAccessor:
    def test_returns_same_instance(self) -> None:
        assert get_rollout_controller() is get_rollout_controller()

    def test_reset_creates_fresh_instance(self) -> None:
        first = get_rollout_controller()
        first.force_promote("PAPER")
        reset_rollout_controller()
        second = get_rollout_controller()
        assert second is not first
        assert second.current_phase == "SHADOW"

    def test_public_api_names_importable(self) -> None:
        import packages.rollout as rollout

        for name in rollout.__all__:
            assert hasattr(rollout, name), f"missing export: {name}"
        for name in (
            "PhasedRolloutController",
            "RolloutPhase",
            "RolloutDecision",
            "RolloutControllerState",
            "RolloutThresholds",
            "CircuitBreaker",
            "CircuitState",
            "KillSwitch",
            "KillSwitchState",
        ):
            assert name in rollout.__all__


class TestLiveHealthRouter:
    async def test_reports_shared_controller_phase(self, live_flag: None) -> None:
        from apps.api.routers import live_health

        get_rollout_controller().force_promote("LIVE_SMALL")
        resp = await live_health.live_health_check()
        assert resp.details["rollout_phase"] == "LIVE_SMALL"
        assert resp.details["kill_switch"] == "disabled"
        assert resp.details["circuit_breaker"] == "closed"
        assert resp.readiness is True
        assert resp.status == "healthy"

    async def test_shadow_phase_blocks_readiness(self, live_flag: None) -> None:
        from apps.api.routers import live_health

        resp = await live_health.live_health_check()
        assert resp.details["rollout_phase"] == "SHADOW"
        assert resp.readiness is False
        # alive but not ready → degraded (unhealthy is reserved for !liveness)
        assert resp.status == "degraded"


class TestKillSwitchEndpoint:
    @pytest.fixture(autouse=True)
    def clean_order_registry(self) -> None:
        from apps.api.routers import live_orders

        live_orders._order_registry.clear()
        live_orders._idempotency_index.clear()

    async def test_activate_uses_shared_controller(self, live_flag: None) -> None:
        from apps.api.routers import live_orders

        shared = get_rollout_controller()
        resp = await live_orders.kill_switch(
            live_orders.KillSwitchRequest(action="activate", reason="test stop")
        )
        assert resp.state == "activated"
        assert resp.confirmed is True
        assert shared.kill_switch.state == "activated"
        assert shared.kill_switch.reason == "test stop"

    async def test_deactivate_uses_shared_controller(self, live_flag: None) -> None:
        from apps.api.routers import live_orders

        get_rollout_controller().force_kill("earlier")
        resp = await live_orders.kill_switch(
            live_orders.KillSwitchRequest(action="deactivate", reason="all clear")
        )
        assert resp.state == "disabled"
        assert get_rollout_controller().kill_switch.state == "disabled"
