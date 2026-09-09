"""Tests for the live health & readiness router."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from apps.api.routers import live_health
from fastapi import FastAPI
from fastapi.testclient import TestClient
from packages.rollout import (
    KillSwitchState,
    get_rollout_controller,
    reset_rollout_controller,
)


@pytest.fixture(autouse=True)
def _reset_rollout() -> Generator[None, None, None]:
    reset_rollout_controller()
    yield
    reset_rollout_controller()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(live_health.router)
    return TestClient(app)


@pytest.fixture
def live_flag_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "packages.governance.feature_flags.FeatureFlags.is_enabled",
        lambda self, flag, environment=None: True,
    )


@pytest.fixture
def live_flag_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "packages.governance.feature_flags.FeatureFlags.is_enabled",
        lambda self, flag, environment=None: False,
    )


def _promote_to_live() -> None:
    get_rollout_controller().force_promote("LIVE_SMALL")


class TestHealthy:
    def test_healthy_when_flag_enabled_phase_live_kill_switch_off_circuit_closed(
        self, client: TestClient, live_flag_enabled: None
    ) -> None:
        _promote_to_live()
        response = client.get("/v1/health/live")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["liveness"] is True
        assert body["readiness"] is True
        assert body["details"]["feature_flag"] is True
        assert body["details"]["rollout_phase"] == "LIVE_SMALL"
        assert body["details"]["kill_switch"] == KillSwitchState.DISABLED
        assert body["details"]["circuit_breaker"] == "closed"


class TestDegraded:
    def test_degraded_when_flag_disabled(
        self, client: TestClient, live_flag_disabled: None
    ) -> None:
        _promote_to_live()
        body = client.get("/v1/health/live").json()
        assert body["status"] == "degraded"
        assert body["liveness"] is True
        assert body["readiness"] is False

    def test_degraded_when_phase_not_live(
        self, client: TestClient, live_flag_enabled: None
    ) -> None:
        # controller starts in SHADOW
        body = client.get("/v1/health/live").json()
        assert body["status"] == "degraded"
        assert body["details"]["rollout_phase"] == "SHADOW"

    def test_degraded_when_kill_switch_activated(
        self, client: TestClient, live_flag_enabled: None
    ) -> None:
        _promote_to_live()
        get_rollout_controller().force_kill("test halt")
        body = client.get("/v1/health/live").json()
        assert body["status"] == "degraded"
        assert body["readiness"] is False
        assert body["details"]["kill_switch"] == KillSwitchState.ACTIVATED

    def test_degraded_when_circuit_open(
        self, client: TestClient, live_flag_enabled: None
    ) -> None:
        _promote_to_live()
        get_rollout_controller().circuit_breaker.force_open("test")
        body = client.get("/v1/health/live").json()
        assert body["status"] == "degraded"
        assert body["readiness"] is False
        assert body["details"]["circuit_breaker"] == "open"


class TestResponseShape:
    def test_response_fields_stable(self, client: TestClient, live_flag_enabled: None) -> None:
        _promote_to_live()
        body = client.get("/v1/health/live").json()
        assert set(body.keys()) == {"status", "liveness", "readiness", "details"}
        assert set(body["details"].keys()) == {
            "feature_flag",
            "rollout_phase",
            "kill_switch",
            "circuit_breaker",
            "timestamp",
        }
        assert isinstance(body["details"]["timestamp"], str)
