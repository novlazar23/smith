"""Tests für den Live-Flag-Router (GET/POST /v1/live/flag)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from packages.governance.feature_flags import FeatureFlags


@pytest.fixture(autouse=True)
def _isolated_flags(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isoliert das Flag-Singleton: Env sauber, Instanz pro Test neu bindbar."""
    original = FeatureFlags._instance
    for key in ("APP_ENV", "ENV", "LIVE_TRADING_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    FeatureFlags._instance = None
    yield
    FeatureFlags._instance = original


def _rebind_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neues Singleton (Env vorher setzen!) + Router-Referenz neu binden."""
    FeatureFlags._instance = None
    fresh = FeatureFlags()
    monkeypatch.setattr("packages.governance.feature_flags.feature_flags", fresh)


@pytest.fixture
def client() -> TestClient:
    from apps.api.routers import live_flags

    app = FastAPI()
    app.include_router(live_flags.router)
    return TestClient(app)


class TestGetLiveFlag:
    def test_returns_state_shape(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _rebind_fresh(monkeypatch)
        response = client.get("/v1/live/flag")
        assert response.status_code == 200
        body = response.json()
        assert body["flag"] == "live_trading_enabled"
        assert body["enabled"] is False
        assert body["environment"] == "development"
        assert body["credentials_configured"] == {}


class TestSetLiveFlag:
    def test_admin_enables_in_production(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        _rebind_fresh(monkeypatch)
        response = client.post(
            "/v1/live/flag",
            json={"enabled": True},
            headers={"X-Security-Role": "administrator"},
        )
        assert response.status_code == 200
        assert response.json()["enabled"] is True

    def test_admin_toggle_is_locked_in_development(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Safety-Lock: selbst ein Admin kann in development nichts aktivieren.
        _rebind_fresh(monkeypatch)
        response = client.post(
            "/v1/live/flag",
            json={"enabled": True},
            headers={"X-Security-Role": "administrator"},
        )
        assert response.status_code == 200
        assert response.json()["enabled"] is False

    @pytest.mark.parametrize(
        "role",
        ["viewer", "operator", "live_operator", "researcher"],
    )
    def test_role_without_manage_kill_switch_is_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, role: str
    ) -> None:
        _rebind_fresh(monkeypatch)
        response = client.post(
            "/v1/live/flag",
            json={"enabled": True},
            headers={"X-Security-Role": role},
        )
        assert response.status_code == 403

    def test_risk_manager_may_toggle(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        _rebind_fresh(monkeypatch)
        response = client.post(
            "/v1/live/flag",
            json={"enabled": True},
            headers={"X-Security-Role": "risk_manager"},
        )
        assert response.status_code == 200
        assert response.json()["enabled"] is True

    def test_missing_role_is_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _rebind_fresh(monkeypatch)
        response = client.post("/v1/live/flag", json={"enabled": True})
        assert response.status_code == 403

    def test_unknown_role_is_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _rebind_fresh(monkeypatch)
        response = client.post(
            "/v1/live/flag",
            json={"enabled": True},
            headers={"X-Security-Role": "mallory"},
        )
        assert response.status_code == 403

    def test_invalid_body_is_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _rebind_fresh(monkeypatch)
        response = client.post(
            "/v1/live/flag",
            json={"enabled": "maybe"},
            headers={"X-Security-Role": "administrator"},
        )
        assert response.status_code == 422
