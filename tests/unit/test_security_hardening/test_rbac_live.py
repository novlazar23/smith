"""Tests for RBAC role parsing and the live permission dependency (EPIC-16 WP06)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from packages.security import ROLE_PERMISSIONS, Permission, Role
from packages.security.hardening.rbac_live import (
    get_role_from_request,
    parse_role,
    require_live_permission,
)


class FakeState:
    def __init__(self, role: Any = None) -> None:
        self.role = role


class FakeRequest:
    def __init__(self, role: Any = None, header: str | None = None) -> None:
        self.state = FakeState(role)
        self.headers = {}
        if header is not None:
            self.headers["X-Security-Role"] = header


class TestParseRole:
    def test_valid_header(self) -> None:
        assert parse_role("live_operator") is Role.LIVE_OPERATOR

    def test_stripped(self) -> None:
        assert parse_role("  risk_manager  ") is Role.RISK_MANAGER

    def test_unknown_returns_none(self) -> None:
        assert parse_role("hacker") is None

    def test_empty_returns_none(self) -> None:
        assert parse_role("") is None

    def test_role_instance_passthrough(self) -> None:
        assert parse_role(Role.ADMINISTRATOR) is Role.ADMINISTRATOR


class TestGetRoleFromRequest:
    def test_state_role_wins_over_header(self) -> None:
        req = FakeRequest(role=Role.ADMINISTRATOR, header="live_operator")
        assert get_role_from_request(req) is Role.ADMINISTRATOR

    def test_header_used_when_state_absent(self) -> None:
        req = FakeRequest(header="risk_manager")
        assert get_role_from_request(req) is Role.RISK_MANAGER

    def test_missing_both_none(self) -> None:
        assert get_role_from_request(FakeRequest()) is None

    def test_invalid_header_none(self) -> None:
        assert get_role_from_request(FakeRequest(header="nope")) is None


@pytest.fixture
def live_flag_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "packages.governance.feature_flags.FeatureFlags.is_enabled",
        lambda self, flag, environment=None: True,
    )


@pytest.fixture
def app() -> FastAPI:
    """App exposing two live endpoints with different permission requirements."""
    app = FastAPI()

    @app.get("/execute")
    async def execute(
        _role: Role = Depends(require_live_permission(Permission.EXECUTE_LIVE)),  # noqa: B008
    ) -> dict[str, str]:
        return {"role": _role.value}

    @app.get("/kill")
    async def kill(
        _role: Role = Depends(require_live_permission(Permission.MANAGE_KILL_SWITCH)),  # noqa: B008
    ) -> dict[str, str]:
        return {"role": _role.value}

    return app


class TestDependencyIntegration:
    def test_correct_role_allowed(self, app: FastAPI, live_flag_enabled: None) -> None:
        client = TestClient(app)
        resp = client.get("/execute", headers={"X-Security-Role": "live_operator"})
        assert resp.status_code == 200
        assert resp.json() == {"role": "live_operator"}

    def test_missing_role_403(self, app: FastAPI, live_flag_enabled: None) -> None:
        client = TestClient(app)
        assert client.get("/execute").status_code == 403

    def test_unknown_role_403(self, app: FastAPI, live_flag_enabled: None) -> None:
        client = TestClient(app)
        resp = client.get("/execute", headers={"X-Security-Role": "hacker"})
        assert resp.status_code == 403

    def test_insufficient_role_403(self, app: FastAPI, live_flag_enabled: None) -> None:
        """risk_manager can manage kill switch but not execute."""
        client = TestClient(app)
        assert client.get(
            "/execute", headers={"X-Security-Role": "risk_manager"}
        ).status_code == 403
        assert client.get(
            "/kill", headers={"X-Security-Role": "risk_manager"}
        ).status_code == 200

    def test_flag_disabled_403(self, app: FastAPI) -> None:
        """Feature flag off → even correct role is rejected (existing gate)."""
        client = TestClient(app)
        resp = client.get("/execute", headers={"X-Security-Role": "live_operator"})
        assert resp.status_code == 403


class TestPermissionMatrix:
    def test_only_live_operator_can_execute(self) -> None:
        roles_with_execute = {
            role
            for role, perms in ROLE_PERMISSIONS.items()
            if Permission.EXECUTE_LIVE in perms
        }
        assert roles_with_execute == {Role.LIVE_OPERATOR}

    def test_live_operator(self) -> None:
        ctx = _ctx(Role.LIVE_OPERATOR)
        assert ctx.has_permission(Permission.EXECUTE_LIVE)
        assert ctx.has_permission(Permission.CANCEL_ORDERS)
        assert ctx.has_permission(Permission.VIEW_LIVE_PNL)
        assert ctx.has_permission(Permission.READ_METRICS)
        assert not ctx.has_permission(Permission.MANAGE_KILL_SWITCH)

    def test_risk_manager(self) -> None:
        ctx = _ctx(Role.RISK_MANAGER)
        assert ctx.has_permission(Permission.MANAGE_KILL_SWITCH)
        assert ctx.has_permission(Permission.VIEW_LIVE_PNL)
        assert not ctx.has_permission(Permission.EXECUTE_LIVE)

    def test_administrator(self) -> None:
        ctx = _ctx(Role.ADMINISTRATOR)
        assert ctx.has_permission(Permission.MANAGE_KILL_SWITCH)
        assert ctx.has_permission(Permission.VIEW_LIVE_PNL)
        assert ctx.has_permission(Permission.CANCEL_ORDERS)
        # Administrator intentionally lacks EXECUTE_LIVE (least privilege).
        assert not ctx.has_permission(Permission.EXECUTE_LIVE)


def _ctx(role: Role) -> Any:
    from packages.security import SecurityContext

    return SecurityContext(role=role)
