"""Integration tests: IP whitelist + rate limit middlewares on live prefixes,
plus RBAC dependency, wired together like in apps/api/main.py.

No real network, exchange, sleep, or database. All clocks are injected.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from packages.security import Permission
from packages.security.hardening.api_rate_limiter import (
    LiveRateLimiter,
    create_live_rate_limit_middleware,
)
from packages.security.hardening.audit_live import (
    get_live_audit,
    reset_live_audit,
)
from packages.security.hardening.ip_whitelist import (
    IPWhitelist,
    create_live_ip_middleware,
)
from packages.security.hardening.rbac_live import require_live_permission


@pytest.fixture(autouse=True)
def _clean_audit() -> None:
    reset_live_audit()


class FakeClock:
    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        return self._t


@pytest.fixture
def live_flag_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "packages.governance.feature_flags.FeatureFlags.is_enabled",
        lambda self, flag, environment=None: True,
    )


def build_app(limiter: LiveRateLimiter | None, whitelist: IPWhitelist, *, strict: bool) -> FastAPI:
    app = FastAPI()
    app.middleware("http")(create_live_ip_middleware(whitelist, strict=strict))
    app.middleware("http")(create_live_rate_limit_middleware(limiter))

    @app.get("/v1/live/ping",
             dependencies=[Depends(require_live_permission(Permission.READ_METRICS))])
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/v1/live/orders",
              dependencies=[Depends(require_live_permission(Permission.EXECUTE_LIVE))])
    async def orders() -> dict[str, bool]:
        return {"created": True}

    @app.get("/v1/health/live")
    async def health() -> dict[str, bool]:
        return {"liveness": True, "readiness": True}

    @app.get("/status")
    async def status() -> dict[str, bool]:
        return {"ok": True}

    return app


class TestStackedMiddleware:
    def test_ip_denied_audits_and_403(
        self, monkeypatch: pytest.MonkeyPatch, live_flag_enabled: None
    ) -> None:
        monkeypatch.setenv("API_TRUST_PROXY_HEADERS", "true")
        app = build_app(None, IPWhitelist(["10.0.0.0/8"]), strict=False)
        client = TestClient(app)

        r = client.get(
            "/v1/live/ping",
            headers={"X-Forwarded-For": "9.9.9.9", "X-Security-Role": "viewer"},
        )
        assert r.status_code == 403
        assert r.json() == {"error": "forbidden"}
        trail = get_live_audit()
        assert len(trail) == 1
        assert trail.entries[0].action == "ip_denied"
        assert trail.entries[0].details["ip"] == "9.9.9.9"

    def test_good_ip_and_role_pass(
        self, monkeypatch: pytest.MonkeyPatch, live_flag_enabled: None
    ) -> None:
        monkeypatch.setenv("API_TRUST_PROXY_HEADERS", "true")
        app = build_app(None, IPWhitelist(["10.0.0.0/8"]), strict=False)
        client = TestClient(app)
        r = client.get(
            "/v1/live/ping",
            headers={"X-Forwarded-For": "10.0.0.7", "X-Security-Role": "viewer"},
        )
        assert r.status_code == 200

    def test_ip_ok_but_role_missing_403(
        self, monkeypatch: pytest.MonkeyPatch, live_flag_enabled: None
    ) -> None:
        monkeypatch.setenv("API_TRUST_PROXY_HEADERS", "true")
        app = build_app(None, IPWhitelist(["10.0.0.0/8"]), strict=False)
        client = TestClient(app)
        r = client.get(
            "/v1/live/ping", headers={"X-Forwarded-For": "10.0.0.7"}
        )
        assert r.status_code == 403  # RBAC fail closed

    def test_health_path_under_ip_scope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("API_TRUST_PROXY_HEADERS", "true")
        app = build_app(None, IPWhitelist(["10.0.0.0/8"]), strict=False)
        client = TestClient(app)
        bad = client.get("/v1/health/live", headers={"X-Forwarded-For": "8.8.8.8"})
        assert bad.status_code == 403
        good = client.get("/v1/health/live", headers={"X-Forwarded-For": "10.0.0.9"})
        assert good.status_code == 200

    def test_non_live_path_bypasses_everything(self, live_flag_enabled: None) -> None:
        app = build_app(None, IPWhitelist(["10.0.0.0/8"]), strict=True)
        client = TestClient(app)
        assert client.get("/status").status_code == 200

    def test_rate_limit_429_with_headers(
        self, monkeypatch: pytest.MonkeyPatch, live_flag_enabled: None
    ) -> None:
        monkeypatch.setenv("API_TRUST_PROXY_HEADERS", "true")
        limiter = LiveRateLimiter(
            request_limit=3, order_limit=5, window_seconds=60, clock=FakeClock()
        )
        app = build_app(limiter, IPWhitelist(["10.0.0.0/8"]), strict=False)
        client = TestClient(app)
        headers = {"X-Forwarded-For": "10.0.0.1", "X-Security-Role": "viewer"}
        for _ in range(3):
            assert client.get("/v1/live/ping", headers=headers).status_code == 200
        r = client.get("/v1/live/ping", headers=headers)
        assert r.status_code == 429
        assert r.json() == {"error": "rate limited"}
        assert r.headers["X-RateLimit-Limit"] == "3"
        assert r.headers["X-RateLimit-Remaining"] == "0"

    def test_order_limit_stricter_than_request(
        self, monkeypatch: pytest.MonkeyPatch, live_flag_enabled: None
    ) -> None:
        """Order scope (2) exhausts before request scope (5) on POST /orders."""
        monkeypatch.setenv("API_TRUST_PROXY_HEADERS", "true")
        limiter = LiveRateLimiter(
            request_limit=5, order_limit=2, window_seconds=60, clock=FakeClock()
        )
        app = build_app(limiter, IPWhitelist(["10.0.0.0/8"]), strict=False)
        client = TestClient(app)
        headers = {"X-Forwarded-For": "10.0.0.1", "X-Security-Role": "live_operator"}
        assert client.post("/v1/live/orders", headers=headers).status_code == 200
        assert client.post("/v1/live/orders", headers=headers).status_code == 200
        r = client.post("/v1/live/orders", headers=headers)
        assert r.status_code == 429
        assert r.headers["X-RateLimit-Limit"] == "2"
        # Request scope still has room → GET passes.
        get_headers = dict(headers, **{"X-Security-Role": "viewer"})
        assert client.get("/v1/live/ping", headers=get_headers).status_code == 200
