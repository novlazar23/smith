"""Tests for the live IP whitelist and its middleware integration."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from packages.security.hardening.audit_live import reset_live_audit
from packages.security.hardening.ip_whitelist import (
    IPWhitelist,
    create_live_ip_middleware,
)


@pytest.fixture(autouse=True)
def _clean_audit() -> None:
    reset_live_audit()


def _app_with(ip_mw: Callable[..., object] | None = None) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/live/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/status")
    async def status() -> dict[str, bool]:
        return {"ok": True}

    if ip_mw is not None:
        app.middleware("http")(ip_mw)
    return app


class TestIPWhitelist:
    def test_single_ip(self) -> None:
        wl = IPWhitelist(["10.0.0.5"])
        assert wl.is_allowed("10.0.0.5")
        assert not wl.is_allowed("10.0.0.6")

    def test_cidr(self) -> None:
        wl = IPWhitelist(["192.168.1.0/24"])
        assert wl.is_allowed("192.168.1.1")
        assert wl.is_allowed("192.168.1.254")
        assert not wl.is_allowed("192.168.2.1")

    def test_invalid_entries_ignored(self) -> None:
        wl = IPWhitelist(["nonsense", "10.0.0.0/8", "999.999.999.999"])
        assert len(wl.networks) == 1
        assert wl.is_allowed("10.1.2.3")
        assert not wl.is_allowed("11.0.0.1")

    def test_empty_list_denies_all(self) -> None:
        wl = IPWhitelist([])
        assert not wl.is_allowed("127.0.0.1")

    def test_invalid_ip_denied(self) -> None:
        wl = IPWhitelist(["10.0.0.0/8"])
        assert not wl.is_allowed("not-an-ip")


class TestMiddleware:
    def test_denied_ip_returns_403_and_audits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("API_TRUST_PROXY_HEADERS", "true")
        app = _app_with(create_live_ip_middleware(IPWhitelist(["10.0.0.0/8"])))
        client = TestClient(app)

        bad = client.get("/v1/live/ping", headers={"X-Forwarded-For": "8.8.8.8"})
        assert bad.status_code == 403
        assert bad.json() == {"error": "forbidden"}

        from packages.security.hardening.audit_live import get_live_audit

        trail = get_live_audit()
        assert len(trail) == 1
        entry = trail.entries[0]
        assert entry.action == "ip_denied"
        assert entry.details["ip"] == "8.8.8.8"
        assert entry.resource == "/v1/live/ping"

    def test_allowed_ip_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_TRUST_PROXY_HEADERS", "true")
        app = _app_with(create_live_ip_middleware(IPWhitelist(["10.0.0.0/8"])))
        client = TestClient(app)
        good = client.get("/v1/live/ping", headers={"X-Forwarded-For": "10.1.2.3"})
        assert good.status_code == 200

    def test_non_live_paths_unaffected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("API_TRUST_PROXY_HEADERS", "true")
        app = _app_with(create_live_ip_middleware(IPWhitelist(["10.0.0.0/8"])))
        client = TestClient(app)
        # /status is not a live path, so any IP is allowed.
        assert client.get("/status").status_code == 200

    def test_strict_empty_blocks_all(self) -> None:
        app = _app_with(create_live_ip_middleware(IPWhitelist([]), strict=True))
        client = TestClient(app)
        assert client.get("/v1/live/ping").status_code == 403

    def test_non_strict_empty_allows_all(self) -> None:
        app = _app_with(create_live_ip_middleware(IPWhitelist([]), strict=False))
        client = TestClient(app)
        assert client.get("/v1/live/ping").status_code == 200

    def test_env_strict_loads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LIVE_IP_WHITELIST", raising=False)
        monkeypatch.delenv("LIVE_IP_WHITELIST_STRICT", raising=False)
        app = _app_with(create_live_ip_middleware())
        client = TestClient(app)
        # Default (non-strict, empty) → allow all.
        assert client.get("/v1/live/ping").status_code == 200
