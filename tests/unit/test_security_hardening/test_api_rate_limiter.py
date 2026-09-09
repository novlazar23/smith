"""Tests for the per-IP sliding-window live API rate limiter."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from packages.security.hardening.api_rate_limiter import (
    LiveRateLimiter,
    create_live_rate_limit_middleware,
)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


@pytest.fixture(autouse=True)
def _clean_audit() -> None:
    from packages.security.hardening.audit_live import reset_live_audit

    reset_live_audit()


class TestLimiter:
    def test_request_scope(self) -> None:
        clock = FakeClock(0)
        limiter = LiveRateLimiter(request_limit=3, order_limit=5, window_seconds=60, clock=clock)
        assert limiter.check("ip", "request").allowed
        assert limiter.check("ip", "request").allowed
        d3 = limiter.check("ip", "request")
        assert d3.allowed
        assert d3.remaining == 0
        d4 = limiter.check("ip", "request")
        assert not d4.allowed
        # Different IP is unaffected.
        assert limiter.check("other", "request").allowed

    def test_window_slides(self) -> None:
        clock = FakeClock(0)
        limiter = LiveRateLimiter(request_limit=1, order_limit=5, window_seconds=60, clock=clock)
        assert limiter.check("ip", "request").allowed
        assert not limiter.check("ip", "request").allowed
        clock.advance(61)
        assert limiter.check("ip", "request").allowed

    def test_reset_seconds(self) -> None:
        clock = FakeClock(0)
        limiter = LiveRateLimiter(request_limit=2, order_limit=5, window_seconds=60, clock=clock)
        limiter.check("ip", "request")
        d = limiter.check("ip", "request")
        assert d.reset_seconds == 60  # window from first hit
        clock.advance(30)
        d = limiter.check("ip", "request")
        assert not d.allowed
        assert d.reset_seconds == 30  # 30 s until first slot frees

    def test_order_scope_isolated(self) -> None:
        clock = FakeClock(0)
        limiter = LiveRateLimiter(request_limit=5, order_limit=2, window_seconds=60, clock=clock)
        assert limiter.check("ip", "order").allowed
        assert limiter.check("ip", "order").allowed
        assert not limiter.check("ip", "order").allowed
        # Request scope unaffected.
        assert limiter.check("ip", "request").allowed

    def test_unknown_scope_raises(self) -> None:
        clock = FakeClock(0)
        limiter = LiveRateLimiter(clock=clock)
        with pytest.raises(ValueError):
            limiter.check("ip", "bogus")


class TestMiddleware:
    @pytest.fixture
    def app(self) -> FastAPI:
        app = FastAPI()
        limiter = LiveRateLimiter(
            request_limit=4, order_limit=2, window_seconds=60, clock=FakeClock(0)
        )
        app.middleware("http")(create_live_rate_limit_middleware(limiter))

        @app.get("/v1/live/ping")
        async def ping() -> dict[str, bool]:
            return {"ok": True}

        @app.post("/v1/live/orders")
        async def orders() -> dict[str, bool]:
            return {"created": True}

        @app.get("/status")
        async def status() -> dict[str, bool]:
            return {"ok": True}

        return app

    def test_headers_present_on_live(self, app: FastAPI) -> None:
        client = TestClient(app)
        r = client.get("/v1/live/ping")
        assert r.status_code == 200
        assert r.headers["X-RateLimit-Limit"] == "4"
        assert r.headers["X-RateLimit-Remaining"] == "3"
        assert r.headers["X-RateLimit-Reset"] == "60"

    def test_429_after_request_limit(self, app: FastAPI) -> None:
        client = TestClient(app)
        for _ in range(4):
            assert client.get("/v1/live/ping").status_code == 200
        r = client.get("/v1/live/ping")
        assert r.status_code == 429
        assert r.json() == {"error": "rate limited"}
        assert r.headers["X-RateLimit-Limit"] == "4"
        assert r.headers["X-RateLimit-Remaining"] == "0"

    def test_order_scope_stricter(self, app: FastAPI) -> None:
        """Order limit (2) fires before request limit (4) on POST /orders."""
        client = TestClient(app)
        # Two orders consume the order scope.
        assert client.post("/v1/live/orders").status_code == 200
        assert client.post("/v1/live/orders").status_code == 200
        # Third order → 429 even though request scope still has room.
        r = client.post("/v1/live/orders")
        assert r.status_code == 429
        assert r.headers["X-RateLimit-Limit"] == "2"
        # GET still works (request scope not exhausted).
        assert client.get("/v1/live/ping").status_code == 200

    def test_non_live_paths_unaffected(self, app: FastAPI) -> None:
        client = TestClient(app)
        r = client.get("/status")
        assert r.status_code == 200
        assert "X-RateLimit-Limit" not in r.headers
