"""Per-IP API-Rate-Limiting für Live-Endpunkte (EPIC-16 WP06).

:class:`LiveRateLimiter` implementiert ein Sliding Window pro IP und pro
Scope (``request`` für alle Live-Endpunkte, ``order`` strikter für
``POST /v1/live/orders``). Der Clock ist injectierbar (deterministische
Tests), Default ist ``time.monotonic``.

Env-Config (Defaults):
- ``LIVE_RATE_LIMIT_REQUESTS`` — 100 Requests/Fenster
- ``LIVE_RATE_LIMIT_ORDERS`` — 10 Orders/Fenster
- ``LIVE_RATE_LIMIT_WINDOW_SECONDS`` — 60 s

Die Middleware setzt auf Live-Antworten die Header
``X-RateLimit-Limit`` / ``X-RateLimit-Remaining`` / ``X-RateLimit-Reset``
und antwortet bei Überschreitung mit 429 + JSON-Error.
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from packages.security.hardening.ip_whitelist import (
    LIVE_PATH_PREFIXES,
    StarletteRequest,
    resolve_client_ip,
)

__all__ = [
    "DEFAULT_ORDER_LIMIT",
    "DEFAULT_REQUEST_LIMIT",
    "DEFAULT_WINDOW_SECONDS",
    "ORDER_SCOPE",
    "ORDER_SUBMIT_PATH",
    "REQUEST_SCOPE",
    "LiveRateLimiter",
    "RateLimitDecision",
    "create_live_rate_limit_middleware",
]

REQUEST_SCOPE = "request"
ORDER_SCOPE = "order"

#: Endpunkt, der zusätzlich im ``order``-Scope zählt.
ORDER_SUBMIT_PATH = "/v1/live/orders"

DEFAULT_REQUEST_LIMIT = 100
DEFAULT_ORDER_LIMIT = 10
DEFAULT_WINDOW_SECONDS = 60


@dataclass(frozen=True)
class RateLimitDecision:
    """Ergebnis einer Rate-Limit-Prüfung."""

    allowed: bool
    limit: int
    remaining: int
    reset_seconds: float


class _MutableHeaders(Protocol):
    """Mutable-Header-Duck-Type (starlette.MutableHeaders-Interface, read/write)."""

    def __getitem__(self, key: str) -> str: ...

    def __setitem__(self, key: str, value: str) -> None: ...


class _Response(Protocol):
    @property
    def headers(self) -> _MutableHeaders: ...


class LiveRateLimiter:
    """Sliding-Window-Rate-Limiter pro IP und Scope.

    ponytail: in-memory Store pro Prozess; verteiltes Limiting (Redis)
    nur falls die API multi-instance betrieben wird.
    """

    def __init__(
        self,
        *,
        request_limit: int = DEFAULT_REQUEST_LIMIT,
        order_limit: int = DEFAULT_ORDER_LIMIT,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if request_limit <= 0 or order_limit <= 0:
            raise ValueError("Limits müssen größer als 0 sein")
        if window_seconds <= 0:
            raise ValueError("window_seconds muss größer als 0 sein")
        self._limits: dict[str, int] = {
            REQUEST_SCOPE: request_limit,
            ORDER_SCOPE: order_limit,
        }
        self._window = float(window_seconds)
        self._clock = clock or time.monotonic
        self._hits: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {REQUEST_SCOPE: [], ORDER_SCOPE: []}
        )

    @classmethod
    def from_env(cls) -> LiveRateLimiter:
        """Erstellt einen Limiter mit ENV-Config (Defaults 100/10/60 s)."""

        def _positive_int(name: str, default: int) -> int:
            raw = os.environ.get(name, "").strip()
            if not raw:
                return default
            try:
                value = int(raw)
            except ValueError:
                return default
            return value if value > 0 else default

        return cls(
            request_limit=_positive_int("LIVE_RATE_LIMIT_REQUESTS", DEFAULT_REQUEST_LIMIT),
            order_limit=_positive_int("LIVE_RATE_LIMIT_ORDERS", DEFAULT_ORDER_LIMIT),
            window_seconds=float(
                _positive_int(
                    "LIVE_RATE_LIMIT_WINDOW_SECONDS", DEFAULT_WINDOW_SECONDS
                )
            ),
        )

    def check(self, ip: str, scope: str, now: float | None = None) -> RateLimitDecision:
        """Prüft (und bei Erlaubnis verbucht) einen Request im Scope."""
        try:
            limit = self._limits[scope]
        except KeyError as exc:
            raise ValueError(f"Unbekannter Scope {scope!r}") from exc
        ts = self._clock() if now is None else now
        cutoff = ts - self._window
        alive = [t for t in self._hits[ip][scope] if t > cutoff]
        if len(alive) >= limit:
            return RateLimitDecision(
                allowed=False,
                limit=limit,
                remaining=0,
                reset_seconds=alive[0] + self._window - ts,
            )
        alive.append(ts)
        self._hits[ip][scope] = alive
        return RateLimitDecision(
            allowed=True,
            limit=limit,
            remaining=limit - len(alive),
            reset_seconds=alive[0] + self._window - ts,
        )


def create_live_rate_limit_middleware(
    limiter: LiveRateLimiter | None = None,
    *,
    prefixes: tuple[str, ...] = LIVE_PATH_PREFIXES,
) -> Callable[[StarletteRequest, Callable[[StarletteRequest], Awaitable[_Response]]], Awaitable[_Response]]:
    """Erstellt die Live-Rate-Limit-Middleware.

    ``POST /v1/live/orders`` zählt in beiden Scopes; alle anderen
    Live-Endpunkte nur im ``request``-Scope. Der Strictere Scope bestimmt
    die Response-Header.
    """
    from fastapi.responses import JSONResponse

    if limiter is None:
        limiter = LiveRateLimiter.from_env()

    def _rate_limited(decision: RateLimitDecision) -> JSONResponse:
        response = JSONResponse(content={"error": "rate limited"}, status_code=429)
        _set_headers(response, decision)
        return response

    def _set_headers(response: _Response, decision: RateLimitDecision) -> None:
        response.headers["X-RateLimit-Limit"] = str(decision.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        response.headers["X-RateLimit-Reset"] = str(
            max(0, math.ceil(decision.reset_seconds))
        )

    async def middleware(
        request: StarletteRequest,
        call_next: Callable[[StarletteRequest], Awaitable[_Response]],
    ) -> _Response:
        path = request.url.path
        if not path.startswith(prefixes):
            return await call_next(request)

        ip = resolve_client_ip(request)
        is_order_submit = request.method == "POST" and path == ORDER_SUBMIT_PATH

        # Order-Submit zuerst prüfen (stricter Scope), damit abgelehnte
        # Order-Submits keinen Request-Scope-Slot verbrauchen.
        if is_order_submit:
            order = limiter.check(ip, ORDER_SCOPE)
            if not order.allowed:
                return _rate_limited(order)
            primary = order
        else:
            primary = None

        request_decision = limiter.check(ip, REQUEST_SCOPE)
        if not request_decision.allowed:
            return _rate_limited(request_decision)
        if primary is None:
            primary = request_decision

        response = await call_next(request)
        _set_headers(response, primary)
        return response

    return middleware
