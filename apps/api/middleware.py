"""Middleware für Rate Limiting und API-Key Authentication.

Stellt zwei FastAPI-Middleware-Funktionen bereit:
- create_rate_limit_middleware(): Pro IP max 60 Requests/Minute
- create_auth_middleware(): Prüft X-API-Key Header gegen ENV var
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

try:
    from fastapi.responses import JSONResponse
except ImportError:
    JSONResponse = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# In-Memory Rate-Limit Store
# ---------------------------------------------------------------------------

_rate_store: dict[str, list[float]] = defaultdict(list)
_rate_lock = asyncio.Lock()

MAX_REQUESTS = 60
WINDOW_SECONDS = 60


async def _cleanup_and_check(ip: str) -> bool:
    """Prüft ob IP rate-limited ist, cleant alte Einträge.

    Returns:
        True wenn request erlaubt ist, False wenn rate-limited.
    """
    async with _rate_lock:
        now = time.time()
        cutoff = now - WINDOW_SECONDS
        # Alte Einträge entfernen
        _rate_store[ip] = [t for t in _rate_store[ip] if t > cutoff]
        # Prüfen
        if len(_rate_store[ip]) >= MAX_REQUESTS:
            return False
        _rate_store[ip].append(now)
        return True


# ---------------------------------------------------------------------------
# Middleware Creator
# ---------------------------------------------------------------------------


def create_rate_limit_middleware() -> Callable:
    """Erstellt Rate-Limiting Middleware (60 req/min pro IP).

    Gibt 429 mit {"error": "rate limited"} wenn Limit überschritten.
    """

    async def middleware(request, call_next):
        client_host = request.client.host if request.client else "unknown"
        allowed = await _cleanup_and_check(client_host)
        if not allowed:
            return JSONResponse(
                content={"error": "rate limited"},
                status_code=429,
            )
        response = await call_next(request)
        return response

    return middleware


# Endpunkte die KEINE Authentifizierung benötigen
# /metrics: Prometheus-Scraper läuft ohne API-Key gegen den Endpunkt
_OPEN_ENDPOINTS = frozenset(["/health", "/status", "/metrics"])


def create_auth_middleware() -> Callable | None:
    """Erstellt API-Key Auth Middleware.

    Liest den Secret-Key aus einer Datei (Pfad via API_SECRET_KEY_FILE ENV)
    oder aus der ENV-Var API_SECRET_KEY.
    Gibt None zurück wenn kein Key konfiguriert — Auth ist dann disabled.
    Öffentliche Endpunkte (/health, /status) werden ausgenommen.
    Gibt 401 mit {"error": "unauthorized"} wenn Key fehlt/ungültig.
    """
    expected_key: str | None = None

    secret_file = os.environ.get("API_SECRET_KEY_FILE")
    if secret_file:
        with contextlib.suppress(FileNotFoundError, PermissionError):
            expected_key = Path(secret_file).read_text().strip()

    if not expected_key:
        expected_key = os.environ.get("API_SECRET_KEY")

    if not expected_key:
        return None

    async def middleware(request, call_next):
        if request.url.path in _OPEN_ENDPOINTS:
            return await call_next(request)
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != expected_key:
            return JSONResponse(
                content={"error": "unauthorized"},
                status_code=401,
            )
        response = await call_next(request)
        return response

    return middleware


__all__ = ["create_auth_middleware", "create_rate_limit_middleware"]
