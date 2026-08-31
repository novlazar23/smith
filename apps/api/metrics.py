"""Prometheus-Metriken und Middleware für die Trading Orchestra API.

Stellt den Request-Zähler (http_requests_total) und das Latenz-Histogramm
(http_request_duration_seconds) bereit sowie die Middleware, die beide
pro Anfrage erhöht. Eine eigene CollectorRegistry wird verwendet, damit
Neuimports (z.B. in Tests) keine Duplikat-Fehler in der Default-Registry
auslösen.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

# Eigene Registry statt der Default-Registry: Der Endpunkt /metrics rendert
# nur diese Metriken, und Test-Neuimports befüllen keine fremde Registry.
REGISTRY: CollectorRegistry = CollectorRegistry()

HTTP_REQUESTS_TOTAL: Counter = Counter(
    "http_requests_total",
    "HTTP requests",
    ["method", "path", "status"],
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION_SECONDS: Histogram = Histogram(
    "http_request_duration_seconds",
    "HTTP latency",
    ["method", "path"],
    registry=REGISTRY,
)


def _request_path(request: Request) -> str:
    """Liefert den generischen Routen-Pfad (Schutz vor Cardinality-Explosion).

    Verwendet den registrierten Pfad-Template (z.B. /items/{item_id}) aus
    request.scope["route"], sofern FastAPI ihn bereitstellt, sonst den
    konkreten Request-Pfad.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if isinstance(template, str) and template:
        return template
    return request.url.path


async def metrics_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Zählt HTTP-Anfragen und misst deren Dauer pro Methode, Pfad und Status."""
    start = time.monotonic()
    response = await call_next(request)
    path = _request_path(request)
    HTTP_REQUESTS_TOTAL.labels(
        method=request.method, path=path, status=str(response.status_code)
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=request.method, path=path).observe(
        time.monotonic() - start
    )
    return response


def render_metrics() -> bytes:
    """Rendert die Metriken im Prometheus-Textformat (Version 0.0.4)."""
    return generate_latest(REGISTRY)
