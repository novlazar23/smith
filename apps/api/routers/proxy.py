"""Proxy-Router — Reverse-Proxy für interne Observability- und Storage-UIs.

Leitet Anfragen unter ``/proxy/{name}`` an fünf interne Dienste weiter:

    grafana      → http://grafana:3000      (stript X-Frame-Options, CSP)
    minio        → http://minio:9001        (stript X-Frame-Options, CSP)
    prometheus   → http://prometheus:9090
    alertmanager → http://alertmanager:9093
    mlflow       → http://mlflow:5000

Der Name wird beim Upstream gestrippt (Upstream-„/" — Ausnahme Grafana:
Grafana liefert nur unter seinem Subpfad, s. ``PROXY_TARGETS``).

Damit kann das Web-Dashboard (gleiche Herkunft, ``localhost:8080``) die UIs
per iframe einbetten, obwohl Grafana und MinIO mit ``X-Frame-Options: DENY``
das Embedding prinzipiell verbieten — die beiden Header (sowie die
Content-Security-Policy) werden aus den Antworten entfernt.

Weiterleitungsregeln:
  - Methode, Query-String, Request-Body (gestreamt) und Request-Header,
    ausgenommen hop-by-hop-Header (host, connection, keep-alive, proxy-*,
    transfer-encoding, te, upgrade); ``Host`` wird auf das Upstream-Netloc
    gesetzt.
  - Statuscode und alle Response-Header, ausgenommen connection, keep-alive,
    proxy-*, transfer-encoding, te, upgrade, content-encoding,
    content-length sowie die ziel-spezifische Strip-Liste.
    Content-Encoding/Content-Length fallen weg, weil httpx den Body
    transparent dekomprimiert — weitergereicht wären sie inkonsistent zum
    (größeren, unkomprimierten) Body und bräuchten den Client
    (net::ERR_CONTENT_LENGTH_MISMATCH, „Failed to fetch").
    Set-Cookie-Header bleiben erhalten (Grafana-Session); mehrfach gesendete
    Header werden kollabiert (letzter Wert gewinnt).
  - HTML-Antworten (Content-Type text/html) erhalten zusätzlich ein
    ``<base href="/">``-Tag auf ``<base href="/proxy/<name>/">`` umgeschrieben
    — SPAs wie die MinIO-Console würden sonst ihre Asset- und API-URLs am
    API-Root auflösen (→ 404 statt dem Proxy-Subpfad).
  - Die Upstream-Antwort wird vollständig empfangen, bevor der Proxy
    antwortet (Content-Type inkl. Charset unverändert aus dem Upstream) —
    ein abgebrochener Upstream-Stream (z. B. Gunicorn-Worker-Recycling bei
    MLflow) bricht den Client nie mit einem gekürzten Body; statt dessen
    antwortet der Proxy mit 502. GET/HEAD-Requests werden in diesem Fall
    einmal automatisch wiederholt (idempotent).
  - MinIO: Die API meldet sich automatisch an (``/api/v1/login`` mit den
    S3-Credentials aus ``MINIO_ACCESS_KEY``/``MINIO_SECRET_KEY_FILE``), wenn
    der Browser kein Session-Token mitbringt; bei 401/403 wird einmal mit
    frischer Session wiederholt. Die Session-Cookie wird dem Browser mit
    ``Path=/proxy/minio`` übergeben — der Tab lädt ohne Login-Formular.

Bei unerreichbarem Upstream antwortet der Proxy mit HTTP 502 und dem
JSON-Body ``{"error": "upstream unavailable", "target": <name>}`` — er hängt
nie und bringt die App nie mit HTTP 500 zu Fall.

Der httpx-Client wird über die modulweite ``client_factory`` erzeugt, damit
Tests eine Fake-Fabrik injizieren können.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/proxy", tags=["proxy"])

# MinIO-Auto-Login: die Console erwartet S3-Credentials unter /api/v1/login
# (accessKey/secretKey) und zeigt andernfalls permanent das Login-Formular.
# Die Credentials kommen aus der Umgebung (Secret-Datei im API-Container).
_MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "orchestra")
_MINIO_SECRET_KEY_FILE = os.environ.get("MINIO_SECRET_KEY_FILE", "")

# Zielregister: Name → (Upstream-Basis-URL, Upstream-Pfadpräfix, zu entfernende Response-Header)
# Grafana betreibt GF_SERVER_SERVE_FROM_SUB_PATH=true und liefert nur unter dem
# Subpfad „/proxy/grafana" (Pfad der GF_SERVER_ROOT_URL) — ein Request an das
# Upstream-„/" würde 301 in eine Endlosschleipe umleiten. Deshalb wird bei
# Grafana der Subpfad an das Upstream weitergegeben, bei allen anderen Zielen
# wird der Name gestrippt (Upstream-„/").
PROXY_TARGETS: dict[str, tuple[str, str, frozenset[str]]] = {
    "grafana": (
        "http://grafana:3000",
        "proxy/grafana",
        frozenset({"x-frame-options", "content-security-policy"}),
    ),
    "minio": ("http://minio:9001", "", frozenset({"x-frame-options", "content-security-policy"})),
    "prometheus": ("http://prometheus:9090", "", frozenset()),
    "alertmanager": ("http://alertmanager:9093", "", frozenset()),
    "mlflow": ("http://mlflow:5000", "", frozenset()),
}

# Hop-by-hop-Header: werden nicht an das Upstream weitergeleitet
_REQUEST_HOP_HEADERS: frozenset[str] = frozenset(
    {"host", "connection", "keep-alive", "transfer-encoding", "te", "upgrade"}
)
# Hop-by-hop-Header: werden aus der Upstream-Antwort entfernt
_RESPONSE_HOP_HEADERS: frozenset[str] = frozenset(
    {"connection", "keep-alive", "transfer-encoding", "te", "upgrade"}
)
# Encoding-Header: werden aus der Upstream-Antwort entfernt. httpx dekomprimiert
# gzip/deflate transparent; der weitergeleitete Body ist damit nicht mehr in der
# deklarierten Encoding/Länge — die Header würden den Client brechen
# (net::ERR_CONTENT_LENGTH_MISMATCH).
_ENCODING_HEADERS: frozenset[str] = frozenset({"content-encoding", "content-length"})

# Sicherheitslimit für gepufferte Antworten (Grafana-Bundles können > 10 MB sein)
_MAX_BODY_BUFFER = 50_000_000
# Wurzel-Base-Tag (Selbstschließende-Strich optional), das die MinIO-Console
# (und weitere SPAs) für die Asset-Auflösung sendet
_BASE_TAG_RE = re.compile(rb"""<base\s+href\s*=\s*"/"\s*/?>""")

# Vom Proxy weitergeleitete HTTP-Methoden
_PROXY_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]


def _default_client() -> httpx.AsyncClient:
    """Erzeugt einen httpx-Client mit 30 s Connect-/Read-Timeout."""
    return httpx.AsyncClient(timeout=30.0)


# Client-Fabrik — injizierbar für Tests (Default: 30 s Connect-/Read-Timeout)
client_factory: Callable[[], httpx.AsyncClient] = _default_client


def _is_hopped_out(header_name: str, excluded: frozenset[str]) -> bool:
    """Prüft, ob ein Header hop-by-hop ist und nicht weitergegeben wird."""
    lowered = header_name.lower()
    return lowered in excluded or lowered.startswith("proxy-")


def _minio_secret_key() -> str:
    """Liest den MinIO-Root-Key (Env-Variablen, sonst Secret-Datei; leer = kein Auto-Login).

    Im Container exportiert der Entrypoint das Docker-Secret als Root nach
    ``MINIO_SECRET_KEY`` (der App-User darf die Secret-Datei nicht lesen).
    """
    key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    if key:
        return key
    if not _MINIO_SECRET_KEY_FILE:
        return ""
    try:
        return Path(_MINIO_SECRET_KEY_FILE).read_text().strip()
    except OSError:
        return ""


async def _minio_login(client: httpx.AsyncClient, base_url: str) -> str | None:
    """Meldet die API bei der MinIO-Console an und liefert die Session-Cookie.

    Die Antwort-Cookie wird auf ``Path=/proxy/minio`` umgeschrieben, damit
    der Browser das Token nur unter dem Proxy-Subpfad sendet (und es nicht
    an andere Upstreams weiterleitet).
    """
    secret = _minio_secret_key()
    if not secret:
        return None
    try:
        login_request = httpx.Request(
            "POST",
            f"{base_url}/api/v1/login",
            json={"accessKey": _MINIO_ACCESS_KEY, "secretKey": secret},
        )
        login_response = await client.send(login_request, stream=True)
        try:
            if login_response.status_code != 204:
                logger.warning(
                    "MinIO-Auto-Login: unerwarteter Status %s", login_response.status_code
                )
                return None
            raw_cookie = login_response.headers.get("set-cookie", "")
        finally:
            await login_response.aclose()
        token_part = raw_cookie.split(";", 1)[0].strip()
        if not token_part:
            return None
        return f"{token_part}; Path=/proxy/minio; HttpOnly; SameSite=Lax; Max-Age=43200"
    except httpx.HTTPError:
        logger.warning("MinIO-Auto-Login fehlgeschlagen", exc_info=True)
        return None


def _build_upstream_url(base_url: str, prefix: str, subpath: str, query_string: str) -> str:
    """Baut die Upstream-URL aus Basis-URL, Präfix, Subpfad und Query-String."""
    path = f"/{prefix}/{subpath}" if prefix else f"/{subpath}"
    url = f"{base_url.rstrip('/')}{path}"
    return f"{url}?{query_string}" if query_string else url


async def _proxy(request: Request, name: str, subpath: str) -> Response:
    """Leitet die Anfrage an das Ziel ``name`` weiter und gibt die volle Antwort zurück.

    Unbekannte Ziele antworten mit HTTP 404; ein unerreichbarer oder
    abbrechender Upstream mit HTTP 502 und
    ``{"error": "upstream unavailable", "target": name}``.
    """
    target = PROXY_TARGETS.get(name)
    if target is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Unbekanntes Proxy-Ziel: {name}"},
        )
    base_url, prefix, strip_headers = target
    upstream_url = _build_upstream_url(base_url, prefix, subpath, request.url.query)
    netloc = urlsplit(base_url).netloc

    headers: dict[str, str] = {
        key: value
        for key, value in request.headers.items()
        if not _is_hopped_out(key, _REQUEST_HOP_HEADERS)
    }
    headers["host"] = netloc

    client_body = await request.body()
    client = client_factory()
    try:
        session_cookie: str | None = None
        if name == "minio" and "token=" not in headers.get("cookie", ""):
            session_cookie = await _minio_login(client, base_url)
            if session_cookie is not None:
                headers = {**headers, "cookie": session_cookie}

        result = await _fetch_upstream(client, name, request.method, upstream_url, headers, client_body)
        if name == "minio" and result is not None and result[0] in (401, 403):
            # Abgelaufene Browser-Session → frische Anmeldung und einmalige
            # Wiederholung der ursprünglichen Anfrage.
            fresh = await _minio_login(client, base_url)
            if fresh is not None:
                session_cookie = fresh
                headers = {**headers, "cookie": fresh}
                result = await _fetch_upstream(
                    client, name, request.method, upstream_url, headers, client_body
                )
    finally:
        await client.aclose()

    if result is None:
        return JSONResponse(
            status_code=502,
            content={"error": "upstream unavailable", "target": name},
        )
    status_code, upstream_headers, body = result

    response_headers: dict[str, str] = {}
    for key, value in upstream_headers.multi_items():
        if _is_hopped_out(key, _RESPONSE_HOP_HEADERS | _ENCODING_HEADERS | strip_headers):
            continue
        response_headers[key] = value
    if session_cookie is not None:
        response_headers["set-cookie"] = session_cookie

    content_type = upstream_headers.get("content-type", "")
    if "text/html" in content_type.lower():
        base_tag = f'<base href="/proxy/{name}/">'.encode("ascii")
        body, _ = _BASE_TAG_RE.subn(base_tag, body, count=1)

    return Response(content=body, status_code=status_code, headers=response_headers)


async def _fetch_upstream(
    client: httpx.AsyncClient,
    name: str,
    method: str,
    upstream_url: str,
    headers: dict[str, str],
    body: bytes,
) -> tuple[int, httpx.Headers, bytes] | None:
    """Lädt die Upstream-Antwort vollständig; gibt None bei Fehlschlag zurück.

    Der Body wird komplett empfangen, bevor der Proxy antwortet — ein
    abgebrochener Upstream-Stream (z. B. Gunicorn-Worker-Recycling bei
    MLflow) bricht den Client nie mit einem gekürzten Body; statt dessen
    antwortet der Proxy mit 502. GET/HEAD-Requests werden bei unvollständiger
    Antwort einmal automatisch wiederholt (idempotent).
    """
    attempts = 2 if method in ("GET", "HEAD") else 1
    for attempt in range(attempts):
        upstream_response: httpx.Response | None = None
        try:
            upstream_request = httpx.Request(
                method=method,
                url=upstream_url,
                headers=headers,
                content=body if body else None,
            )
            upstream_response = await client.send(upstream_request, stream=True)
            data = b""
            async for chunk in upstream_response.aiter_bytes():
                data += chunk
                if len(data) > _MAX_BODY_BUFFER:
                    logger.warning("Proxy-Antwort von %s übersteigt das Pufferlimit", name)
                    return None
            return upstream_response.status_code, upstream_response.headers, data
        except httpx.HTTPError:
            if attempt + 1 < attempts:
                logger.warning(
                    "Upstream %s lieferte eine unvollständige Antwort, Wiederholung", name, exc_info=True
                )
                continue
            logger.warning("Proxy-Ziel %s ist nicht erreichbar", name, exc_info=True)
            return None
        finally:
            if upstream_response is not None:
                await upstream_response.aclose()
    return None


# include_in_schema=False: Der Proxy ist ein Reverse-Proxy für feste Pfade —
# seine 7x2 Methoden-Varianten wären im OpenAPI-Schema nur Rauschen.
@router.api_route("/{name}/{path:path}", methods=_PROXY_METHODS, include_in_schema=False)
async def proxy_passthrough(request: Request, name: str, path: str) -> Response:
    """Leitet ``/proxy/{name}/{path}`` an das Ziel ``name`` weiter."""
    return await _proxy(request, name, path)


@router.api_route("/{name}", methods=_PROXY_METHODS, include_in_schema=False)
async def proxy_root(request: Request, name: str) -> Response:
    """Leitet ``/proxy/{name}`` an die Upstream-Wurzel „/" weiter."""
    return await _proxy(request, name, "")
