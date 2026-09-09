"""IP-Whitelisting für Live-Endpunkte (EPIC-16 WP06).

- :class:`IPWhitelist` — erlaubte Single-IPs und CIDRs (``ipaddress``-basiert,
  invalide Einträge werden deterministisch ignoriert).
- :func:`create_live_ip_middleware` — FastAPI-Middleware für Live-Präfixe;
  abgelehnte IPs werden im Live-Audit-Trail protokolliert und mit 403
  beantwortet.

Env-Config:
- ``LIVE_IP_WHITELIST`` — kommagetrennte IPs/CIDRs
- ``LIVE_IP_WHITELIST_STRICT`` — ``true``: leere Whitelist blockiert alle
  Live-Endpunkte; Default (aus): leere Whitelist blockiert nichts.
- ``API_TRUST_PROXY_HEADERS`` — ``true``: erster ``X-Forwarded-For``-Wert
  wird als Client-IP verwendet.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from packages.security.hardening.audit_live import LiveAuditTrail, get_live_audit

__all__ = [
    "LIVE_IP_WHITELIST_ENV",
    "LIVE_IP_WHITELIST_STRICT_ENV",
    "LIVE_PATH_PREFIXES",
    "IPWhitelist",
    "StarletteRequest",
    "create_live_ip_middleware",
    "load_ip_whitelist",
    "resolve_client_ip",
]

LIVE_IP_WHITELIST_ENV = "LIVE_IP_WHITELIST"
LIVE_IP_WHITELIST_STRICT_ENV = "LIVE_IP_WHITELIST_STRICT"
TRUST_PROXY_HEADERS_ENV = "API_TRUST_PROXY_HEADERS"

#: Pfad-Präfixe, die als Live gelten (IP-Whitelist + Rate-Limiting).
LIVE_PATH_PREFIXES: tuple[str, ...] = (
    "/v1/live/",
    "/v1/health/live",
    "/v1/live-signal",
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

Network = ipaddress.IPv4Network | ipaddress.IPv6Network
Address = ipaddress.IPv4Address | ipaddress.IPv6Address


class _URL(Protocol):
    path: str


class _Client(Protocol):
    host: str


class StarletteRequest(Protocol):
    """Minimaler Request-Duck-Type für die Middleware (kein FastAPI-Import nötig)."""

    method: str
    url: _URL
    headers: Mapping[str, str]
    client: _Client | None


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE_VALUES


def _parse_entry(entry: str) -> Network | None:
    text = entry.strip()
    if not text:
        return None
    try:
        return ipaddress.ip_network(text, strict=False)
    except ValueError:
        return None


def _parse_ip(ip: str) -> Address | None:
    try:
        return ipaddress.ip_address(ip.strip())
    except ValueError:
        return None


class IPWhitelist:
    """Allowlist aus Single-IPs und CIDR-Netzen.

    Invalide Einträge werden deterministisch ignoriert (nicht gemeldet,
    da sie aus ENV kommen und keinen Hinweis auf Secrets geben).
    """

    def __init__(self, entries: Sequence[str] | None = None) -> None:
        self._networks: list[Network] = []
        for entry in entries or ():
            network = _parse_entry(entry)
            if network is not None:
                self._networks.append(network)

    @property
    def networks(self) -> list[Network]:
        """Kopie der erlaubten Netzwerke."""
        return list(self._networks)

    def is_allowed(self, ip: str) -> bool:
        """True wenn die IP in einem erlaubten Netz liegt. Fail closed."""
        if not self._networks:
            return False
        address = _parse_ip(ip)
        if address is None:
            return False
        return any(address in network for network in self._networks)


def load_ip_whitelist() -> tuple[IPWhitelist, bool]:
    """Liest Whitelist und Strict-Modus aus ENV."""
    raw = os.environ.get(LIVE_IP_WHITELIST_ENV, "")
    entries = [part.strip() for part in raw.split(",") if part.strip()]
    strict = _truthy(os.environ.get(LIVE_IP_WHITELIST_STRICT_ENV))
    return IPWhitelist(entries), strict


def resolve_client_ip(request: StarletteRequest) -> str:
    """Bestimmt die Client-IP.

    Default: ``request.client.host``. Bei ``API_TRUST_PROXY_HEADERS=true``
    wird der erste ``X-Forwarded-For``-Wert verwendet (falls vorhanden).
    """
    if _truthy(os.environ.get(TRUST_PROXY_HEADERS_ENV)):
        forwarded = request.headers.get("X-Forwarded-For", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = request.client
    return client.host if client is not None else "unknown"


def create_live_ip_middleware(
    whitelist: IPWhitelist | None = None,
    *,
    strict: bool | None = None,
    audit: LiveAuditTrail | None = None,
) -> Callable[[StarletteRequest, Callable[[StarletteRequest], Awaitable[object]]], Awaitable[object]]:
    """Erstellt die IP-Whitelist-Middleware für Live-Präfixe.

    - Leere Whitelist + strict=True → blockiert alle Live-Endpunkte.
    - Leere Whitelist + strict=False (Default) → blockiert nichts.
    - Sonst: nur erlaubte IPs; alle Abweisungen landen im Live-Audit-Trail.
    """
    from fastapi.responses import JSONResponse

    env_strict = _truthy(os.environ.get(LIVE_IP_WHITELIST_STRICT_ENV))
    if whitelist is None:
        whitelist, _ = load_ip_whitelist()
    if strict is None:
        strict = env_strict

    async def middleware(
        request: StarletteRequest,
        call_next: Callable[[StarletteRequest], Awaitable[object]],
    ) -> object:
        path = request.url.path
        if not path.startswith(LIVE_PATH_PREFIXES):
            return await call_next(request)

        ip = resolve_client_ip(request)
        allowed = not strict if not whitelist.networks else whitelist.is_allowed(ip)

        if not allowed:
            trail = audit if audit is not None else get_live_audit()
            trail.record(
                "ip_denied",
                actor=ip,
                resource=path,
                status="denied",
                details={"ip": ip, "path": path},
            )
            return JSONResponse(content={"error": "forbidden"}, status_code=403)

        return await call_next(request)

    return middleware
