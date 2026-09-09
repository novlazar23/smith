"""RBAC für Live-API-Endpunkte (EPIC-16 WP06).

- :func:`get_role_from_request` — liest die Rolle aus ``request.state.role``
  (priorisiert) oder dem ``X-Security-Role``-Header; unbekannte Rollen
  werden abgelehnt (fail closed).
- :func:`require_live_permission` — FastAPI-Dependency-Fabriek, die die
  ``live_trading_enabled``-Feature-Flag und die benötigte Permission
  prüft und andernfalls 403 liefert.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from fastapi import HTTPException, Request
from packages.security import Permission, Role, SecurityContext

__all__ = ["ROLE_HEADER", "get_role_from_request", "parse_role", "require_live_permission"]

ROLE_HEADER = "X-Security-Role"


class RequestLike(Protocol):
    """Minimaler Request-Duck-Type (state + headers)."""

    @property
    def state(self) -> object: ...

    @property
    def headers(self) -> Mapping[str, str]: ...


def parse_role(value: Role | str) -> Role | None:
    """Parst eine Rolle; unbekannte Werte liefern None (fail closed)."""
    if isinstance(value, Role):
        return value
    try:
        return Role(str(value).strip())
    except (TypeError, ValueError):
        return None


def get_role_from_request(request: RequestLike) -> Role | None:
    """Liefert die Rolle der Anfrage oder None.

    Priorität: ``request.state.role`` (z. B. gesetzt durch Auth-Middleware)
    vor dem ``X-Security-Role``-Header.
    """
    raw: object = None
    state = getattr(request, "state", None)
    if state is not None:
        raw = getattr(state, "role", None)
    if raw is None:
        raw = request.headers.get(ROLE_HEADER)
    if raw is None:
        return None
    if isinstance(raw, (str, Role)):
        return parse_role(raw)
    return None


def require_live_permission(permission: Permission) -> Callable[[Request], Awaitable[Role]]:
    """Erstellt eine FastAPI-Dependency für ``permission``.

    Fail-closed: Feature-Flag deaktiviert, fehlende/unbekannte Rolle
    oder fehlende Permission führen alle zu 403.
"""

    async def dependency(request: Request) -> Role:
        from packages.governance.feature_flags import feature_flags

        if not feature_flags.is_enabled("live_trading_enabled"):
            raise HTTPException(
                status_code=403,
                detail="Live trading is disabled — feature flag not enabled.",
            )
        role = get_role_from_request(request)
        if role is None:
            raise HTTPException(
                status_code=403,
                detail="Missing or unknown security role.",
            )
        if not SecurityContext(role=role).has_permission(permission):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role.value}' lacks permission '{permission.value}'.",
            )
        return role

    return dependency
