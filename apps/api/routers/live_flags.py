"""Live-Flag-Verwaltung: Zustand und Runtime-Schaltung von ``live_trading_enabled``.

Endpoints
---------
- ``GET  /v1/live/flag`` — Flag-Zustand inkl. Umgebung und Credential-Status
- ``POST /v1/live/flag`` — Flag schalten (MANAGE_KILL_SWITCH-Berechtigung,
  auditget via set_flag)

Das POST-Endpunkt prüft das Flag selbst bewusst nicht (Henne-Ei-Problem:
zum Einschalten darf es noch nicht gesetzt sein). Schutz erfolgt über die
Rollen-Berechtigung (``X-Security-Role`` wie im Rest der Live-API) und den
Audit-Trail-Eintrag pro Änderung in ``set_flag``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import packages.governance.feature_flags as feature_flags_module
from fastapi import APIRouter, Depends, HTTPException, Request
from packages.security import Permission, Role, SecurityContext
from packages.security.hardening.rbac_live import get_role_from_request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/live", tags=["live-flags"])


class FlagUpdate(BaseModel):
    enabled: bool = Field(..., description="Neuer Zustand des live_trading_enabled-Flags")


async def _require_flag_manager(request: Request) -> Role:
    role = get_role_from_request(request)
    if role is None:
        raise HTTPException(status_code=403, detail="Missing or unknown security role.")
    if not SecurityContext(role=role).has_permission(Permission.MANAGE_KILL_SWITCH):
        raise HTTPException(
            status_code=403,
            detail=f"Role '{role.value}' lacks permission 'manage_kill_switch'.",
        )
    return role


def _state() -> dict[str, Any]:
    from packages.live_execution.credentials import build_ccxt_config_from_env

    try:
        _venues, config, _ring = build_ccxt_config_from_env()
        credentials = {venue: bool(cfg) for venue, cfg in config.items()}
    except Exception:  # Defekte Credential-Config darf den Read nicht brechen
        credentials = {}
    return {
        "flag": "live_trading_enabled",
        "enabled": feature_flags_module.feature_flags.is_enabled("live_trading_enabled"),
        "environment": os.environ.get("APP_ENV", os.environ.get("ENV", "development")),
        "credentials_configured": credentials,
    }


@router.get("/flag")
async def get_live_flag() -> dict[str, Any]:
    return _state()


@router.post("/flag")
async def set_live_flag(
    update: FlagUpdate,
    request: Request,
    role: Role = Depends(_require_flag_manager),  # noqa: B008
) -> dict[str, Any]:
    feature_flags_module.feature_flags.set_flag("live_trading_enabled", enabled=update.enabled)
    logger.warning(
        "live_trading_enabled auf %s geschaltet (Rolle: %s)", update.enabled, role.value
    )
    return _state()
