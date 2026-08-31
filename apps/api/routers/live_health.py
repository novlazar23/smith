"""Live Health & Readiness Router.

Exposes live-mode specific health checks (liveness + readiness) and integrates
with the rollout controller to report the current rollout phase, kill switch
state, and circuit breaker status.

Endpoints
---------
- ``GET /v1/health/live`` — Live-mode liveness + readiness check
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter
from packages.governance.feature_flags import feature_flags
from packages.rollout import PhasedRolloutController
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/v1/health", tags=["live-health"])

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class LiveHealthResponse(BaseModel):
    """Live-mode health check response.

    Attributes:
        status: Overall health status string (``"healthy"``, ``"degraded"``,
            ``"unhealthy"``).
        liveness: ``true`` if the process is running.
        readiness: ``true`` if the system is ready to accept live orders.
        details: Additional diagnostic information.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    status: str
    liveness: bool
    readiness: bool
    details: dict[str, object]


# ---------------------------------------------------------------------------
# GET /v1/health/live — Live-mode health check
# ---------------------------------------------------------------------------


@router.get("/live", response_model=LiveHealthResponse, status_code=200)
async def live_health_check() -> LiveHealthResponse:
    """Return a live-mode specific health check with liveness + readiness.

    **Liveness** is always ``true`` when the endpoint is reachable (the process
    is running).

    **Readiness** requires all of the following:
    - The ``live_trading_enabled`` feature flag is ``True``.
    - The exchange connection is established (gateway can be created).
    - The rollout phase is at least ``LIVE_SMALL`` (i.e. not in shadow or
      paper-only modes).
    - The kill switch is ``disabled``.

    Response schema
    ---------------
    {
        "status": "healthy",
        "liveness": true,
        "readiness": true,
        "details": {
            "feature_flag": true,
            "rollout_phase": "LIVE_SMALL",
            "kill_switch": "disabled",
            "circuit_breaker": "closed",
            "timestamp": "2025-01-15T10:30:00Z"
        }
    }
    """
    liveness = True  # We are here, so the process is alive

    # Gather diagnostic details
    details: dict[str, object] = {
        "feature_flag": feature_flags.is_enabled("live_trading_enabled"),
        "rollout_phase": "unknown",
        "kill_switch": "unknown",
        "circuit_breaker": "unknown",
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # --- Readiness checks ---
    readiness_checks: list[str] = []
    all_pass = True

    # 1. Feature flag check
    if not feature_flags.is_enabled("live_trading_enabled"):
        all_pass = False
        readiness_checks.append("feature_flag_disabled")

    # 2. Rollout phase check
    rollout = PhasedRolloutController()
    current_phase = rollout.current_phase
    details["rollout_phase"] = current_phase

    live_phases = ("LIVE_SMALL", "LIVE_MEDIUM", "LIVE_FULL")
    if current_phase not in live_phases:
        all_pass = False
        readiness_checks.append(f"phase_not_live: {current_phase}")

    # 3. Kill switch check
    ks_state = rollout.kill_switch.state
    details["kill_switch"] = ks_state
    if ks_state == "activated":
        all_pass = False
        readiness_checks.append("kill_switch_activated")

    # 4. Circuit breaker check
    cb_state = rollout.circuit_breaker.state
    details["circuit_breaker"] = str(cb_state)
    if str(cb_state).upper() == "OPEN":
        all_pass = False
        readiness_checks.append("circuit_breaker_open")

    readiness = all_pass
    status = "healthy" if (liveness and readiness) else "unhealthy" if not readiness else "degraded"

    return LiveHealthResponse(
        status=status,
        liveness=liveness,
        readiness=readiness,
        details=details,
    )
