"""Live-Signal Router — Generiert Order-Vorschläge ohne reale Order-Ausführung.

Dieser Endpunkt ist durch das Feature-Flag `live_trading_enabled` geschützt.
Er erzeugt niemals eine reale Order — nur ein Vorschlags-Dictionary.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from packages.governance.feature_flags import feature_flags

router = APIRouter(prefix="/v1/live-signal", tags=["live-signal"])


class LiveSignalRequest(BaseModel):
    """Anfrage für einen Live-Signal-Check."""

    model_config = ConfigDict(str_strip_whitespace=True)

    instrument: str = Field(..., min_length=1, max_length=50)
    analysis_time: datetime
    strategy: dict[str, Any] | None = Field(default=None)


class OrderSuggestion(BaseModel):
    """Strukturiertes Order-Vorschlags-Dictionary."""

    model_config = ConfigDict(frozen=True)

    action: str = Field(..., pattern="^(BUY|SELL|HOLD)$")
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)


class LiveSignalResponse(BaseModel):
    """Antwort des Live-Signal-Endpunkts."""

    model_config = ConfigDict(str_strip_whitespace=True)

    order_suggestion: OrderSuggestion | None
    status: str
    message: str
    audit_id: str


def _generate_suggestion(
    instrument: str,
    strategy: dict[str, Any] | None,
    analysis_time: datetime,
) -> OrderSuggestion:
    """Generiert ein Order-Vorschlags-Dictionary.

    CRITICAL: Diese Funktion erzeugt NUR einen Vorschlag und platziert
    KEINE reale Order. Sie darf keine execute/trade-Methoden aufrufen.
    """
    # Simple deterministic suggestion logic based on strategy params
    if strategy and strategy.get("aggressive", False):
        action = "BUY"
        confidence = 0.85
        reasoning = (
            f"Aggressive Strategie für {instrument} — "
            "Order-Vorschlag nur, keine Ausführung."
        )
    elif strategy and strategy.get("conservative", False):
        action = "HOLD"
        confidence = 0.6
        reasoning = (
            f"Konservative Strategie für {instrument} — "
            "kein Trade empfohlen."
        )
    else:
        action = "BUY"
        confidence = 0.75
        reasoning = (
            f"Standard-Signal für {instrument} — "
            "Order-Vorschlag nur, keine Ausführung."
        )

    return OrderSuggestion(
        action=action,
        quantity=round(abs(hash(f"{instrument}{analysis_time.isoformat()}")) % 100 + 1, 2),
        price=round((abs(hash(f"price{instrument}")) % 500 + 10) + 0.01, 2),
        confidence=confidence,
        reasoning=reasoning,
    )


@router.post("/", response_model=LiveSignalResponse, status_code=200)
async def live_signal_endpoint(request: LiveSignalRequest) -> LiveSignalResponse:
    """Generiert einen Live-Signal-Vorschlag.

    - Prüft das Feature-Flag `live_trading_enabled`.
    - Protokolliert den Signal-Check im Audit-Trail.
    - Generiert einen Order-Vorschlag (KEINE reale Order-Ausführung).
    """
    if not feature_flags.is_enabled("live_trading_enabled"):
        raise HTTPException(
            status_code=403,
            detail="Live trading disabled — feature flag not enabled.",
        )

    audit_id = f"AUDIT-{uuid.uuid4().hex[:8]}"

    # Audit-Trail-Eintrag erstellen
    from packages.governance.audit import AuditTrail

    AuditTrail().log_decision(
        agent_id="live-signal",
        decision=f"signal_generated:{request.instrument}",
        actor="system",
        details={
            "event": "live_signal",
            "instrument": request.instrument,
            "analysis_time": request.analysis_time.isoformat(),
            "strategy": request.strategy,
            "audit_id": audit_id,
        },
    )

    suggestion = _generate_suggestion(
        instrument=request.instrument,
        strategy=request.strategy,
        analysis_time=request.analysis_time,
    )

    return LiveSignalResponse(
        order_suggestion=suggestion,
        status="signal_generated",
        message=f"Signal für {request.instrument} generiert (KEINE Ausführung).",
        audit_id=audit_id,
    )