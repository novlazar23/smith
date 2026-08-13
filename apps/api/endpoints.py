"""Endpunkt-Logik für die Trading Orchestra API.

Die Endpunkt-Funktionen sind von der FastAPI-App getrennt, um die Logik
klar zu halten und unit-tests zu erleichtern.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from packages.governance.audit import AuditTrail
from packages.governance.feature_flags import feature_flags

logger = logging.getLogger(__name__)


async def analyze_endpoint(request: Any) -> dict[str, Any]:  # noqa: ANN401
    """Verarbeitet eine Analyse-Anfrage.

    Nimmt ein AnalyzeRequest-Objekt entgegen und gibt ein Ergebnis-Dictionary zurück.
    Die eigentliche Trading-Analyse wird an Worker delegiert.
    """
    instrument: str = request.instrument
    horizons: list[str] = request.horizons
    strategy: dict[str, Any] = request.strategy

    logger.info(
        "Processing analysis for %s, horizons=%s, strategy_keys=%s",
        instrument,
        horizons,
        list(strategy.keys()),
    )

    return {
        "instrument": instrument,
        "horizons": horizons,
        "status": "processing",
        "analysis_id": "pending",
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def status_endpoint() -> dict[str, Any]:
    """Gibt den Systemstatus einschließlich Modul-Verfügbarkeit zurück."""
    return {
        "version": "0.1.0",
        "status": "running",
        "uptime_seconds": 0.0,
        "modules": {},
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def health_endpoint() -> dict[str, Any]:
    """Gibt einen einfachen Health-Check zurück."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def live_signal_endpoint(request: Any) -> dict[str, Any]:
    """Generiert einen Live-Signal-Vorschlag ohne reale Order-Ausführung.

    Prüft das Feature-Flag und generiert ein Order-Vorschlags-Dictionary.
    """
    if not feature_flags.is_enabled("live_trading_enabled"):
        return {
            "error": "Live trading disabled — feature flag not enabled.",
            "status": "disabled",
        }

    audit_id = f"AUDIT-{uuid.uuid4().hex[:8]}"

    # Audit-Trail-Eintrag
    AuditTrail().log_decision(
        agent_id="live-signal",
        decision=f"signal_generated:{request.instrument}",
        actor="system",
        details={
            "event": "live_signal",
            "instrument": request.instrument,
            "analysis_time": (
                request.analysis_time.isoformat()
                if hasattr(request, "analysis_time")
                else str(request.analysis_time)
            ),
            "strategy": request.strategy,
            "audit_id": audit_id,
        },
    )

    # Generiere Order-Vorschlag (KEINE reale Order-Ausführung)
    strategy = getattr(request, "strategy", None) or {}
    instrument = request.instrument

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

    suggestion: dict[str, Any] = {
        "action": action,
        "quantity": round(abs(hash(f"{instrument}{request.analysis_time.isoformat()}")) % 100 + 1, 2),
        "price": round((abs(hash(f"price{instrument}")) % 500 + 10) + 0.01, 2),
        "confidence": confidence,
        "reasoning": reasoning,
    }

    return {
        "order_suggestion": suggestion,
        "status": "signal_generated",
        "message": f"Signal für {instrument} generiert (KEINE Ausführung).",
        "audit_id": audit_id,
    }