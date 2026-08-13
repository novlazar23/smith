"""Endpunkt-Logik für die Trading Orchestra API.

Die Endpunkt-Funktionen sind von der FastAPI-App getrennt, um die Logik
klar zu halten und unit-tests zu erleichtern.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from packages.config.instrument_pool import InstrumentPool
from packages.governance.audit import AuditTrail
from packages.governance.feature_flags import feature_flags
from packages.orchestration.batch_engine import BatchEngine

logger = logging.getLogger(__name__)

# Optional FastAPI imports — only used by batch endpoint
try:
    from fastapi import HTTPException, status as http_status  # noqa: F401
except ImportError:
    HTTPException = None  # type: ignore[assignment]
    http_status = None  # type: ignore[assignment]


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


async def batch_analysis_endpoint(request: Any) -> dict[str, Any]:  # noqa: ANN401
    """Verarbeitet eine Batch-Analyse-Anfrage.

    Nimmt eine Liste von Instrumenten entgegen und analysiert sie
    im Batch mit gemeinsamem Feature-Computing und Ressourcen-Monitoring.

    Args:
        request: BatchAnalyzeRequest mit instruments, horizons, strategy.

    Returns:
        Dictionary mit instrument_results, shared_features,
        resource_metrics und total_time_seconds.
    """
    instruments: list[str] = request.instruments
    horizons: list[str] = getattr(request, "horizons", ["15m", "4h", "1d"])
    strategy: dict[str, Any] = getattr(request, "strategy", {})

    logger.info(
        "Processing batch analysis for %d instruments, horizons=%s, strategy_keys=%s",
        len(instruments),
        horizons,
        list(strategy.keys()),
    )

    # Erstelle Pool und Engine mit Defaults
    pool = InstrumentPool()
    engine = BatchEngine(pool)

    # Fuege Instrumente zum Pool hinzu
    try:
        pool.add_instruments(instruments)
    except ValueError as exc:
        raise HTTPException(  # type: ignore[call-arg, unused-ignore]
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,  # type: ignore[union-attr, unused-ignore]
            detail=f"Invalid instruments: {exc}",
        )

    # Fuehre Batch-Analyse aus
    result = engine.execute_batch(instruments, horizons, strategy)

    # Falls teilweise oder fehlgeschlagen, HTTP 200 mit Status im Body
    logger.info(
        "Batch analysis completed with status: %s, %d instruments processed",
        result.get("status"),
        len(result.get("instrument_results", [])),
    )

    return result