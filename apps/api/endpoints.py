"""Endpunkt-Logik für die Trading Orchestra API.

Die Endpunkt-Funktionen sind von der FastAPI-App getrennt, um die Logik
klar zu halten und unit-tests zu erleichtern.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
from packages.config.instrument_pool import InstrumentPool
from packages.governance.audit import AuditTrail
from packages.governance.feature_flags import feature_flags
from packages.orchestration.batch_engine import BatchEngine
from packages.persistence.sqlalchemy.engine import DatabaseConfig, SQLAlchemyEngine
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Optional FastAPI imports — only used by batch endpoint
try:
    from fastapi import HTTPException
    from fastapi import status as http_status
except ImportError:
    HTTPException = None  # type: ignore[assignment]
    http_status = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from apps.api.routers.live_signal import LiveSignalRequest


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


# ---------------------------------------------------------------------------
# /status — Systemzustand mit echten Probes
# ---------------------------------------------------------------------------

# Zeitpunkt des Modul-Imports für die Uptime-Berechnung
_START = time.monotonic()

# Maximale Dauer eines einzelnen Probes in Sekunden
_PROBE_TIMEOUT_SECONDS = 3.0

# Module, deren Verfügbarkeit im Status gemeldet wird
_STATUS_MODULES: tuple[str, ...] = (
    "packages.orchestration.batch_engine",
    "packages.persistence.sqlalchemy",
    "apps.news_ingestion.scheduler",
    "apps.market_producer.producer",
    "apps.ingestion.consumer",
    "confluent_kafka",
)

# Standard-Antwort für einen ausgefallenen Datenbank-Probe
_DB_DOWN: dict[str, Any] = {"connected": False, "latency_ms": None}


def _check_module_available(module_name: str) -> str:
    """Prüft, ob ein Modul importiert werden kann."""
    try:
        __import__(module_name, fromlist=[""])
        return "ready"
    except Exception:
        return "unavailable"


def _module_map() -> dict[str, str]:
    """Baut die Verfügbarkeits-Karte aller Status-Module."""
    return {name: _check_module_available(name) for name in _STATUS_MODULES}


def _database_config() -> DatabaseConfig:
    """Liest die PostgreSQL-Konfiguration aus der Umgebung."""
    return DatabaseConfig(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "5432")),
        database=os.environ.get("DB_NAME", "trading"),
        user=os.environ.get("DB_USER", "orchestra"),
        password=os.environ.get("DB_PASSWORD", ""),
    )


def _probe_postgres() -> dict[str, Any]:
    """Prüft PostgreSQL per SELECT 1 und misst die Latenz in Millisekunden."""
    start = time.monotonic()
    try:
        engine = SQLAlchemyEngine(_database_config()).engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"connected": True, "latency_ms": round((time.monotonic() - start) * 1000.0, 2)}
    except Exception:
        return dict(_DB_DOWN)


def _probe_clickhouse() -> dict[str, Any]:
    """Prüft ClickHouse per /ping-Endpunkt und misst die Latenz in Millisekunden."""
    start = time.monotonic()
    host = os.environ.get("CH_HOST", "localhost")
    port = os.environ.get("CH_PORT", "8123")
    try:
        response = httpx.get(f"http://{host}:{port}/ping", timeout=_PROBE_TIMEOUT_SECONDS)
        if response.status_code != 200:
            return dict(_DB_DOWN)
        return {"connected": True, "latency_ms": round((time.monotonic() - start) * 1000.0, 2)}
    except Exception:
        return dict(_DB_DOWN)


def _probe_redpanda() -> bool:
    """Prüft Redpanda per TCP-Verbindung zum ersten Server aus REDPANDA_SERVERS."""
    servers = os.environ.get("REDPANDA_SERVERS", "localhost:9092")
    first = servers.split(",")[0].strip()
    host, _, port = first.partition(":")
    try:
        with socket.create_connection((host, int(port or "9092")), timeout=_PROBE_TIMEOUT_SECONDS):
            return True
    except Exception:
        return False


def _count_candles() -> int | None:
    """Zählt Kerzen in ClickHouse (None bei Fehlern)."""
    host = os.environ.get("CH_HOST", "localhost")
    port = os.environ.get("CH_PORT", "8123")
    database = os.environ.get("CH_DB", "trading_events")
    user = os.environ.get("CH_USER", "orchestra")
    password = os.environ.get("CH_PASSWORD", "")
    try:
        # ClickHouse setzt TCP-Verbindungen für POSTs auf Database-Pfade zurück —
        # Root-URL plus X-ClickHouse-Database-Header ist das funktionierende Muster.
        response = httpx.post(
            f"http://{host}:{port}/",
            content=b"SELECT count() FROM candles",
            headers={"X-ClickHouse-Database": database},
            auth=(user, password),
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return None
        return int(response.text.strip().splitlines()[0])
    except Exception:
        return None


def _count_news_events() -> int | None:
    """Zählt News-Events in PostgreSQL (None, wenn z.B. die Tabelle fehlt)."""
    try:
        engine = SQLAlchemyEngine(_database_config()).engine
        with engine.connect() as conn:
            return int(conn.execute(text("SELECT count(*) FROM news_events")).scalar_one())
    except Exception:
        return None


def _feature_flag_map() -> dict[str, bool]:
    """Liefert alle existierenden Feature-Flags mit ihrem Zustand."""
    try:
        return feature_flags.get_all_flags()
    except Exception:
        return {}


async def _run_probe[T](fn: Callable[[], T], default: T) -> T:
    """Führt einen blockierenden Probe in einem Thread mit Zeitlimit aus.

    Bei Timeout oder unerwartetem Fehler wird der Default-Wert zurückgegeben,
    damit der Status-Endpunkt nie hängt oder mit 500 antwortet.
    """
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn), timeout=_PROBE_TIMEOUT_SECONDS)
    except Exception:
        return default


async def status_endpoint() -> dict[str, Any]:
    """Gibt den Systemstatus mit echten Probes zurück.

    Untersucht Modul-Verfügbarkeit, PostgreSQL, ClickHouse und Redpanda
    sowie Zähler und Feature-Flags. Die Antwort enthält zusätzlich zu den
    alten Keys (version, status, uptime_seconds, modules, timestamp) die
    Blöcke database, streaming und feature_flags.
    """
    postgres, clickhouse, redpanda, candles, news, modules = await asyncio.gather(
        _run_probe(_probe_postgres, dict(_DB_DOWN)),
        _run_probe(_probe_clickhouse, dict(_DB_DOWN)),
        _run_probe(_probe_redpanda, default=False),
        _run_probe(_count_candles, None),
        _run_probe(_count_news_events, None),
        _run_probe(_module_map, {}),
    )

    all_connected = postgres["connected"] and clickhouse["connected"] and redpanda
    return {
        "version": "0.1.0",
        "status": "running" if all_connected else "degraded",
        "uptime_seconds": round(time.monotonic() - _START, 2),
        "modules": modules,
        "database": {"postgres": postgres, "clickhouse": clickhouse},
        "streaming": {
            "redpanda": {"connected": redpanda},
            "candles_1h": candles,
            "news_events_total": news,
        },
        "feature_flags": _feature_flag_map(),
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def health_endpoint() -> dict[str, Any]:
    """Gibt einen einfachen Health-Check zurück."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def live_signal_endpoint(request: LiveSignalRequest) -> dict[str, Any]:
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
        ) from exc

    # Fuehre Batch-Analyse aus
    result = engine.execute_batch(instruments, horizons, strategy)

    # Falls teilweise oder fehlgeschlagen, HTTP 200 mit Status im Body
    logger.info(
        "Batch analysis completed with status: %s, %d instruments processed",
        result.get("status"),
        len(result.get("instrument_results", [])),
    )

    return result
