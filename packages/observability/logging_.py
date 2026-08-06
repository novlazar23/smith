"""Logging — Strukturiertes Logging mit structlog und JSON-Output.

Stellt eine zentralisierte Logging-Schicht bereit mit:
- JSON-Formatierung für Container-Orchestrierung
- Kontext-Verfolgung (correlation_id, run_id, instrument)
- Integrationspunkte für Prometheus und Tracing
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog


def setup_logging(
    *,
    level: str = "INFO",
    json: bool = True,
    timestamp_key: str = "timestamp",
) -> None:
    """Initialisiert structlog mit den gegebenen Konfigurationen.

    Args:
        level: Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json: JSON-Formatierung aktivieren (für Produktion empfohlen).
        timestamp_key: Schlüssel für Zeitstempel im Log-Output.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", key=timestamp_key),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Gibt einen strukturierten Logger zurück.

    Args:
        name: Logger-Name (typischerweise __name__).

    Returns:
        Konfigurierter structlog-Logger.
    """
    return structlog.get_logger(name or "trading-orchestra")


@contextmanager
def log_context(**context: Any) -> Generator[None, None, None]:
    """Kontextmanager zum Hinzufügen von temporärem Logging-Kontext.

    Alle Logs innerhalb des Context-Managers enthalten die angegebenen
    Schlüssel-Wert-Paare.

    Args:
        **context: Schlüssel-Wert-Paare für den Log-Kontext.

    Example:
        >>> with log_context(run_id="abc-123", instrument="BTC/USDT"):
        ...     logger.info("processing_event")
        {"event": "processing_event", "run_id": "abc-123", "instrument": "BTC/USDT"}
    """
    from structlog.contextvars import bind_contextvars, clear_contextvars

    try:
        bind_contextvars(**context)
        yield
    finally:
        clear_contextvars()
