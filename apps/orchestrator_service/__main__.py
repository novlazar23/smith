"""CLI-Einstiegspunkt für den Orchestrator-Service (Shadow-Modus).

Aufruf: ``python -m apps.orchestrator_service``

Der Service führt die getrackte ``OrchestratorPipeline`` periodisch für
alle konfigurierten Instrumente aus — ausschließlich Analyse, Konsens und
Audit. Order-Ausführung findet **nie** statt, unabhängig vom
Feature-Flag ``live_trading_enabled`` (dessen Zustand wird pro Zyklus
zum Audit protokolliert).
"""

from __future__ import annotations

import logging
import sys
import threading

from apps.orchestrator_service.service import (
    OrchestratorService,
    build_service,
    config_from_env,
    install_signal_handlers,
    run_service,
)

logger = logging.getLogger(__name__)


def main() -> int:
    """Startet den Orchestrator-Service-Loop.

    Returns:
        Exit-Code (0 = normaler Ablauf, 2 = ungültige Konfiguration).
    """
    config = config_from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not config.instruments:
        logger.error("Keine Instrumente konfiguriert (ORCHESTRATOR_INSTRUMENTS leer)")
        return 2

    service: OrchestratorService = build_service(config=config)
    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    logger.info(
        "Orchestrator-Service gestartet: instruments=%s interval=%.0fs horizon=%s "
        "agent_status=%s (keine Order-Ausführung)",
        list(config.instruments),
        config.interval_seconds,
        config.horizon,
        config.agent_status,
    )
    run_service(service, stop_event.is_set)
    logger.info("Orchestrator-Service beendet (graceful)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
