"""CLI-Einstiegspunkt für den Demo-Trader (virtuelles Trading).

Aufruf: ``python -m apps.demo_trader``

Der Service führt die getrackte ``OrchestratorPipeline`` periodisch für
alle konfigurierten Instrumente aus — mit einem ACTIVEn Ensemble, damit
der gewichtete Konsens echte Entscheidungen liefert. Die Entscheidung
wird als Paper-Trade (long-only) über den ``PaperExecutor`` ausgeführt
(imaginäres Geld, Slippage und Kommission inklusive); es werden **nie**
reale Orders platziert. Jeder Trade landet in ``demo_trades``, der
Account-Snapshot in ``demo_account``.
"""

from __future__ import annotations

import logging
import sys
import threading

from apps.demo_trader.service import (
    build_trader,
    config_from_env,
    install_signal_handlers,
    run_service,
)

logger = logging.getLogger(__name__)


def main() -> int:
    """Startet den Demo-Trader-Loop.

    Returns:
        Exit-Code (0 = normaler Ablauf, 2 = ungültige Konfiguration).
    """
    config = config_from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not config.instruments:
        logger.error("Keine Instrumente konfiguriert (DEMO_INSTRUMENTS leer)")
        return 2

    trader = build_trader(config=config)
    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    logger.info(
        "Demo-Trader gestartet: instruments=%s interval=%.0fs venue=%s "
        "initial_cash=%.2f trade_notional=%.2f min_confidence=%.2f (Paper-Trading, imaginäres Geld)",
        list(config.instruments),
        config.interval_seconds,
        config.candle_venue,
        config.initial_cash,
        config.trade_notional,
        config.min_confidence,
    )
    logger.info(
        "Paper-Konto '%s' wird in-Process gehalten — bei einem Neustart setzt es auf "
        "initial_cash zurück (Trade-Historie bleibt in demo_trades erhalten)",
        config.account_id,
    )
    run_service(trader, stop_event.is_set)
    logger.info("Demo-Trader beendet (graceful)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
