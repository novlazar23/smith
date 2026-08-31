"""Einstiegspunkt für den News-Ingestion-Service.

Startet die Endlosschleife mit periodischen RSS-Zyklen.

Aufruf: ``python -m apps.news_ingestion [--tick 30] [--once]``
"""

from __future__ import annotations

import argparse
import logging

from apps.news_ingestion.service import HEARTBEAT_PATH, run_forever

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parst die Kommandozeilen-Argumente."""
    parser = argparse.ArgumentParser(description="News-Ingestion-Service")
    parser.add_argument("--tick", type=float, default=30.0, help="Sekunden zwischen Zyklen")
    parser.add_argument("--once", action="store_true", help="Nur einen Zyklus ausführen")
    return parser.parse_args()


def main() -> None:
    """Startet den News-Ingestion-Service."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    HEARTBEAT_PATH.touch()
    args = _parse_args()
    logger.info("news-ingestion started (tick=%.1fs, once=%s)", args.tick, args.once)
    run_forever(tick_seconds=args.tick, once=args.once)


if __name__ == "__main__":
    main()
