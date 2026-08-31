"""CLI-Einstiegspunkt fuer den Market-Data-Producer.

Aufruf: ``python -m apps.market_producer``
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from apps.market_producer.producer import (
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_TOPIC,
    DummyMarketDataProducer,
)

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT"
DEFAULT_SERVERS = "redpanda:9092"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parst die Kommandozeilen-Argumente.

    Args:
        argv: Argumentliste (default: sys.argv[1:]).

    Returns:
        Geparste Argumente.
    """
    parser = argparse.ArgumentParser(
        prog="market_producer",
        description="Produziert synthetische Candle-Events auf ein Redpanda-Topic.",
    )
    parser.add_argument(
        "--symbols",
        default=os.environ.get("MARKET_PRODUCER_SYMBOLS", DEFAULT_SYMBOLS),
        help="Kommagetrennte Instrumente (default: BTC/USDT,ETH/USDT)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Tick-Abstand in Sekunden (default: 60)",
    )
    parser.add_argument(
        "--servers",
        default=os.environ.get("REDPANDA_SERVERS", DEFAULT_SERVERS),
        help="Redpanda Bootstrap-Server (default: redpanda:9092)",
    )
    parser.add_argument(
        "--topic",
        default=os.environ.get("MARKET_DATA_TOPIC", DEFAULT_TOPIC),
        help="Ziel-Topic (default: market_data)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log-Level (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Startet den Producer-Loop.

    Args:
        argv: Argumentliste (default: sys.argv[1:]).

    Returns:
        Exit-Code (0 = normaler Ablauf).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("Keine Symbole konfiguriert")
        return 2
    producer = DummyMarketDataProducer(
        symbols=symbols,
        bootstrap_servers=args.servers,
        topic=args.topic,
        interval_seconds=args.interval,
    )
    try:
        asyncio.run(producer.run())
    except KeyboardInterrupt:
        logger.info("Abbruch signalisiert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
