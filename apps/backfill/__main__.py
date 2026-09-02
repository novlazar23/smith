"""CLI-Einstiegspunkt für den Candle-Backfill (Binance Futures → ClickHouse).

Aufruf:

    python -m apps.backfill --months 12 --instruments BTC/USDT,ETH/USDT
    python -m apps.backfill --start 2025-01-01 --end 2025-12-31 --dry-run

Der Backfill lädt historische 1m-Kerzen von der Binance-Futures-REST-API
und schreibt sie idempotent in die ClickHouse-Tabelle ``candles_history``
(Venue ``BINANCE_FUTURES``, Instrument im kanonischen "BTC/USDT"-Format).
Die Tabelle wird bei Bedarf automatisch angelegt (gleiche Spalten wie
``candles``, aber ohne TTL — dedizierte Backtest-Tabelle, deren Daten
sonst von der 1-Jahre-TTL der Live-Tabelle entfernt würden).
Bereits vorhandene Zeiträume werden übersprungen — ein zweiter Lauf lädt
nur die fehlenden Lücken (Deduplication über ReplacingMergeTree).

ClickHouse-Zugang über die Umgebungsvariablen ``CH_HOST``/``CH_PORT``/
``CH_DB``/``CH_PASSWORD`` (Compose-Defaults: ``clickhouse``/``8123``/
``trading_events``), überschreibbar per ``--ch-*``-Flags.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime

from apps.backfill import storage
from apps.backfill.client import KlineClient
from apps.backfill.service import BackfillConfig, BackfillService
from apps.orchestrator_service.service import parse_instruments
from packages.ingestion.adapter.binance import BINANCE_FUTURES_VENUE
from packages.persistence.clickhouse.engine import ClickHouseConfig, create_ch_engine

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Baut den Argument-Parser des Backfill-CLIs."""
    parser = argparse.ArgumentParser(
        prog="python -m apps.backfill",
        description=(
            "Backfillt historische Binance-Futures-1m-Kerzen idempotent "
            "in die ClickHouse-Tabelle candles_history (Venue BINANCE_FUTURES)."
        ),
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Historische Tiefe in Monaten, wenn --start fehlt (Default: 12)",
    )
    parser.add_argument(
        "--instruments",
        default="BTC/USDT,ETH/USDT",
        help="Komma-getrennte Instrumente (Default: BTC/USDT,ETH/USDT)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Erster Tag (inklusive), Format YYYY-MM-DD, UTC",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Letzter Tag (inklusive), Format YYYY-MM-DD, UTC",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Plan ausgeben (ohne Downloads, ohne ClickHouse-Schreibzugriffe)",
    )
    parser.add_argument(
        "--ch-host",
        default=os.environ.get("CH_HOST", "clickhouse"),
        help="ClickHouse-Host (Default: $CH_HOST oder 'clickhouse')",
    )
    parser.add_argument(
        "--ch-port",
        type=int,
        default=int(os.environ.get("CH_PORT", "8123")),
        help="ClickHouse-HTTP-Port (Default: $CH_PORT oder 8123)",
    )
    parser.add_argument(
        "--ch-db",
        default=os.environ.get("CH_DB", "trading_events"),
        help="ClickHouse-Datenbank (Default: $CH_DB oder 'trading_events')",
    )
    parser.add_argument(
        "--ch-password",
        default=os.environ.get("CH_PASSWORD", ""),
        help="ClickHouse-Passwort (Default: $CH_PASSWORD)",
    )
    return parser


def _parse_day(value: str, *, at_day_end: bool = False) -> datetime:
    """Parst ein YYYY-MM-DD-Datum als UTC-Datetime mit Tagsgrenzen.

    Args:
        value: Datum im Format YYYY-MM-DD.
        at_day_end: ``True`` → Tagesende (23:59:00, letzter 1m-Tick des
            Tages, inclusive), ``False`` → Tagesanfang (00:00:00).

    Returns:
        UTC-datetime der Tagsgrenze.
    """
    day = date.fromisoformat(value)
    if at_day_end:
        return datetime(day.year, day.month, day.day, 23, 59, 0, tzinfo=UTC)
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI-Einstiegspunkt.

    Args:
        argv: Argumente (Default: ``sys.argv[1:]``).

    Returns:
        Exit-Code: 0 = Erfolg, 1 = Lauffehler (z. B. ClickHouse nicht
        erreichbar, Download-Fehler), 2 = ungültige Aufrufparameter.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        start = None if args.start is None else _parse_day(args.start)
        end = None if args.end is None else _parse_day(args.end, at_day_end=True)
    except ValueError as exc:
        logger.error("Ungültiges Datumsformat (erwartet YYYY-MM-DD): %s", exc)
        return 2
    if args.months < 1:
        logger.error("--months muss >= 1 sein (erhalten: %d)", args.months)
        return 2
    if start is not None and end is not None and start >= end:
        logger.error("--start muss vor --end liegen")
        return 2
    instruments = parse_instruments(args.instruments)
    if not instruments:
        logger.error("Keine Instrumente angegeben (--instruments leer)")
        return 2

    config = BackfillConfig(
        months=args.months,
        instruments=instruments,
        start=start,
        end=end,
        dry_run=args.dry_run,
    )
    engine = create_ch_engine(
        ClickHouseConfig(
            host=args.ch_host,
            port=args.ch_port,
            database=args.ch_db,
            user="orchestra",
            password=args.ch_password,
        )
    )
    try:
        engine.query("SELECT 1")
    except Exception as exc:
        logger.error(
            "ClickHouse nicht erreichbar unter http://%s:%s/%s: %s",
            args.ch_host,
            args.ch_port,
            args.ch_db,
            exc,
        )
        return 1
    try:
        storage.ensure_table(engine)
    except Exception as exc:
        logger.error("candles_history nicht anlegbar: %s", exc)
        return 1

    logger.info(
        "Backfill gestartet: months=%d instruments=%s venue=%s dry_run=%s",
        args.months,
        list(instruments),
        BINANCE_FUTURES_VENUE,
        args.dry_run,
    )
    with KlineClient() as client:
        service = BackfillService(config, client, engine)
        try:
            result = service.run()
        except Exception as exc:
            logger.exception("Backfill fehlgeschlagen: %s", exc)
            return 1

    for summary in result.summaries:
        if summary.total_after is not None:
            logger.info(
                "Zusammenfassung %s: %s → %s, %d Kerzen geladen, %d Zeilen in candles_history",
                summary.instrument,
                summary.start.strftime("%Y-%m-%d %H:%M"),
                summary.end.strftime("%Y-%m-%d %H:%M"),
                summary.fetched_candles,
                summary.total_after,
            )
    if result.failures:
        logger.error(
            "Backfill mit Fehlern beendet: %s", [name for name, _ in result.failures]
        )
        return 1
    logger.info("Backfill abgeschlossen (dry_run=%s)", args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
