"""ClickHouse-Schreibseite des Backfills (Tabelle ``candles_history``).

Backfill-Tabelle ``candles_history``: gleiche Spalten wie die Live-Tabelle
``candles``, aber **ohne TTL** — die Tabelle ist eine dedizierte
Backtest-Datenbank, deren Umfang ausschließlich durch die gewählten
Backfill-Fenster bestimmt wird. Eine TTL wäre hier ein Fallstrick:
Szenario-Daten aus 2021 sind bereits > 5 Jahre alt und würden von
ClickHouse-Merges direkt nach dem Einfügen wieder entfernt.

Dedup-Entscheidung — aus der DDL in ``packages/persistence/clickhouse/engine.py``:

    ENGINE = ReplacingMergeTree(ingestion_time)
    ORDER BY (instrument, venue, timeframe, open_time)

Duplikate mit identischem ``(instrument, venue, timeframe, open_time)``
mergt ClickHouse automatisch — überlebt die Zeile mit der neuesten
``ingestion_time``. Daraus folgt die Backfill-Strategie:

- **Reiner INSERT-Workflow.** Ein erneuter Lauf inseriert dieselben
  Zeitschlüssel erneut; der frische ``ingestion_time``-Stempel ersetzt
  die alten Zeilen beim Merge — idempotent, ohne DELETE.
- ``delete_range()`` ist aus diesem Grund ein **dokumentierter No-Op**:
  Ein DELETE vor dem INSERT wäre redundant und würde bei einem
  abgebrochenen Backfill zusätzlich bereits vorhandene (ggf. vom
  Market-Producer live geschriebene) Zeilen des Fensters entfernen.

Das Zeilen-Format entspricht dem laufenden Ingestion-Sink
(``apps/ingestion/persistence.py``): gleiche Spalten, DateTime-Werte als
``'YYYY-MM-DD HH:MM:SS'``-Strings (UTC), ``close_time`` = ``open_time`` +
59 s (letzte Sekunde der Kerze). ``source`` markiert den Ursprung als
``backfill``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from apps.backfill.client import BackfillCandle
from packages.persistence.clickhouse.engine import ClickHouseConfig

logger = logging.getLogger(__name__)

_DT_FORMAT = "%Y-%m-%d %H:%M:%S"
_FMT_FORMAT = "%Y-%m-%d %H:%M"
_TIMEFRAME = "1m"
_SOURCE = "backfill"
_ONE_CANDLE_SPAN = timedelta(seconds=59)
INSERT_BATCH_SIZE = 5000
TABLE_NAME = "candles_history"

_TABLE_DDL = """CREATE TABLE IF NOT EXISTS {table}
(
    `instrument` String,
    `venue` String,
    `timeframe` String,
    `open_time` DateTime,
    `close_time` DateTime,
    `open` Float64,
    `high` Float64,
    `low` Float64,
    `close` Float64,
    `volume` Float64,
    `trade_count` UInt32,
    `is_closed` UInt8,
    `metadata` String,
    `source` String,
    `event_time` DateTime,
    `ingestion_time` DateTime,
    `availability_time` DateTime,
    `sequence` UInt64,
    `revision` UInt32,
    `quality` Float32
)
ENGINE = ReplacingMergeTree(ingestion_time)
PARTITION BY toYYYYMM(open_time)
ORDER BY (instrument, venue, timeframe, open_time)
SETTINGS index_granularity = 8192"""


def ensure_table(engine: CandleEngine) -> None:
    """Stellt sicher, dass ``candles_history`` existiert (idempotent, ohne TTL).

    Gleiche Spalten wie die Live-Tabelle ``candles``, aber **ohne TTL**
    (dedizierte Backtest-Tabelle — Details im Modul-Docstring). Falls eine
    ältere Version der Tabelle mit TTL existiert, wird die TTL entfernt
    (nur, wenn vorhanden — ``REMOVE TTL`` ohne bestehende TTL ist ein
    Hard-Error in ClickHouse), damit alte Szenario-Fenster nicht von
    Merges entfernt werden.
    """
    table = f"{engine.config.database}.{TABLE_NAME}"
    engine._execute(_TABLE_DDL.format(table=table))
    _names, rows = engine.query(
        "SELECT create_table_query FROM system.tables "
        f"WHERE database = {_sql_str(engine.config.database)} AND name = {_sql_str(TABLE_NAME)}"
    )
    if rows and rows[0] and "TTL" in rows[0][0]:
        engine._execute(f"ALTER TABLE {table} REMOVE TTL")
_COLUMNS = (
    "instrument, venue, timeframe, open_time, close_time, "
    "open, high, low, close, volume, trade_count, is_closed, "
    "metadata, source, event_time, ingestion_time, availability_time, "
    "sequence, revision, quality"
)


class CandleEngine(Protocol):
    """Duck-typ-Schnittstelle der ClickHouse-Engine für den Backfill.

    Erfüllt von ``packages.persistence.clickhouse.engine.ClickHouseEngine``
    (HTTP-Interface mit ``query()`` für SELECTs und ``_execute()`` für
    INSERTs — identisch genutzt vom Ingestion-Sink).
    """

    config: ClickHouseConfig

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        """Führt eine SELECT-Query aus (TabSeparatedWithNames)."""
        ...

    def _execute(self, sql: str) -> None:
        """Führt eine Nicht-SELECT-Query (z. B. INSERT) aus."""
        ...


def existing_range(
    engine: CandleEngine,
    instrument: str,
    venue: str,
) -> tuple[datetime, datetime] | None:
    """Liefert das vorhandene Kerzenfenster ``(min, max)`` von ``open_time``.

    Args:
        engine: ClickHouse-Engine.
        instrument: Kanonisches Instrument (z. B. ``"BTC/USDT"``).
        venue: Venue (z. B. ``"BINANCE_FUTURES"``).

    Returns:
        Tuple ``(min_open_time, max_open_time)`` als UTC-datetimes oder
        ``None``, wenn das ``(Instrument, Venue)`` noch keine Kerzen hat.
    """
    sql = (
        "SELECT min(open_time) AS min_open_time, max(open_time) AS max_open_time "
        f"FROM {TABLE_NAME} "
        f"WHERE instrument = {_sql_str(instrument)} AND venue = {_sql_str(venue)}"
    )
    names, rows = engine.query(sql)
    if not rows:
        return None
    index = {name: i for i, name in enumerate(names)}
    min_dt = _parse_dt(rows[0][index["min_open_time"]])
    max_dt = _parse_dt(rows[0][index["max_open_time"]])
    if min_dt is None or max_dt is None:
        return None
    return min_dt, max_dt


def existing_day_coverage(
    engine: CandleEngine,
    instrument: str,
    venue: str,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, datetime, datetime, int]]:
    """Liefert die pro-Tage-Abdeckung des Fensters ``[start, end]``.

    Pro Tag mit Daten ein Tuple ``(day, first, last, count)``:
    ``day`` = Tagbeginn 00:00 UTC, ``first``/``last`` = erste/letzte
    ``open_time`` des Tags, ``count`` = Anzahl **distinkter** Minuten
    (``uniqExact`` — Duplikate vor dem ReplacingMergeTree-Merge zählen
    nicht mit). Tage ohne Daten fehlen in der Liste.

    Args:
        engine: ClickHouse-Engine.
        instrument: Kanonisches Instrument (z. B. ``"BTC/USDT"``).
        venue: Venue (z. B. ``"BINANCE_FUTURES"``).
        start: Fensterbeginn (inklusive).
        end: Fensterende (inklusive).

    Returns:
        Liste in Zeitreihenfolge; leer, wenn das Fenster keine Daten hat.
    """
    sql = (
        "SELECT toStartOfDay(open_time) AS d, min(open_time) AS mn, "
        "max(open_time) AS mx, uniqExact(open_time) AS n "
        f"FROM {TABLE_NAME} "
        f"WHERE instrument = {_sql_str(instrument)} AND venue = {_sql_str(venue)} "
        f"AND open_time BETWEEN '{_fmt(start)}' AND '{_fmt(end)}' "
        "GROUP BY d ORDER BY d"
    )
    names, rows = engine.query(sql)
    if not rows:
        return []
    index = {name: i for i, name in enumerate(names)}
    result: list[tuple[datetime, datetime, datetime, int]] = []
    for row in rows:
        day = _parse_dt(row[index["d"]])
        first = _parse_dt(row[index["mn"]])
        last = _parse_dt(row[index["mx"]])
        count = int(row[index["n"]])
        if day is None or first is None or last is None:
            continue
        result.append((day, first, last, count))
    return result


def existing_minutes(
    engine: CandleEngine,
    instrument: str,
    venue: str,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    """Liefert alle ``open_time``-Werte im Fenster ``[start, end]``.

    Nur für Teil-Tage (wenige tausend Zeilen) gedacht: ermöglicht die
    exakte Lückenbestimmung innerhalb eines unvollständigen Tages.

    Args:
        engine: ClickHouse-Engine.
        instrument: Kanonisches Instrument.
        venue: Venue.
        start: Fensterbeginn (inklusive).
        end: Fensterende (inklusive).

    Returns:
        Liste von UTC-datetimes (nicht sortiert, kann Duplikate enthalten).
    """
    sql = (
        "SELECT open_time "
        f"FROM {TABLE_NAME} "
        f"WHERE instrument = {_sql_str(instrument)} AND venue = {_sql_str(venue)} "
        f"AND open_time BETWEEN '{_fmt(start)}' AND '{_fmt(end)}'"
    )
    names, rows = engine.query(sql)
    if not rows:
        return []
    index = {name: i for i, name in enumerate(names)}
    result: list[datetime] = []
    for row in rows:
        value = _parse_dt(row[index["open_time"]])
        if value is not None:
            result.append(value)
    return result


def delete_range(
    engine: CandleEngine,
    instrument: str,
    venue: str,
    start: datetime,
    end: datetime,
) -> None:
    """Entfernt Kerzen aus einem Zeitfenster — hier aus dok. Gründen ein No-Op.

    ``candles`` ist ``ReplacingMergeTree(ingestion_time)`` mit ORDER BY
    ``(instrument, venue, timeframe, open_time)``: Neu-INSERTs derselben
    Zeitschlüssel ersetzen alte Zeilen beim Merge (neueste
    ``ingestion_time`` gewinnt). Ein DELETE ist daher für die
    Idempotenz nicht erforderlich — Details im Modul-Docstring.

    Args:
        engine: ClickHouse-Engine (hier nicht benötigt, API-Vollständigkeit).
        instrument: Kanonisches Instrument.
        venue: Venue.
        start: Fensterbeginn.
        end: Fensterende.
    """
    logger.debug(
        "delete_range(%s, %s, %s → %s): keine DELETE-Query — "
        "ReplacingMergeTree(ingestion_time) merge-t Duplikate automatisch",
        instrument,
        venue,
        _fmt(start),
        _fmt(end),
    )


def count_candles(engine: CandleEngine, instrument: str, venue: str) -> int:
    """Zählt die Kerzen eines ``(Instrument, Venue)`` (für die Final-Summary).

    Args:
        engine: ClickHouse-Engine.
        instrument: Kanonisches Instrument.
        venue: Venue.

    Returns:
        Anzahl der Zeilen in ``candles_history`` für das Paar.
    """
    sql = (
        f"SELECT count() AS total FROM {TABLE_NAME} "
        f"WHERE instrument = {_sql_str(instrument)} AND venue = {_sql_str(venue)}"
    )
    _names, rows = engine.query(sql)
    if not rows or not rows[0][0]:
        return 0
    return int(rows[0][0])


def insert_candles(engine: CandleEngine, rows: Sequence[BackfillCandle]) -> int:
    """Schreibt Kerzen in ``candles_history`` (``INSERT ... VALUES``-Batches à 5000).

    Args:
        engine: ClickHouse-Engine.
        rows: Zu schreibende Kerzen.

    Returns:
        Anzahl der insertierten Zeilen.
    """
    now = datetime.now(UTC)
    inserted = 0
    for offset in range(0, len(rows), INSERT_BATCH_SIZE):
        batch = rows[offset : offset + INSERT_BATCH_SIZE]
        values = ", ".join(_format_row(row, now) for row in batch)
        sql = f"INSERT INTO {engine.config.database}.{TABLE_NAME} ({_COLUMNS}) VALUES {values}"
        engine._execute(sql)
        inserted += len(batch)
    return inserted


def _format_row(candle: BackfillCandle, now: datetime) -> str:
    """Formatiert eine Kerze als ClickHouse-VALUES-Tupel (20 Spalten)."""
    open_s = candle.open_time.strftime(_DT_FORMAT)
    close_s = (candle.open_time + _ONE_CANDLE_SPAN).strftime(_DT_FORMAT)
    now_s = now.strftime(_DT_FORMAT)
    return (
        f"({_sql_str(candle.instrument)}, {_sql_str(candle.venue)}, '{_TIMEFRAME}', "
        f"'{open_s}', '{close_s}', "
        f"{candle.open}, {candle.high}, {candle.low}, {candle.close}, {candle.volume}, "
        f"0, 1, '{{}}', '{_SOURCE}', "
        f"'{open_s}', '{now_s}', '{now_s}', 0, 1, 1.0)"
    )


def _parse_dt(value: str) -> datetime | None:
    """Parst ein ClickHouse-DateTime-Feld als UTC-datetime.

    Empty Strings (z. B. ``min()`` über leerer Menge) liefern ``None``.
    """
    if not value:
        return None
    return datetime.strptime(value, _DT_FORMAT).replace(tzinfo=UTC)


def _fmt(moment: datetime) -> str:
    """Formatiert einen Zeitstempel für Log-Nachrichten."""
    return moment.strftime(_FMT_FORMAT)


def _sql_str(value: str) -> str:
    """Escapt einen String für ein ClickHouse-String-Literal."""
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"
