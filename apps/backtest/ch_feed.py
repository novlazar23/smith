"""ClickHouse-Datenfeed für den Backtest (implementiert ``DataFeed``).

Liest die ``candles_history``-Tabelle (historische Backfill-Daten mit
5-Jahre-TTL, siehe ``apps.backfill.storage.ensure_table``) über eine
injizierbare Engine (Default im CLI:
die reale ClickHouse-Engine aus ``CH_HOST``/``CH_PORT``/``CH_DB``/
``CH_PASSWORD``, wie im Demo-Trader) und liefert die Kerzen als
``packages.backtesting.core.Candle`` (UTC, aufsteigend).

Die Abfrage entspricht dem Lese-Pattern des ``DemoCandleProvider``
(instrument/venue-Filter, Escaping der String-Literale); zusätzlich wird ein
Zeitfenster (``start``/``end``) gefiltert. ``resample="5m"`` aggregiert 1m-
Kerzen in-Memory auf 5-Minuten-UTC-Grenzen (open=first, high=max, low=min,
close=last, volume=sum; ein unvollständiger leading Bucket wird verworfen).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Protocol

from packages.backtesting.core import Candle
from packages.backtesting.datafeed import DataFeed

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("open_time", "open", "high", "low", "close", "volume")
TABLE_NAME = "candles_history"


class QueryEngine(Protocol):
    """Duck-Typ-Schnittstelle einer ClickHouse-Engine (query → Spalten + Zeilen)."""

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        """Führt eine SELECT-Query aus und liefert (Spaltennamen, Zeilen)."""
        ...


def _escape(value: str) -> str:
    """Escapt einen String für ein ClickHouse-String-Literal (wie DemoCandleProvider)."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _to_utc(value: datetime | date | str | None) -> datetime | None:
    """Normiert eine Zeitgrenze auf ein UTC-datetime (None bleibt None)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    else:
        moment = datetime.fromisoformat(str(value))
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _normalize_start(value: datetime | date | str | None) -> datetime | None:
    """Startgrenze: Kalendertage beginnen um 00:00:00 UTC."""
    return _to_utc(value)


def _normalize_end(value: datetime | date | str | None) -> datetime | None:
    """Endgrenze: Kalendertage umfassen den ganzen Tag (23:59:59 UTC)."""
    moment = _to_utc(value)
    return None if moment is None else moment + timedelta(seconds=86_399)


def _format_bound(moment: datetime) -> str:
    """Formatiert eine Zeitgrenze als ClickHouse-DateTime-Literal."""
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _parse_timestamp(value: str) -> datetime:
    """Parst ein ClickHouse-DateTime-Feld ("YYYY-MM-DD HH:MM:SS") nach UTC."""
    text = str(value).strip()
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        moment = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def resample_to_5m(candles: list[Candle]) -> list[Candle]:
    """Aggregiert 1m-Kerzen auf 5m-Kerzen, ausgerichtet an UTC-5-Minuten-Grenzen.

    open=erste, high=max, low=min, close=letzte, volume=summe. Ein leading
    Bucket, der nicht exakt auf einer 5-Minuten-Grenze beginnt, ist
    unvollständig und wird verworfen.
    """
    if not candles:
        return []
    groups: dict[datetime, list[Candle]] = {}
    for candle in candles:
        boundary = candle.timestamp.replace(
            minute=(candle.timestamp.minute // 5) * 5, second=0, microsecond=0
        )
        groups.setdefault(boundary, []).append(candle)
    boundaries = sorted(groups)
    if groups[boundaries[0]][0].timestamp != boundaries[0]:
        boundaries = boundaries[1:]
    result: list[Candle] = []
    for boundary in boundaries:
        bucket = groups[boundary]
        result.append(
            Candle(
                timestamp=boundary,
                symbol=candles[0].symbol,
                open=bucket[0].open,
                high=max(c.high for c in bucket),
                low=min(c.low for c in bucket),
                close=bucket[-1].close,
                volume=sum(c.volume for c in bucket),
            )
        )
    return result


class ClickHouseDataFeed(DataFeed):
    """DataFeed, das Kerzen aus ClickHouse liest (instrument/venue/Zeitfenster).

    Die Kerzen werden beim ersten Zugriff geladen und gecacht, damit mehrere
    Engine-Runs (z.B. Gate-Sweep) dieselbe Datenbasis nutzen und ClickHouse
    nur einmal befragt wird.
    """

    def __init__(
        self,
        engine: QueryEngine,
        instrument: str,
        venue: str = "BINANCE_FUTURES",
        start: datetime | date | str | None = None,
        end: datetime | date | str | None = None,
        resample: str | None = None,
    ) -> None:
        """Initialisiert den Feed.

        Args:
            engine: ClickHouse-Engine (oder Duck-Typ-Ersatz in Tests).
            instrument: Handelspaar, z.B. ``"BTC/USDT"``.
            venue: Venue-Filter (Default ``BINANCE_FUTURES``).
            start: Untere Zeitgrenze (datetime/date/ISO-String; None = ohne).
            end: Obere Zeitgrenze (Kalendertage umfassen den ganzen Tag).
            resample: ``None`` oder ``"5m"`` (1m→5m-Aggregation).
        """
        if resample is not None and resample != "5m":
            raise ValueError(f"resample muss '5m' oder None sein, ist {resample!r}")
        self._engine = engine
        self._instrument = instrument
        self._venue = venue
        self._start = _normalize_start(start)
        self._end = _normalize_end(end)
        self._resample = resample
        self._candles: list[Candle] | None = None

    @property
    def symbol(self) -> str:
        """Das Instrument dieses Feeds."""
        return self._instrument

    def get_candles(self, symbol: str | None = None) -> list[Candle]:
        """Liefert alle Kerzen des Feeds (gecacht; fremde Symbole → leer)."""
        if symbol is not None and symbol != self._instrument:
            return []
        if self._candles is None:
            self._candles = self._fetch()
            logger.info(
                "ClickHouse-Feed geladen: %s/%s %s→%s (%d Kerzen, resample=%s)",
                self._instrument,
                self._venue,
                _format_bound(self._start) if self._start else "—",
                _format_bound(self._end) if self._end else "—",
                len(self._candles),
                self._resample or "none",
            )
        return list(self._candles)

    def iter_candles(self, symbol: str | None = None) -> Iterator[Candle]:
        """Yieldet die Kerzen des Feeds (chronologisch)."""
        yield from self.get_candles(symbol)

    def _build_sql(self) -> str:
        """Baut die SELECT-Query (instrument/venue/Zeitfenster, aufsteigend)."""
        conditions = [
            f"instrument = '{_escape(self._instrument)}'",
            f"venue = '{_escape(self._venue)}'",
        ]
        if self._start is not None and self._end is not None:
            conditions.append(
                f"open_time BETWEEN '{_format_bound(self._start)}' AND '{_format_bound(self._end)}'"
            )
        elif self._start is not None:
            conditions.append(f"open_time >= '{_format_bound(self._start)}'")
        elif self._end is not None:
            conditions.append(f"open_time <= '{_format_bound(self._end)}'")
        return (
            "SELECT open_time, open, high, low, close, volume "
            f"FROM {TABLE_NAME} "
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY open_time"
        )

    def _fetch(self) -> list[Candle]:
        """Lädt und konvertiert die Kerzen aus ClickHouse."""
        names, rows = self._engine.query(self._build_sql())
        index = {name: i for i, name in enumerate(names)}
        missing = [column for column in REQUIRED_COLUMNS if column not in index]
        if missing:
            raise ValueError(f"ClickHouse-Abfrage liefert fehlende Spalten: {missing}")
        candles = [
            Candle(
                timestamp=_parse_timestamp(row[index["open_time"]]),
                symbol=self._instrument,
                open=float(row[index["open"]]),
                high=float(row[index["high"]]),
                low=float(row[index["low"]]),
                close=float(row[index["close"]]),
                volume=float(row[index["volume"]]),
            )
            for row in rows
        ]
        candles.sort(key=lambda candle: candle.timestamp)
        if self._resample == "5m":
            candles = resample_to_5m(candles)
        return candles
