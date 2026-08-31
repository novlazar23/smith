"""Persistiert verarbeitete Marktdaten-Events in ClickHouse.

Der Sink empfängt die von ``DataIngestionService.consume`` verarbeiteten
Events und schreibt Candle-Events in die ``candles``-Tabelle. Alle Fehler
werden geloggt, nie weitergeworfen — der Consumer-Service bleibt dadurch
langlebig stabil.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from packages.persistence.clickhouse.engine import ClickHouseConfig, create_ch_engine

logger = logging.getLogger(__name__)

_TIMEFRAME = "1m"
_SOURCE = "dummy-adapter"
_DEFAULT_VENUE = "DUMMY_EXCHANGE"
_DT_FORMAT = "%Y-%m-%d %H:%M:%S"
_ONE_CANDLE_SPAN = timedelta(seconds=59)


class ClickHouseMarketDataSink:
    """Sichert Candle-Events in der ClickHouse-Tabelle ``candles``.

    Nicht-Candle-Events (z. B. Ticks) werden ignoriert. Die Verbindung
    wird über die bestehende ``ClickHouseEngine`` (HTTP-Interface mit
    Basic-Auth und ``X-ClickHouse-Database``-Header) hergestellt.
    """

    def __init__(self, config: ClickHouseConfig | None = None) -> None:
        """Initialisiert den Sink mit Config oder Umgebungswerten.

        Args:
            config: Optionale ClickHouseConfig; Defaults stammen aus den
                Umgebungsvariablen CH_HOST/CH_PORT/CH_DB/CH_PASSWORD.
        """
        self._config = config or ClickHouseConfig(
            host=os.environ.get("CH_HOST", "localhost"),
            port=int(os.environ.get("CH_PORT", "8123")),
            database=os.environ.get("CH_DB", "trading_events"),
            user="orchestra",
            password=os.environ.get("CH_PASSWORD", ""),
        )
        self._engine = create_ch_engine(self._config)

    def ensure_schema(self) -> None:
        """Stellt sicher, dass Datenbank und alle Tabellen existieren."""
        self._engine.create_tables()

    def persist(self, event: dict[str, Any]) -> bool:
        """Persistiert ein verarbeitetes Event in ClickHouse.

        Args:
            event: Verarbeitetes Event (Standard-Format des Processors).

        Returns:
            True, wenn das Event als Candle geschrieben wurde;
            False bei anderen Event-Typen oder Fehlern.
        """
        if event.get("type") != "candle":
            return False
        try:
            query = self._build_candle_insert(event)
        except Exception as exc:
            logger.warning("Candle-INSERT konnte nicht gebaut werden: %s", exc)
            return False
        try:
            self._engine._execute(query)
        except Exception as exc:
            logger.warning("ClickHouse-Persistierung fehlgeschlagen: %s", exc)
            return False
        return True

    def _build_candle_insert(self, event: dict[str, Any]) -> str:
        """Baut das INSERT-Statement für eine Candle."""
        open_dt = self._to_dt(event.get("open_time") or event.get("timestamp"))
        close_dt = self._to_dt(event.get("close_time"), fallback=open_dt + _ONE_CANDLE_SPAN)
        # DateTime-Werte müssen in ClickHouse-VALUES-Tupeln als
        # String-Literal (mit Anführungszeichen) übergeben werden.
        open_time = self._sql_str(open_dt.strftime(_DT_FORMAT))
        close_time = self._sql_str(close_dt.strftime(_DT_FORMAT))
        now = self._sql_str(datetime.now(tz=UTC).strftime(_DT_FORMAT))

        values = (
            self._sql_str(str(event["symbol"])),
            self._sql_str(str(event.get("venue", _DEFAULT_VENUE))),
            f"'{_TIMEFRAME}'",
            open_time,
            close_time,
            str(float(event["open"])),
            str(float(event["high"])),
            str(float(event["low"])),
            str(float(event["close"])),
            str(float(event["volume"])),
            str(int(float(event.get("trade_count", 0)))),
            str(int(bool(event.get("is_closed", True)))),
            "'{}'",
            f"'{_SOURCE}'",
            open_time,
            now,
            now,
            "0",
            "1",
            "1.0",
        )
        columns = (
            "instrument, venue, timeframe, open_time, close_time, "
            "open, high, low, close, volume, trade_count, is_closed, "
            "metadata, source, event_time, ingestion_time, availability_time, "
            "sequence, revision, quality"
        )
        return (
            f"INSERT INTO {self._config.database}.candles ({columns}) VALUES ({', '.join(values)})"
        )

    @staticmethod
    def _to_dt(value: object, fallback: datetime | None = None) -> datetime:
        """Konvertiert einen Zeitwert (datetime/ISO-String/Epoche) nach UTC."""
        base = fallback or datetime.now(tz=UTC)
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(value, tz=UTC)
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
            except ValueError:
                return base
        else:
            return base
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    @staticmethod
    def _sql_str(value: str) -> str:
        """Escapt einen Stringwert für ein ClickHouse-String-Literal."""
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
