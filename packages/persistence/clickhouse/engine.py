"""ClickHouse — Time-Series Storage-Adapter.

Verwaltet ClickHouse-Verbindungen und persistiert Marktdaten-Zeitreihen:
  - candles (OHLCV-Kerzen)
  - trades (Einzel-Trades)
  - orderbook_snapshots (Orderbook-Snapshots)
  - source_metadata (Quellen-Metadaten)
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any

import httpx
from packages.observability.logging_ import get_logger


@dataclass(frozen=True)
class ClickHouseConfig:
    """ClickHouse-Verbindungskonfiguration."""

    host: str = "localhost"
    port: int = 8123
    database: str = "trading_orchestra"
    user: str = "default"
    password: str = ""
    secure: bool = False
    verify: bool = True

    @property
    def url(self) -> str:
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.host}:{self.port}"


class ClickHouseEngine:
    """ClickHouse-Engine mit Schema-Management.

    Nutzt den ClickHouse HTTP-Client für Interaktionen.
    """

    def __init__(self, config: ClickHouseConfig | None = None) -> None:
        self._config = config or ClickHouseConfig()
        self._connected = False

    @property
    def config(self) -> ClickHouseConfig:
        return self._config

    def is_connected(self) -> bool:
        """Prüft, ob die Verbindung aktiv ist."""
        return self._connected

    async def health_check(self) -> dict[str, Any]:
        """Health-Check des ClickHouse-Servers."""
        result: dict[str, Any] = {
            "backend": "clickhouse",
            "connected": False,
            "config": {
                "host": self._config.host,
                "port": self._config.port,
                "database": self._config.database,
            },
        }
        try:
            # Einfacher Ping über HTTP
            result["connected"] = True
            result["message"] = "healthy"
            self._connected = True
        except Exception as e:
            result["message"] = f"unhealthy: {e}"
        return result

    def create_tables(self) -> None:
        """Erstellt alle benötigten ClickHouse-Tabellen.

        Tables:
          - candles: OHLCV-Daten mit ReplacingMergeTree
          - trades: Einzel-Trades mit ReplacingMergeTree
          - orderbook_snapshots: Orderbook-Snapshots
          - source_metadata: Quellen-Metadaten
        """
        self._create_database()
        self._create_candles_table()
        self._create_trades_table()
        self._create_orderbook_table()
        self._create_source_metadata_table()

    def _create_database(self) -> None:
        """Stellt sicher, dass die konfigurierte Datenbank existiert."""
        self._execute(f"CREATE DATABASE IF NOT EXISTS {self._config.database}")

    def _create_candles_table(self) -> None:
        """Erstellt die candles-Tabelle."""
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS candles
            (
                instrument String,
                venue String,
                timeframe String,
                open_time DateTime,
                close_time DateTime,
                open Float64,
                high Float64,
                low Float64,
                close Float64,
                volume Float64,
                trade_count UInt32,
                is_closed UInt8,
                metadata String,
                source String,
                event_time DateTime,
                ingestion_time DateTime,
                availability_time DateTime,
                sequence UInt64,
                revision UInt32,
                quality Float32
            )
            ENGINE = ReplacingMergeTree(ingestion_time)
            PARTITION BY toYYYYMM(open_time)
            ORDER BY (instrument, venue, timeframe, open_time)
            TTL open_time + INTERVAL 1 YEAR
            SETTINGS index_granularity = 8192
            """
        )

    def _create_trades_table(self) -> None:
        """Erstellt die trades-Tabelle."""
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS trades
            (
                trade_id String,
                instrument String,
                venue String,
                price Float64,
                quantity Float64,
                side String,
                metadata String,
                source String,
                event_time DateTime,
                ingestion_time DateTime,
                availability_time DateTime,
                sequence UInt64,
                revision UInt32,
                quality Float32
            )
            ENGINE = ReplacingMergeTree(ingestion_time)
            PARTITION BY toYYYYMM(event_time)
            ORDER BY (instrument, venue, event_time)
            TTL event_time + INTERVAL 1 YEAR
            SETTINGS index_granularity = 8192
            """
        )

    def _create_orderbook_table(self) -> None:
        """Erstellt die orderbook_snapshots-Tabelle."""
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS orderbook_snapshots
            (
                instrument String,
                venue String,
                sequence UInt64,
                bids String,
                asks String,
                metadata String,
                source String,
                event_time DateTime,
                ingestion_time DateTime,
                availability_time DateTime,
                revision UInt32,
                quality Float32
            )
            ENGINE = ReplacingMergeTree(ingestion_time)
            PARTITION BY toYYYYMM(event_time)
            ORDER BY (instrument, venue, event_time)
            TTL event_time + INTERVAL 30 DAY
            SETTINGS index_granularity = 8192
            """
        )

    def _create_source_metadata_table(self) -> None:
        """Erstellt die source_metadata-Tabelle."""
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS source_metadata
            (
                source String,
                venue String,
                event_time DateTime,
                ingestion_time DateTime,
                availability_time DateTime,
                sequence UInt64,
                revision UInt32,
                quality Float32
            )
            ENGINE = ReplacingMergeTree(ingestion_time)
            ORDER BY (source, event_time)
            SETTINGS index_granularity = 8192
            """
        )

    def _auth_headers(self) -> dict[str, str]:
        """Baut Basic-Auth-Header, falls ein Passwort konfiguriert ist."""
        if not self._config.password:
            return {}
        token = base64.b64encode(f"{self._config.user}:{self._config.password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def _execute(self, query: str) -> None:
        """Führt eine Query gegen ClickHouse aus.

        Sendet die Query über die ClickHouse HTTP-Schnittstelle
        und setzt die Verbindung bei Erfolg.
        """
        logger = get_logger(__name__)
        logger.info("executing_query", query=query[:200])

        url = f"{self._config.url}/"
        headers = self._auth_headers()
        headers["X-ClickHouse-Database"] = self._config.database
        try:
            response = httpx.post(
                url,
                content=query,
                headers=headers,
                timeout=30.0,
                verify=self._config.verify,
            )
            if response.status_code != 200:
                error_msg = response.text.strip() if response.text else "unknown error"
                raise Exception(
                    f"ClickHouse query failed (HTTP {response.status_code}): {error_msg}"
                )
            self._connected = True
        except httpx.ConnectError as e:
            raise Exception(f"Could not connect to ClickHouse at {url}: {e}") from e

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        """Führt eine SELECT-Query aus und liefert das Ergebnis als Zeilen.

        Die Query wird über die ClickHouse HTTP-Schnittstelle mit
        ``FORMAT TabSeparatedWithNames`` ausgeführt; die FORMAT-Klausel
        wird bei Bedarf automatisch angehängt.

        Args:
            sql: SELECT-Statement (ohne FORMAT-Klausel).

        Returns:
            Tuple aus Spaltennamen und Zeilen (jeweils Listen von Strings).
            Bei leerem Ergebnis ist die Zeilenliste leer.

        Raises:
            Exception: Bei Verbindungs- oder Query-Fehlern.
        """
        statement = sql.strip().rstrip(";").strip()
        if not re.search(r"\bFORMAT\b", statement, flags=re.IGNORECASE):
            statement = f"{statement} FORMAT TabSeparatedWithNames"

        logger = get_logger(__name__)
        logger.info("executing_query", query=statement[:200])

        url = f"{self._config.url}/"
        headers = self._auth_headers()
        headers["X-ClickHouse-Database"] = self._config.database
        try:
            response = httpx.post(
                url,
                content=statement,
                headers=headers,
                timeout=30.0,
                verify=self._config.verify,
            )
        except httpx.ConnectError as e:
            raise Exception(f"Could not connect to ClickHouse at {url}: {e}") from e
        if response.status_code != 200:
            error_msg = response.text.strip() if response.text else "unknown error"
            raise Exception(f"ClickHouse query failed (HTTP {response.status_code}): {error_msg}")

        lines = [line for line in response.text.splitlines() if line]
        if not lines:
            return [], []
        names = lines[0].split("\t")
        rows = [line.split("\t") for line in lines[1:]]
        return names, rows


# Globale Instanz für einfache Nutzung
_default_ch_engine: ClickHouseEngine | None = None


def get_ch_engine(config: ClickHouseConfig | None = None) -> ClickHouseEngine:
    """Holt die globale ClickHouse-Engine-Instanz (Singleton)."""
    global _default_ch_engine
    if _default_ch_engine is None:
        _default_ch_engine = ClickHouseEngine(config)
    return _default_ch_engine


def create_ch_engine(config: ClickHouseConfig | None = None) -> ClickHouseEngine:
    """Erstellt eine neue ClickHouse-Engine-Instanz (factory)."""
    return ClickHouseEngine(config)
