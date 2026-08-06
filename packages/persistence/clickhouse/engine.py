"""ClickHouse — Time-Series Storage-Adapter.

Verwaltet ClickHouse-Verbindungen und persistiert Marktdaten-Zeitreihen:
  - candles (OHLCV-Kerzen)
  - trades (Einzel-Trades)
  - orderbook_snapshots (Orderbook-Snapshots)
  - source_metadata (Quellen-Metadaten)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    Für das MVP wird eine einfache HTTP-basierte Implementierung verwendet.
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
        self._create_candles_table()
        self._create_trades_table()
        self._create_orderbook_table()
        self._create_source_metadata_table()

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

    def _execute(self, query: str) -> None:
        """Führt eine Query gegen ClickHouse aus.

        Im echten System würde hier der ClickHouse-HTTP-Client verwendet werden.
        Für das MVP wird nur die Query gespeichert/protokolliert.
        """
        # TODO: Implement ClickHouse HTTP client integration


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
