"""Thread-sicherer InfluxDB-Client-Wrapper mit In-Memory-Fallback.

Quant-Plattform Phase 1 (P1-4): kapselt den Zugriff auf InfluxDB hinter
einer kleinen, testbaren Schnittstelle für die Shadow-Trading-Integration.

Muster aus dem Bestand:
- Lazy Connection + Fallback-Zustand wie ``services/db.py`` (``_ensure_pool``).
- Re-entrant Lock für Thread-Safety wie ``services/kill_switch.py``;
  Netzwerk-I/O läuft außerhalb des Locks.

Das Paket ``influxdb_client`` wird lazy importiert (erst beim ersten
Verbindungsversuch), damit die Basis-Smith-Runtime ohne das optionale
``[quant]``-Extra weiterläuft. Bei InfluxDB-Ausfall (Paket fehlt, Server
unerreichbar, Ping fehlerhaft, Schreib-/Lese-Fehler) degradiert der Store
stets zum In-Memory-Modus statt Exceptions zu werfen:
Schreiben puffert im Buffer, Abfragen liefern leere Listen.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.query_api import QueryApi
    from influxdb_client.client.write_api import WriteApi

logger = logging.getLogger(__name__)

FieldDict = dict[str, float | int | str]


class InfluxDBStore:
    """Thread-sicherer InfluxDB-Client mit In-Memory-Fallback.

    - Lazy Connection: der Client wird erst beim ersten Aufruf aufgebaut.
    - Fallback: bei InfluxDB-Ausfall landen Writes im In-Memory-Buffer
      und Queries liefern leere Listen — niemals eine Exception.
    - Thread-Safe: alle Zustandszugriffe laufen unter einem RLock;
      Netzwerk-I/O außerhalb des Locks.
    - Async-freundlich: die synchronen influxdb-client-Aufrufe laufen im
      Default-Executor, damit der Event Loop des Shadow-Trading-Loops
      nicht blockiert wird.
    """

    def __init__(self, url: str, token: str, org: str, bucket: str) -> None:
        """Initialisiert den Client. Verbindung wird lazy aufgebaut."""
        self._url = url
        self._token = token
        self._org = org
        self._bucket = bucket
        self._client: InfluxDBClient | None = None
        self._write_api: WriteApi | None = None
        self._query_api: QueryApi | None = None
        self._ready = False
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Zustand
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True wenn InfluxDB erreichbar und verbunden."""
        with self._lock:
            return self._client is not None and self._ready

    def buffer_size(self) -> int:
        """Anzahl gepufferter Punkte bei InfluxDB-Ausfall."""
        with self._lock:
            return len(self._buffer)

    # ------------------------------------------------------------------
    # Öffentliche Async-API
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Prüft InfluxDB-Verbindung. Gibt True bei Erfolg."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_health_check)

    async def write_points(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: FieldDict,
        timestamp: int,
    ) -> None:
        """Schreibt einen einzelnen Punkt. Bei InfluxDB-Ausfall: In-Memory-Buffer."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._sync_write_point, measurement, tags, fields, timestamp
        )

    async def write_batch(
        self,
        measurement: str,
        tags: dict[str, str],
        points: list[FieldDict],
        timestamps: list[int],
    ) -> None:
        """Schreibt mehrere Punkte effizient. Bei InfluxDB-Ausfall: In-Memory-Buffer."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._sync_write_batch, measurement, tags, points, timestamps
        )

    async def query(self, flux: str) -> list[dict]:
        """Führt eine Flux-Query aus. Bei InfluxDB-Ausfall: leere Liste."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_query, flux)

    # ------------------------------------------------------------------
    # Sync-Kern (läuft im Executor / separaten Threads)
    # ------------------------------------------------------------------

    def _sync_health_check(self) -> bool:
        """Verbindungsprüfung im Thread-Kontext (wirft nie)."""
        self._ensure_client()
        return self.is_available

    def _sync_write_point(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: FieldDict,
        timestamp: int,
    ) -> None:
        """Schreibt einen Punkt; bei Ausfall puffert er im Memory-Buffer."""
        self._ensure_client()
        if self._client is None or self._write_api is None:
            self._buffer_point(measurement, tags, fields, timestamp)
            return
        point = self._build_point(measurement, tags, fields, timestamp)
        try:
            self._write_api.write(bucket=self._bucket, record=point)
        except Exception:
            logger.warning(
                "InfluxDB write failed (%s) — point buffered in memory",
                measurement,
                exc_info=True,
            )
            self._mark_unavailable()
            self._buffer_point(measurement, tags, fields, timestamp)

    def _sync_write_batch(
        self,
        measurement: str,
        tags: dict[str, str],
        points: list[FieldDict],
        timestamps: list[int],
    ) -> None:
        """Schreibt alle Punkte in einem Write-Aufruf; bei Ausfall: Buffer."""
        if len(points) != len(timestamps):
            raise ValueError("points and timestamps must have the same length")
        self._ensure_client()
        if self._client is None or self._write_api is None:
            self._buffer_points(measurement, tags, points, timestamps)
            return
        records = [
            self._build_point(measurement, tags, fields, ts)
            for fields, ts in zip(points, timestamps, strict=True)
        ]
        try:
            self._write_api.write(bucket=self._bucket, record=records)
        except Exception:
            logger.warning(
                "InfluxDB batch write failed (%s) — %d points buffered in memory",
                measurement,
                len(points),
                exc_info=True,
            )
            self._mark_unavailable()
            self._buffer_points(measurement, tags, points, timestamps)

    def _sync_query(self, flux: str) -> list[dict]:
        """Führt die Flux-Query aus; bei Ausfall: leere Liste."""
        self._ensure_client()
        if self._client is None or self._query_api is None:
            return []
        try:
            tables = self._query_api.query(flux)
        except Exception:
            logger.warning("InfluxDB query failed — returning empty result", exc_info=True)
            self._mark_unavailable()
            return []
        results: list[dict] = []
        for table in tables:
            for record in table.records:
                results.append(dict(record.values))
        return results

    # ------------------------------------------------------------------
    # Verbindungsaufbau (lazy, Muster aus services/db.py)
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        """Baut den InfluxDB-Client bei Bedarf auf (lazy).

        Alle Fehlerzustände konvergieren in den In-Memory-Fallback;
        Exceptions werden nie an den Aufrufer weitergereicht.
        """
        with self._lock:
            if self._ready:
                return
            try:
                self._connect()
            except ImportError:
                logger.warning(
                    "influxdb-client package not installed — using in-memory fallback"
                )
            except Exception:
                logger.warning(
                    "InfluxDB unavailable at %s — using in-memory fallback",
                    self._url,
                    exc_info=True,
                )

    def _connect(self) -> None:
        """Einmaliger Verbindungsversuch; hinterlässt Fallback-Zustand bei Fehler.

        ``influxdb_client`` wird hier lazy importiert, damit das Modul
        auch ohne das optionale ``[quant]``-Extra importierbar bleibt.
        """
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS

        client = InfluxDBClient(url=self._url, token=self._token, org=self._org)
        connected = False
        try:
            connected = client.ping()
            if connected:
                self._client = client
                self._write_api = client.write_api(write_options=SYNCHRONOUS)
                self._query_api = client.query_api()
                self._ready = True
        finally:
            if not connected:
                try:
                    client.close()
                except Exception:
                    # Best effort — der Client war nie verbunden, Close-Fehler sind trivial.
                    logger.exception("InfluxDB client close failed")
                self._mark_unavailable()

    def _mark_unavailable(self) -> None:
        """Setzt den Verbindungszustand auf In-Memory-Fallback zurück."""
        with self._lock:
            self._client = None
            self._write_api = None
            self._query_api = None
            self._ready = False

    # ------------------------------------------------------------------
    # Punkt-Bau + Buffer
    # ------------------------------------------------------------------

    @staticmethod
    def _build_point(
        measurement: str, tags: dict[str, str], fields: FieldDict, timestamp: int
    ) -> Point:
        """Baut einen InfluxDB-Point mit Nanosekunden-Timestamp."""
        from influxdb_client import Point

        point = Point(measurement)
        for tag_key, tag_value in tags.items():
            point = point.tag(tag_key, tag_value)
        for field_key, field_value in fields.items():
            point = point.field(field_key, field_value)
        return point.time(timestamp, write_precision="ns")

    @staticmethod
    def _buffer_entry(
        measurement: str, tags: dict[str, str], fields: FieldDict, timestamp: int
    ) -> dict[str, Any]:
        """Erzeugt einen Buffer-Eintrag (Kopien, kein Aliasing auf Aufrufer-Data)."""
        return {
            "measurement": measurement,
            "tags": dict(tags),
            "fields": dict(fields),
            "timestamp": timestamp,
        }

    def _buffer_point(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: FieldDict,
        timestamp: int,
    ) -> None:
        """Puffert einen einzelnen Punkt im In-Memory-Buffer."""
        with self._lock:
            self._buffer.append(self._buffer_entry(measurement, tags, fields, timestamp))

    def _buffer_points(
        self,
        measurement: str,
        tags: dict[str, str],
        points: list[FieldDict],
        timestamps: list[int],
    ) -> None:
        """Puffert alle Punkte eines Batches im In-Memory-Buffer."""
        with self._lock:
            for fields, ts in zip(points, timestamps, strict=True):
                self._buffer.append(self._buffer_entry(measurement, tags, fields, ts))
