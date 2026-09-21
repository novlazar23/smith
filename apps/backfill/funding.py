"""Binance-Futures-Funding-Rate-Backfill (Tabelle ``funding_rates``).

Synchroner, paginierter Client für den öffentlichen ``GET /fundingRate``-
Endpunkt der Binance-Futures-REST-API — exakt das Gegenstück zum Kline-
Backfill (``apps/backfill/client.py``), nur mit Funding-Settlements statt
1m-Kerzen. Basis-URL und Venue übernimmt er aus dem bestehenden
``BinanceAdapter`` (``packages/ingestion/adapter/binance.py``).

Funding-Settlements fallen dreimal täglich pro Symbol an (00:00/08:00/
16:00 UTC) — die Historie ist damit winzig (seit 2019 ≈ 7 Seiten à 1000
Zeilen pro Symbol). Deshalb **kein** Gap-Planner: ein Lauf lädt das volle
Fenster, und die Deduplication über ``ReplacingMergeTree(ingestion_time)``
macht Re-Läufe idempotent (frische ``ingestion_time`` gewinnt beim Merge).

Dedup-Entscheidung (aus der DDL unten):

    ENGINE = ReplacingMergeTree(ingestion_time)
    ORDER BY (instrument, venue, funding_time)

Duplikate mit identischem ``(instrument, venue, funding_time)`` mergt
ClickHouse automatisch. Reiner INSERT-Workflow, kein DELETE — wie beim
Kline-Backfill (``apps/backfill/storage.py``).

Retries/Paging folgen ``KlineClient``: HTTP 429/418/5xx → ``Retry-After``
honorieren, sonst exponentieller Backoff (1 s, 2 s, 4 s), max. 3 Retries;
Cursor-Pagination in 1000-Zeilen-Fenstern (``limit=1000``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

import httpx
from apps.backfill import storage
from apps.backfill.client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_DELAY,
    RATE_LIMIT_STATUSES,
    RETRYABLE_STATUSES,
    BinanceAPIError,
    BinanceRateLimitError,
)
from apps.market_producer.producer import to_exchange_symbol
from packages.ingestion.adapter.binance import (
    BINANCE_FUTURES_BASE_URL,
    BINANCE_FUTURES_VENUE,
)

logger = logging.getLogger(__name__)

FUNDING_TABLE_NAME = "funding_rates"
FUNDING_PAGE_SIZE = 1000
INSERT_BATCH_SIZE = 5000

_DT_FORMAT = "%Y-%m-%d %H:%M:%S"
_FMT_FORMAT = "%Y-%m-%d %H:%M"

# Exakte DDL (ohne TTL) — ``{db}`` wird zur Laufzeit ersetzt.
_FUNDING_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS {db}.funding_rates (instrument String, venue String, "
    "funding_time DateTime, funding_rate Float64, mark_price Float64, event_time DateTime, "
    "ingestion_time DateTime) ENGINE = ReplacingMergeTree(ingestion_time) "
    "PARTITION BY toYYYYMM(funding_time) ORDER BY (instrument, venue, funding_time) "
    "SETTINGS index_granularity = 8192"
)

_FUNDING_COLUMNS = (
    "instrument, venue, funding_time, funding_rate, mark_price, "
    "event_time, ingestion_time"
)


@dataclass(frozen=True)
class FundingRateRow:
    """Eine auf das ClickHouse-Format gemappte Funding-Rate."""

    instrument: str
    venue: str
    funding_time: datetime
    funding_rate: float
    mark_price: float


def map_funding_row(instrument: str, venue: str, row: Mapping[str, Any]) -> FundingRateRow:
    """Mappt eine Roh-Funding-Zeile der Binance-API auf eine FundingRateRow.

    Args:
        instrument: Kanonisches Instrument (z. B. ``"BTC/USDT"``).
        venue: Venue (z. B. ``"BINANCE_FUTURES"``).
        row: Roh-Zeile aus der API-Antwort mit ``fundingTime`` (ms),
            ``fundingRate`` (String) und optional ``markPrice`` (String).

    Returns:
        Gemappte Funding-Rate; ``funding_time`` als UTC-datetime.
        ``markPrice`` fehlt in Historie-Antworten oft → dann ``0.0``.
    """
    mark = row.get("markPrice")
    return FundingRateRow(
        instrument=instrument,
        venue=venue,
        funding_time=_ms_to_dt(int(row["fundingTime"])),
        funding_rate=float(row["fundingRate"]),
        mark_price=float(mark) if mark not in (None, "") else 0.0,
    )


def _dt_to_ms(moment: datetime) -> int:
    """Konvertiert ein UTC-datetime auf die Millisekunde-Ebene der API."""
    return int(moment.timestamp()) * 1000


def _ms_to_dt(ms: int) -> datetime:
    """Konvertiert einen Millisekunden-Zeitstempel zu einem UTC-datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


class FundingRateClient:
    """Paginierter Funding-Rate-Client für Binance Futures (synchron, httpx).

    Lädt Funding-Settlements von ``start`` bis ``end`` per Cursor-Pagination
    in 1000-Zeilen-Fenstern (``limit=1000``). Prozeß-sicher: Re-Läufe
    inserieren dieselben ``(instrument, venue, funding_time)`` erneut;
    ``ReplacingMergeTree`` dedupliziert.
    """

    def __init__(
        self,
        base_url: str = BINANCE_FUTURES_BASE_URL,
        venue: str = BINANCE_FUTURES_VENUE,
        *,
        request_delay: float = DEFAULT_REQUEST_DELAY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialisiert den Client.

        Args:
            base_url: Binance-Futures-Basis-URL (inkl. ``/fapi/v1``).
            venue: Venue-Stempel für die gemappten Funding-Raten.
            request_delay: Pause zwischen Requests in Sekunden.
            max_retries: Max. Retry-Anzahl bei 429/418/5xx (Default 3).
            timeout: HTTP-Timeout in Sekunden.
            transport: Optionaler httpx-Transport (Test-Doppel).
            sleep: Warte-Funktion (Default ``time.sleep``, in Tests injizierbar).
        """
        self._base_url = base_url
        self._venue = venue
        self._request_delay = request_delay
        self._max_retries = max_retries
        self._timeout = timeout
        self._sleep = sleep
        self._client: httpx.Client | None = (
            httpx.Client(base_url=base_url, transport=transport, timeout=timeout)
            if transport is not None
            else None
        )

    def fetch_page(self, symbol: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        """Holt eine Funding-Seite (max. 1000 Zeilen) mit Retry-Policy.

        Args:
            symbol: Binance-Symbol (z. B. ``"BTCUSDT"``).
            start_ms: Fensterbeginn in Millisekunden (inklusive).
            end_ms: Fensterende in Millisekunden (inklusive).

        Returns:
            Roh-Funding-Zeilen des Fensters (aufsteigend nach ``fundingTime``).

        Raises:
            BinanceAPIError: Bei nicht retrybaren HTTP-Fehlern (4xx) oder
                5xx nach ``max_retries`` Retries.
            BinanceRateLimitError: Bei 429/418 nach ``max_retries`` Retries.
        """
        params = {
            "symbol": symbol,
            "startTime": str(start_ms),
            "endTime": str(end_ms),
            "limit": str(FUNDING_PAGE_SIZE),
        }
        for attempt in range(self._max_retries + 1):
            response = self._ensure_client().get("/fundingRate", params=params)
            if response.status_code in RETRYABLE_STATUSES:
                if attempt >= self._max_retries:
                    raise self._retry_error(response, symbol)
                delay = self._retry_delay(response, attempt)
                logger.warning(
                    "Funding HTTP %d (%s) — Retry %d/%d in %.1fs",
                    response.status_code,
                    symbol,
                    attempt + 1,
                    self._max_retries,
                    delay,
                )
                self._sleep(delay)
                continue
            if response.status_code != 200:
                raise BinanceAPIError(
                    f"Funding HTTP {response.status_code} ({symbol}): {response.text[:200]}"
                )
            payload = response.json()
            if not isinstance(payload, list):
                raise BinanceAPIError(f"Funding-Antwort ist kein Array ({symbol})")
            return payload
        raise BinanceAPIError(f"Funding-Request nicht ausgeführt ({symbol})")

    def fetch_range(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
        on_chunk: Callable[[datetime, datetime, int], None] | None = None,
    ) -> list[FundingRateRow]:
        """Lädt alle Funding-Sätze von ``start`` bis ``end`` (beides inklusive).

        Cursor-Pagination: Cursor startet bei ``start``; nach jeder Seite
        rückt er auf ``fundingTime`` der letzten Zeile + 1 ms vor (nur, wenn
        die Seite exakt 1000 Zeilen hatte — sonst ist das Ende erreicht).
        Zwischen den Requests wird ``request_delay`` Sekunden gewartet.

        Args:
            instrument: Kanonisches Instrument (z. B. ``"BTC/USDT"``).
            start: Fensterbeginn (UTC, inklusive).
            end: Fensterende (UTC, inklusive).
            on_chunk: Optionaler Progress-Callback pro Seite
                (erste/letzte ``funding_time``, Sätzezahl der Seite).

        Returns:
            Alle Funding-Sätze des Fensters (aufsteigend nach ``funding_time``).
        """
        symbol = to_exchange_symbol(instrument)
        start_ms = _dt_to_ms(start)
        end_ms = _dt_to_ms(end)
        rows: list[FundingRateRow] = []
        cursor = start_ms
        while True:
            if cursor > start_ms:
                self._sleep(self._request_delay)
            page = self.fetch_page(symbol, cursor, end_ms)
            chunk = [
                map_funding_row(instrument, self._venue, raw)
                for raw in page
                if _ms_to_dt(int(raw["fundingTime"])) <= end
            ]
            if on_chunk is not None:
                if chunk:
                    on_chunk(chunk[0].funding_time, chunk[-1].funding_time, len(chunk))
                else:
                    on_chunk(_ms_to_dt(cursor), _ms_to_dt(end_ms), 0)
            rows.extend(chunk)
            if len(page) != FUNDING_PAGE_SIZE:
                break
            cursor = int(page[-1]["fundingTime"]) + 1
        return rows

    def close(self) -> None:
        """Schließt den HTTP-Client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> FundingRateClient:
        """Kontextmanager: liefert den Client selbst."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Kontextmanager: schließt den HTTP-Client."""
        del exc_type, exc, tb
        self.close()

    def _ensure_client(self) -> httpx.Client:
        """Liefert (bei Bedarf lazily erstellten) httpx-Client."""
        if self._client is None:
            self._client = httpx.Client(base_url=self._base_url, timeout=self._timeout)
        return self._client

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Berechnet die Backoff-Pause vor dem Retry ``attempt`` (0-basiert).

        Exponentiell: 1 s, 2 s, 4 s. Bei 429/418 schlägt ein gültiger
        ``Retry-After``-Header (in Sekunden) den Backoff.
        """
        if response.status_code in RATE_LIMIT_STATUSES:
            try:
                retry_after = float(response.headers.get("Retry-After", ""))
            except ValueError:
                retry_after = -1.0
            if retry_after > 0:
                return retry_after
        return 2.0 ** attempt

    def _retry_error(self, response: httpx.Response, symbol: str) -> Exception:
        """Erzeugt den Fehler, wenn die Retries bei 429/418/5xx erschöpft sind."""
        detail = f"Funding HTTP {response.status_code} ({symbol}) nach {self._max_retries} Retries"
        if response.status_code in RATE_LIMIT_STATUSES:
            return BinanceRateLimitError(f"{detail}: Rate-Limit nicht aufgehoben")
        return BinanceAPIError(f"{detail}: {response.text[:200]}")


def ensure_funding_table(engine: storage.CandleEngine) -> None:
    """Stellt sicher, dass ``funding_rates`` existiert (idempotent, ohne TTL).

    Dedizierte Backtest-Tabelle — wie ``candles_history`` **ohne TTL**, damit
    alte Funding-Settlements (2019) nicht von ClickHouse-Merges entfernt
    werden. Reiner ``CREATE TABLE IF NOT EXISTS``-Workflow.
    """
    engine._execute(_FUNDING_TABLE_DDL.format(db=engine.config.database))


def insert_funding_rates(engine: storage.CandleEngine, rows: Sequence[FundingRateRow]) -> int:
    """Schreibt Funding-Raten in ``funding_rates`` (``INSERT ... VALUES``-Batches à 5000).

    Args:
        engine: ClickHouse-Engine.
        rows: Zu schreibende Funding-Raten.

    Returns:
        Anzahl der insertierten Zeilen.
    """
    now = datetime.now(UTC)
    inserted = 0
    for offset in range(0, len(rows), INSERT_BATCH_SIZE):
        batch = rows[offset : offset + INSERT_BATCH_SIZE]
        values = ", ".join(_format_funding_row(row, now) for row in batch)
        sql = (
            f"INSERT INTO {engine.config.database}.{FUNDING_TABLE_NAME} "
            f"({_FUNDING_COLUMNS}) VALUES {values}"
        )
        engine._execute(sql)
        inserted += len(batch)
    return inserted


def count_funding_rates(engine: storage.CandleEngine, instrument: str, venue: str) -> int:
    """Zählt die Funding-Sätze eines ``(Instrument, Venue)`` (für die Summary).

    Args:
        engine: ClickHouse-Engine.
        instrument: Kanonisches Instrument (z. B. ``"BTC/USDT"``).
        venue: Venue (z. B. ``"BINANCE_FUTURES"``).

    Returns:
        Anzahl der Zeilen in ``funding_rates`` für das Paar.
    """
    sql = (
        f"SELECT count() AS total FROM {FUNDING_TABLE_NAME} "
        f"WHERE instrument = {_sql_str(instrument)} AND venue = {_sql_str(venue)}"
    )
    _names, rows = engine.query(sql)
    if not rows or not rows[0][0]:
        return 0
    return int(rows[0][0])


def refresh_funding(
    engine: storage.CandleEngine,
    client: FundingRateClient,
    instruments: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    venue: str = "BINANCE_FUTURES",
) -> int:
    """Lädt die Funding-Historie aller Instrumente und schreibt sie idempotent.

    Pro Instrument: ``fetch_range`` (volles Fenster) + ``insert_funding_rates``.
    Ein Fehler bei einem Instrument beendet nicht den Lauf; das Instrument
    wird **einmal automatisch wiederholt** (mirrors ``BackfillService.run``);
    erst nach dem zweiten Fehlschlag steht es in der Fehlerliste.

    Args:
        engine: ClickHouse-Engine.
        client: Funding-Rate-Client (Binance-Futures).
        instruments: Kanonische Instrumente (z. B. ``"BTC/USDT"``).
        start: Fensterbeginn (UTC, inklusive).
        end: Fensterende (UTC, inklusive).
        venue: Venue-Label (Default ``BINANCE_FUTURES``).

    Returns:
        Gesamtzahl der insertierten Zeilen.

    Raises:
        RuntimeError: Wenn ein Instrument nach dem Retry fehlschlägt
            (mit Liste der fehlgeschlagenen Instrumente).
    """
    # ponytail: kein Gap-Planner — Funding-Historie ist winzig (3 Sätze/Tag/
    # Symbol → ≈ 7 Seiten seit 2019); Vollfenster-Load + ReplacingMergeTree-
    # Dedup ist das Idempotenz-Modell. Upgrade-Pfad: Minute-Planner-Analog
    # (compute_missing_intervals) nur, wenn das Funding-Volumen wächst.
    total = 0
    pending = list(instruments)
    failures: list[tuple[str, str]] = []
    for attempt in (1, 2):
        still_failing: list[str] = []
        for instrument in pending:
            try:
                rows = client.fetch_range(instrument, start, end)
                inserted = insert_funding_rates(engine, rows)
                total += inserted
                logger.info(
                    "Funding %s (%s): %s → %s, %d Sätze geschrieben",
                    instrument,
                    venue,
                    _fmt(start),
                    _fmt(end),
                    inserted,
                )
            except Exception as exc:
                logger.exception(
                    "Funding-Backfill für %s fehlgeschlagen (Versuch %d): %s",
                    instrument,
                    attempt,
                    exc,
                )
                still_failing.append(instrument)
                failures.append((instrument, str(exc)))
        if not still_failing:
            break
        if attempt == 1:
            logger.warning("Automatischer Funding-Retry für: %s", still_failing)
            pending = still_failing
            failures.clear()
    if failures:
        raise RuntimeError(f"Funding-Backfill unvollständig: {[name for name, _ in failures]}")
    return total


def _format_funding_row(row: FundingRateRow, now: datetime) -> str:
    """Formatiert eine Funding-Rate als ClickHouse-VALUES-Tupel (7 Spalten).

    ``event_time`` = ``funding_time``; ``ingestion_time`` = ``now`` (der
    frische Stempel macht Re-INSERTs idempotent).
    """
    funding_s = row.funding_time.strftime(_DT_FORMAT)
    now_s = now.strftime(_DT_FORMAT)
    return (
        f"({_sql_str(row.instrument)}, {_sql_str(row.venue)}, '{funding_s}', "
        f"{row.funding_rate}, {row.mark_price}, "
        f"'{funding_s}', '{now_s}')"
    )


def _fmt(moment: datetime) -> str:
    """Formatiert einen Zeitstempel für Log-Nachrichten."""
    return moment.strftime(_FMT_FORMAT)


def _sql_str(value: str) -> str:
    """Escapt einen String für ein ClickHouse-String-Literal."""
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"
