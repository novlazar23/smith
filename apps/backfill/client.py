"""Binance-Futures-Klines-Client für den Candle-Backfill.

Synchroner, paginierter Client für den öffentlichen
``GET /klines``-Endpunkt der Binance-Futures-REST-API. Basis-URL und
Venue übernimmt er aus dem bestehenden ``BinanceAdapter``
(``packages/ingestion/adapter/binance.py``) — der Adapter selbst ist
asynchron (aiohttp) und kennt keine ``startTime``/``endTime``-Parameter,
deshalb nutzt der Backfill hier einen eigenen httpx-Client (dieselbe
HTTP-Bibliothek wie die ClickHouse-Engine) und paginiert vorwärts in
1000-Kerzen-Fenstern:

- ``limit=1000``, ``interval=1m``, ``startTime``/``endTime`` in Millisekunden
- 0,25 s Pause zwischen Requests (Binance-Limit: 1500 Gewicht/min)
- HTTP 429/418 und 5xx: ``Retry-After`` honorieren, sonst exponentieller
  Backoff (1 s, 2 s, 4 s), max. 3 Retries, danach Fehler
- Kline-Zeilen werden auf das kanonische Kerzen-Format gemappt:
  ``instrument`` bleibt "BTC/USDT" (CH-Format), ``open_time`` wird aus
  den Millisekunden als UTC-datetime gebildet (CH-Spalte ``DateTime``)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

import httpx
from apps.market_producer.producer import to_exchange_symbol
from packages.ingestion.adapter.binance import (
    BINANCE_FUTURES_BASE_URL,
    BINANCE_FUTURES_VENUE,
)

logger = logging.getLogger(__name__)

KLINE_INTERVAL = "1m"
PAGE_SIZE = 1000
ONE_MINUTE_MS = 60_000
PAGE_SPAN_MS = PAGE_SIZE * ONE_MINUTE_MS
DEFAULT_REQUEST_DELAY = 0.25
DEFAULT_MAX_RETRIES = 3
RATE_LIMIT_STATUSES = frozenset({418, 429})
RETRYABLE_STATUSES = frozenset({418, 429, 500, 502, 503, 504})


class BinanceAPIError(Exception):
    """Nicht retrybarer Binance-API-Fehler (oder 5xx nach 3 Retries)."""


class BinanceRateLimitError(BinanceAPIError):
    """Rate-Limit (HTTP 429/418) auch nach 3 Retries nicht aufgehoben."""


@dataclass(frozen=True)
class BackfillCandle:
    """Eine auf das ClickHouse-Format gemappte 1m-Kerze."""

    instrument: str
    venue: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def map_kline_row(instrument: str, venue: str, row: list[Any]) -> BackfillCandle:
    """Mappt eine Roh-Kline-Zeile der Binance-API auf eine Backfill-Kerze.

    Binance-Format: ``[open_time_ms, open, high, low, close, volume,
    close_time_ms, quote_volume, trades, taker_base, taker_quote, ...]``.

    Args:
        instrument: Kanonisches Instrument (z. B. ``"BTC/USDT"``).
        venue: Venue (z. B. ``"BINANCE_FUTURES"``).
        row: Roh-Zeile aus der API-Antwort.

    Returns:
        Gemappte Kerze; ``open_time`` als UTC-datetime aus
        ``open_time_ms`` (CH-Spalte ``open_time`` ist ``DateTime``).
    """
    return BackfillCandle(
        instrument=instrument,
        venue=venue,
        open_time=datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def _dt_to_ms(moment: datetime) -> int:
    """Konvertiert ein UTC-datetime auf die Millisekunde-Ebene der API."""
    return int(moment.timestamp()) * 1000


def _ms_to_dt(ms: int) -> datetime:
    """Konvertiert eine Millisekunden-Zeites zu einem UTC-datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


class KlineClient:
    """Paginierter Klines-Client für Binance Futures (synchron, httpx).

    Lädt 1m-Kerzen von ``start`` bis ``end`` in 1000-Kerzen-Fenstern im
    Vorwärtsmodus (``startTime`` rückt pro Request um 1000 Minuten).
    Prozeß-sicher: ein einziger Lauf lädt jede Kerze exakt einmal —
    die Deduplication im ClickHouse (``ReplacingMergeTree``) fängt
    dennoch Überlappungen ab.
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
            venue: Venue-Stempel für die gemappten Kerzen.
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

    def fetch_page(self, symbol: str, start_ms: int, end_ms: int) -> list[list[Any]]:
        """Holt eine Kline-Seite (max. 1000 Kerzen) mit Retry-Policy.

        Args:
            symbol: Binance-Symbol (z. B. ``"BTCUSDT"``).
            start_ms: Fensterbeginn in Millisekunden (inklusive).
            end_ms: Fensterende in Millisekunden (inklusive).

        Returns:
            Roh-Kline-Zeilen des Fensters (aufsteigend nach ``open_time``).

        Raises:
            BinanceAPIError: Bei nicht retrybaren HTTP-Fehlern (4xx) oder
                5xx nach ``max_retries`` Retries.
            BinanceRateLimitError: Bei 429/418 nach ``max_retries`` Retries.
        """
        params = {
            "symbol": symbol,
            "interval": KLINE_INTERVAL,
            "startTime": str(start_ms),
            "endTime": str(end_ms),
            "limit": str(PAGE_SIZE),
        }
        for attempt in range(self._max_retries + 1):
            response = self._ensure_client().get("/klines", params=params)
            if response.status_code in RETRYABLE_STATUSES:
                if attempt >= self._max_retries:
                    raise self._retry_error(response, symbol)
                delay = self._retry_delay(response, attempt)
                logger.warning(
                    "Klines HTTP %d (%s) — Retry %d/%d in %.1fs",
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
                    f"Klines HTTP {response.status_code} ({symbol}): {response.text[:200]}"
                )
            payload = response.json()
            if not isinstance(payload, list):
                raise BinanceAPIError(f"Klines-Antwort ist kein Array ({symbol})")
            return payload
        raise BinanceAPIError(f"Klines-Request nicht ausgeführt ({symbol})")

    def fetch_range(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
        on_chunk: Callable[[datetime, datetime, int], None] | None = None,
    ) -> list[BackfillCandle]:
        """Lädt alle 1m-Kerzen von ``start`` bis ``end`` (beides inklusive).

        Paginiert vorwärts in 1000-Kerzen-Fenstern; zwischen den Requests
        wird ``request_delay`` Sekunden gewartet.

        Args:
            instrument: Kanonisches Instrument (z. B. ``"BTC/USDT"``).
            start: Fensterbeginn (UTC, Minutenauflösung).
            end: Fensterende (UTC, Minutenauflösung, inklusive).
            on_chunk: Optionaler Progress-Callback pro 1000-Kerzen-Fenster
                (open_time der ersten Kerze, open_time der letzten Kerze,
                Kerzenzahl des Fensters).

        Returns:
            Alle Kerzen des Fensters (aufsteigend nach ``open_time``).
        """
        symbol = to_exchange_symbol(instrument)
        start_ms = _dt_to_ms(start)
        end_ms = _dt_to_ms(end)
        candles: list[BackfillCandle] = []
        cursor = start_ms
        while cursor <= end_ms:
            if cursor > start_ms:
                self._sleep(self._request_delay)
            window_end = min(cursor + PAGE_SPAN_MS - 1, end_ms)
            rows = self.fetch_page(symbol, cursor, window_end)
            chunk = [
                map_kline_row(instrument, self._venue, row)
                for row in rows
                if _ms_to_dt(int(row[0])) <= end
            ]
            if on_chunk is not None:
                if chunk:
                    on_chunk(chunk[0].open_time, chunk[-1].open_time, len(chunk))
                else:
                    on_chunk(_ms_to_dt(cursor), _ms_to_dt(window_end), 0)
            candles.extend(chunk)
            cursor += PAGE_SPAN_MS
        return candles

    def close(self) -> None:
        """Schließt den HTTP-Client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> KlineClient:
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
        detail = f"Klines HTTP {response.status_code} ({symbol}) nach {self._max_retries} Retries"
        if response.status_code in RATE_LIMIT_STATUSES:
            return BinanceRateLimitError(f"{detail}: Rate-Limit nicht aufgehoben")
        return BinanceAPIError(f"{detail}: {response.text[:200]}")
