"""Tests für den Binance-Klines-Client (apps/backfill/client.py).

Der HTTP-Layer wird mit ``httpx.MockTransport`` gefälscht — es findet
kein echtes Netzwerk statt. Alle Sleeps (Request-Pausen, Backoff) werden
über eine injizierte Fake-Funktion erfasst statt real gewartet.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

import httpx
import pytest
from apps.backfill.client import (
    PAGE_SPAN_MS,
    BinanceAPIError,
    BinanceRateLimitError,
    KlineClient,
    map_kline_row,
)

START = datetime(2025, 1, 1, tzinfo=UTC)
START_MS = 1_735_689_600_000
ONE_MINUTE_MS = 60_000


def _kline_row(open_ms: int) -> list[Any]:
    """Baut eine gültige Binance-Kline-Zeile (12 Felder)."""
    return [
        open_ms,
        "100.5",
        "101.0",
        "99.5",
        "100.75",
        "12.25",
        open_ms + 59_999,
        "1230.5",
        42,
        "6.1",
        "613.25",
        "0",
    ]


def _noop(seconds: float) -> None:
    del seconds


def _collect(
    chunks: list[tuple[datetime, datetime, int]],
) -> Callable[[datetime, datetime, int], None]:
    """Progress-Callback, der alle (start, end, count)-Triplets sammelt."""

    def on_chunk(chunk_start: datetime, chunk_end: datetime, count: int) -> None:
        chunks.append((chunk_start, chunk_end, count))

    return on_chunk


def _klines_transport(
    data_start: datetime,
    data_end: datetime,
) -> tuple[httpx.MockTransport, list[dict[str, str]]]:
    """Simuliert Binance: liefert Klines für [startTime, endTime].

    Verfügbare Daten beschränken sich auf [data_start, data_end]; alle
    Requests (als Query-Params) werden in ``calls`` aufgezeichnet.
    """
    data_start_ms = int(data_start.timestamp()) * 1000
    data_end_ms = int(data_end.timestamp()) * 1000
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        window_start = int(params["startTime"])
        window_end = int(params["endTime"])
        limit = int(params["limit"])
        open_ms = max(window_start, data_start_ms)
        rows: list[list[Any]] = []
        while open_ms <= min(window_end, data_end_ms) and len(rows) < limit:
            rows.append(_kline_row(open_ms))
            open_ms += ONE_MINUTE_MS
        return httpx.Response(200, json=rows)

    return httpx.MockTransport(handler), calls


def _scripted_transport(
    responses: list[httpx.Response],
) -> tuple[httpx.MockTransport, list[dict[str, str]]]:
    """Spielt eine feste Folge von HTTP-Antworten ab (für Retry-Tests)."""
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        return responses.pop(0)

    return httpx.MockTransport(handler), calls


class TestFetchRangePagination:
    def test_paginates_forward_in_1000_candle_chunks(self) -> None:
        """startTime rückt pro Request um 1000 Kerzen (1000 Min) vor."""
        end = START + timedelta(minutes=2499)  # 2500 Kerzen → 3 Requests
        transport, calls = _klines_transport(START, end)
        client = KlineClient(transport=transport, request_delay=0.0, sleep=_noop)
        candles = client.fetch_range("BTC/USDT", START, end)

        assert [call["startTime"] for call in calls] == [
            str(START_MS),
            str(START_MS + PAGE_SPAN_MS),
            str(START_MS + 2 * PAGE_SPAN_MS),
        ]
        assert [call["endTime"] for call in calls] == [
            str(START_MS + PAGE_SPAN_MS - 1),
            str(START_MS + 2 * PAGE_SPAN_MS - 1),
            str(int(end.timestamp()) * 1000),
        ]
        assert [call["interval"] for call in calls] == ["1m", "1m", "1m"]
        assert [call["limit"] for call in calls] == ["1000", "1000", "1000"]
        assert len(candles) == 2500
        assert candles[0].open_time == START
        assert candles[-1].open_time == end
        assert all(a.open_time < b.open_time for a, b in pairwise(candles))

    def test_converts_symbol_and_keeps_canonical_instrument(self) -> None:
        """'BTC/USDT' geht als 'BTCUSDT' raus, bleibt 'BTC/USDT' in der Kerze."""
        end = START + timedelta(minutes=10)
        transport, calls = _klines_transport(START, end)
        client = KlineClient(transport=transport, request_delay=0.0, sleep=_noop)
        candles = client.fetch_range("BTC/USDT", START, end)

        assert calls[0]["symbol"] == "BTCUSDT"
        assert {candle.instrument for candle in candles} == {"BTC/USDT"}
        assert {candle.venue for candle in candles} == {"BINANCE_FUTURES"}

    def test_sleeps_between_requests(self) -> None:
        end = START + timedelta(minutes=1500)  # 1501 Kerzen → 2 Requests
        transport, _ = _klines_transport(START, end)
        sleeps: list[float] = []
        client = KlineClient(transport=transport, request_delay=0.25, sleep=sleeps.append)
        candles = client.fetch_range("BTC/USDT", START, end)

        assert len(candles) == 1501
        assert sleeps == [0.25]

    def test_reports_chunk_progress(self) -> None:
        end = START + timedelta(minutes=2499)
        transport, _ = _klines_transport(START, end)
        chunks: list[tuple[datetime, datetime, int]] = []
        client = KlineClient(transport=transport, request_delay=0.0, sleep=_noop)
        client.fetch_range("BTC/USDT", START, end, on_chunk=_collect(chunks))

        assert chunks == [
            (START, START + timedelta(minutes=999), 1000),
            (START + timedelta(minutes=1000), START + timedelta(minutes=1999), 1000),
            (START + timedelta(minutes=2000), end, 500),
        ]


class TestKlineRowMapping:
    def test_ms_converted_to_utc_datetime_and_fields_mapped(self) -> None:
        candle = map_kline_row("BTC/USDT", "BINANCE_FUTURES", _kline_row(START_MS))
        assert candle.instrument == "BTC/USDT"
        assert candle.venue == "BINANCE_FUTURES"
        assert candle.open_time == START
        assert (candle.open, candle.high, candle.low, candle.close, candle.volume) == (
            100.5,
            101.0,
            99.5,
            100.75,
            12.25,
        )


class TestRetryPolicy:
    def test_429_retried_with_backoff_then_succeeds(self) -> None:
        responses = [
            httpx.Response(429, text="Too Many Requests"),
            httpx.Response(200, json=[_kline_row(START_MS)]),
        ]
        transport, calls = _scripted_transport(responses)
        sleeps: list[float] = []
        client = KlineClient(transport=transport, sleep=sleeps.append)
        rows = client.fetch_page("BTCUSDT", START_MS, START_MS + PAGE_SPAN_MS - 1)

        assert rows == [_kline_row(START_MS)]
        assert len(calls) == 2
        assert sleeps == [1.0]

    def test_429_honors_retry_after_header(self) -> None:
        responses = [
            httpx.Response(429, headers={"Retry-After": "7"}, text="rate limited"),
            httpx.Response(200, json=[_kline_row(START_MS)]),
        ]
        transport, _ = _scripted_transport(responses)
        sleeps: list[float] = []
        client = KlineClient(transport=transport, sleep=sleeps.append)
        client.fetch_page("BTCUSDT", START_MS, START_MS)

        assert sleeps == [7.0]

    def test_418_retried_like_429(self) -> None:
        responses = [
            httpx.Response(418, text="banned"),
            httpx.Response(200, json=[_kline_row(START_MS)]),
        ]
        transport, _ = _scripted_transport(responses)
        sleeps: list[float] = []
        client = KlineClient(transport=transport, sleep=sleeps.append)
        rows = client.fetch_page("BTCUSDT", START_MS, START_MS)

        assert rows == [_kline_row(START_MS)]
        assert sleeps == [1.0]

    def test_429_exhausts_retries_then_raises(self) -> None:
        responses = [httpx.Response(429, text="rate limit")] * 4
        transport, calls = _scripted_transport(responses)
        sleeps: list[float] = []
        client = KlineClient(transport=transport, sleep=sleeps.append)

        with pytest.raises(BinanceRateLimitError):
            client.fetch_page("BTCUSDT", START_MS, START_MS)

        assert len(calls) == 4  # Initialversuch + 3 Retries
        assert sleeps == [1.0, 2.0, 4.0]

    def test_5xx_retried_then_succeeds(self) -> None:
        responses = [
            httpx.Response(503, text="unavailable"),
            httpx.Response(200, json=[_kline_row(START_MS)]),
        ]
        transport, _ = _scripted_transport(responses)
        sleeps: list[float] = []
        client = KlineClient(transport=transport, sleep=sleeps.append)
        rows = client.fetch_page("BTCUSDT", START_MS, START_MS)

        assert rows == [_kline_row(START_MS)]
        assert sleeps == [1.0]

    def test_5xx_exhausts_retries_then_raises(self) -> None:
        responses = [httpx.Response(502, text="bad gateway")] * 4
        transport, calls = _scripted_transport(responses)
        sleeps: list[float] = []
        client = KlineClient(transport=transport, sleep=sleeps.append)

        with pytest.raises(BinanceAPIError):
            client.fetch_page("BTCUSDT", START_MS, START_MS)

        assert len(calls) == 4
        assert sleeps == [1.0, 2.0, 4.0]

    def test_4xx_not_retried(self) -> None:
        responses = [httpx.Response(400, text='{"code": -1121, "msg": "Invalid symbol."}')]
        transport, calls = _scripted_transport(responses)
        sleeps: list[float] = []
        client = KlineClient(transport=transport, sleep=sleeps.append)

        with pytest.raises(BinanceAPIError, match="400"):
            client.fetch_page("NOPE", START_MS, START_MS)

        assert len(calls) == 1
        assert sleeps == []
