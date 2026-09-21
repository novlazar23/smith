"""Tests für den Funding-Rate-Backfill (``apps/backfill/funding.py``).

Der HTTP-Layer wird mit ``httpx.MockTransport`` gefälscht — es findet kein
echtes Netzwerk statt; alle Sleeps (Request-Pausen, Backoff) werden über
eine injizierte Fake-Funktion erfasst (Pattern aus ``test_client.py``).
ClickHouse wird durch eine ``FakeEngine`` ersetzt, die alle SQL-Statements
aufzeichnet (Pattern aus ``test_storage.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from types import TracebackType
from typing import Any

import httpx
import pytest
from apps.backfill import __main__ as backfill_main
from apps.backfill.client import BinanceAPIError, BinanceRateLimitError
from apps.backfill.funding import (
    FUNDING_TABLE_NAME,
    FundingRateClient,
    FundingRateRow,
    count_funding_rates,
    ensure_funding_table,
    insert_funding_rates,
    map_funding_row,
    refresh_funding,
)
from packages.persistence.clickhouse.engine import ClickHouseConfig

START = datetime(2025, 1, 1, tzinfo=UTC)
START_MS = 1_735_689_600_000
EIGHT_HOURS_MS = 28_800_000


class FakeEngine:
    """ClickHouse-Test-Doppel: zeichnet ``_execute`` auf, ``query`` canned."""

    def __init__(self, query_rows: list[list[str]] | None = None) -> None:
        self.config = ClickHouseConfig(database="trading_events")
        self.executed: list[str] = []
        self._query_rows: list[list[str]] = query_rows if query_rows is not None else []

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        del sql
        return (["total"], self._query_rows)

    def _execute(self, query: str) -> None:
        self.executed.append(query)


def _noop(seconds: float) -> None:
    del seconds


def _funding_row_dict(ms: int, mark_price: str | None = None) -> dict[str, Any]:
    """Baut eine Roh-Funding-Zeile der Binance-API (``markPrice`` optional)."""
    row: dict[str, Any] = {"fundingTime": ms, "fundingRate": "0.0001"}
    if mark_price is not None:
        row["markPrice"] = mark_price
    return row


def _funding_transport(
    data_start: datetime,
    data_end: datetime,
) -> tuple[httpx.MockTransport, list[dict[str, str]]]:
    """Simuliert Binance: Funding-Sätze alle 8 h in [data_start, data_end]."""
    data_start_ms = int(data_start.timestamp()) * 1000
    data_end_ms = int(data_end.timestamp()) * 1000
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        window_start = int(params["startTime"])
        window_end = int(params["endTime"])
        limit = int(params["limit"])
        if window_start > data_start_ms:
            offset = math.ceil((window_start - data_start_ms) / EIGHT_HOURS_MS)
            open_ms = data_start_ms + offset * EIGHT_HOURS_MS
        else:
            open_ms = max(window_start, data_start_ms)
        rows: list[dict[str, Any]] = []
        while open_ms <= min(window_end, data_end_ms) and len(rows) < limit:
            rows.append(_funding_row_dict(open_ms))
            open_ms += EIGHT_HOURS_MS
        return httpx.Response(200, json=rows)

    return httpx.MockTransport(handler), calls


def _scripted_transport(
    responses: list[httpx.Response],
) -> tuple[httpx.MockTransport, list[dict[str, str]]]:
    """Spielt eine feste Folge von HTTP-Antworten ab (Retry-/Ende-Tests)."""
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        return responses.pop(0)

    return httpx.MockTransport(handler), calls


def _per_symbol_transport(script: dict[str, list[httpx.Response]]) -> httpx.MockTransport:
    """Spielt pro Symbol eine feste Folge von HTTP-Antworten ab."""

    def handler(request: httpx.Request) -> httpx.Response:
        symbol = request.url.params["symbol"]
        return script[symbol].pop(0)

    return httpx.MockTransport(handler)


def _client(
    transport: httpx.BaseTransport,
    request_delay: float = 0.0,
    max_retries: int = 3,
    sleeps: list[float] | None = None,
) -> FundingRateClient:
    """FundingRateClient mit MockTransport und injizierbarem Sleep."""
    return FundingRateClient(
        transport=transport,
        request_delay=request_delay,
        max_retries=max_retries,
        sleep=sleeps.append if sleeps is not None else _noop,
    )


def _collect(
    chunks: list[tuple[datetime, datetime, int]],
) -> Callable[[datetime, datetime, int], None]:
    """Progress-Callback, der alle (start, end, count)-Triplets sammelt."""

    def on_chunk(chunk_start: datetime, chunk_end: datetime, count: int) -> None:
        chunks.append((chunk_start, chunk_end, count))

    return on_chunk


def _funding_rows(count: int, start: datetime) -> list[FundingRateRow]:
    """count FundingRateRows im 8h-Raster ab ``start``."""
    return [
        FundingRateRow(
            instrument="BTC/USDT",
            venue="BINANCE_FUTURES",
            funding_time=start + timedelta(hours=8 * index),
            funding_rate=0.0001,
            mark_price=100.5,
        )
        for index in range(count)
    ]


class TestFetchRangePagination:
    def test_paginates_forward_in_1000_funding_chunks(self) -> None:
        """1000-Zeilen-Seiten rücken den Cursor auf letzte fundingTime + 1 ms."""
        end = START + timedelta(hours=8 * 2500)  # 2501 Sätze → 3 Requests
        transport, calls = _funding_transport(START, end)
        client = _client(transport)
        rows = client.fetch_range("BTC/USDT", START, end)

        assert [call["startTime"] for call in calls] == [
            str(START_MS),
            str(START_MS + 999 * EIGHT_HOURS_MS + 1),
            str(START_MS + 1999 * EIGHT_HOURS_MS + 1),
        ]
        assert all(call["endTime"] == str(int(end.timestamp()) * 1000) for call in calls)
        assert [call["limit"] for call in calls] == ["1000", "1000", "1000"]
        assert calls[0]["symbol"] == "BTCUSDT"
        assert len(rows) == 2501
        assert rows[0].funding_time == START
        assert rows[-1].funding_time == end
        assert all(a.funding_time < b.funding_time for a, b in pairwise(rows))
        assert {row.instrument for row in rows} == {"BTC/USDT"}
        assert {row.venue for row in rows} == {"BINANCE_FUTURES"}

    def test_single_page_when_under_1000(self) -> None:
        end = START + timedelta(hours=8 * 5)  # 6 Sätze → 1 Request
        transport, calls = _funding_transport(START, end)
        client = _client(transport)
        rows = client.fetch_range("BTC/USDT", START, end)

        assert len(calls) == 1
        assert len(rows) == 6

    def test_filters_rows_beyond_end(self) -> None:
        responses = [
            httpx.Response(
                200,
                json=[
                    _funding_row_dict(START_MS),
                    _funding_row_dict(START_MS + EIGHT_HOURS_MS),
                    _funding_row_dict(START_MS + 2 * EIGHT_HOURS_MS),
                ],
            )
        ]
        transport, _ = _scripted_transport(responses)
        client = _client(transport, max_retries=0)
        rows = client.fetch_range("BTC/USDT", START, START + timedelta(hours=8))

        assert [row.funding_time for row in rows] == [START, START + timedelta(hours=8)]

    def test_no_data_in_window_returns_empty(self) -> None:
        transport, calls = _funding_transport(
            START + timedelta(hours=80), START + timedelta(hours=160)
        )
        client = _client(transport)
        rows = client.fetch_range("BTC/USDT", START, START + timedelta(hours=8))

        assert rows == []
        assert len(calls) == 1

    def test_sleeps_between_requests(self) -> None:
        end = START + timedelta(hours=8 * 2500)
        transport, _ = _funding_transport(START, end)
        sleeps: list[float] = []
        client = _client(transport, request_delay=0.25, sleeps=sleeps)
        client.fetch_range("BTC/USDT", START, end)

        assert sleeps == [0.25, 0.25]

    def test_reports_chunk_progress(self) -> None:
        end = START + timedelta(hours=8 * 2500)
        transport, _ = _funding_transport(START, end)
        chunks: list[tuple[datetime, datetime, int]] = []
        client = _client(transport)
        client.fetch_range("BTC/USDT", START, end, on_chunk=_collect(chunks))

        assert chunks == [
            (START, START + timedelta(hours=8 * 999), 1000),
            (START + timedelta(hours=8 * 1000), START + timedelta(hours=8 * 1999), 1000),
            (START + timedelta(hours=8 * 2000), end, 501),
        ]


class TestFundingRowMapping:
    def test_fields_mapped_to_utc_datetime(self) -> None:
        row = map_funding_row(
            "BTC/USDT",
            "BINANCE_FUTURES",
            {"fundingTime": START_MS, "fundingRate": "0.000125", "markPrice": "100.5"},
        )

        assert row.instrument == "BTC/USDT"
        assert row.venue == "BINANCE_FUTURES"
        assert row.funding_time == START
        assert row.funding_rate == 0.000125
        assert row.mark_price == 100.5

    def test_missing_or_empty_mark_price_defaults_to_zero(self) -> None:
        row = map_funding_row(
            "BTC/USDT", "BINANCE_FUTURES", {"fundingTime": START_MS, "fundingRate": "0.0001"}
        )
        assert row.mark_price == 0.0
        row = map_funding_row(
            "BTC/USDT",
            "BINANCE_FUTURES",
            {"fundingTime": START_MS, "fundingRate": "0.0001", "markPrice": ""},
        )
        assert row.mark_price == 0.0


class TestRetryPolicy:
    def test_429_honors_retry_after_header(self) -> None:
        responses = [
            httpx.Response(429, headers={"Retry-After": "7"}, text="rate limited"),
            httpx.Response(200, json=[_funding_row_dict(START_MS)]),
        ]
        transport, _ = _scripted_transport(responses)
        sleeps: list[float] = []
        client = _client(transport, sleeps=sleeps)
        rows = client.fetch_page("BTCUSDT", START_MS, START_MS)

        assert rows == [_funding_row_dict(START_MS)]
        assert sleeps == [7.0]

    def test_429_without_header_uses_exponential_backoff(self) -> None:
        responses = [
            httpx.Response(429, text="rate limited"),
            httpx.Response(429, text="rate limited"),
            httpx.Response(200, json=[_funding_row_dict(START_MS)]),
        ]
        transport, _ = _scripted_transport(responses)
        sleeps: list[float] = []
        client = _client(transport, sleeps=sleeps)
        client.fetch_page("BTCUSDT", START_MS, START_MS)

        assert sleeps == [1.0, 2.0]

    def test_429_exhausts_retries_then_raises_rate_limit_error(self) -> None:
        responses = [httpx.Response(429, text="rate limit")] * 4
        transport, calls = _scripted_transport(responses)
        sleeps: list[float] = []
        client = _client(transport, sleeps=sleeps)

        with pytest.raises(BinanceRateLimitError):
            client.fetch_page("BTCUSDT", START_MS, START_MS)

        assert len(calls) == 4  # Initialversuch + 3 Retries
        assert sleeps == [1.0, 2.0, 4.0]

    def test_5xx_exhausts_retries_then_raises_api_error(self) -> None:
        responses = [httpx.Response(502, text="bad gateway")] * 4
        transport, calls = _scripted_transport(responses)
        sleeps: list[float] = []
        client = _client(transport, sleeps=sleeps)

        with pytest.raises(BinanceAPIError):
            client.fetch_page("BTCUSDT", START_MS, START_MS)

        assert len(calls) == 4
        assert sleeps == [1.0, 2.0, 4.0]

    def test_4xx_not_retried(self) -> None:
        responses = [httpx.Response(400, text='{"code": -1121, "msg": "Invalid symbol."}')]
        transport, calls = _scripted_transport(responses)
        sleeps: list[float] = []
        client = _client(transport, sleeps=sleeps)

        with pytest.raises(BinanceAPIError, match="400"):
            client.fetch_page("NOPE", START_MS, START_MS)

        assert len(calls) == 1
        assert sleeps == []


class TestEnsureFundingTable:
    def test_executes_exact_ddl_without_ttl(self) -> None:
        engine = FakeEngine()
        ensure_funding_table(engine)

        assert engine.executed == [
            "CREATE TABLE IF NOT EXISTS trading_events.funding_rates (instrument String, venue String, "
            "funding_time DateTime, funding_rate Float64, mark_price Float64, event_time DateTime, "
            "ingestion_time DateTime) ENGINE = ReplacingMergeTree(ingestion_time) "
            "PARTITION BY toYYYYMM(funding_time) ORDER BY (instrument, venue, funding_time) "
            "SETTINGS index_granularity = 8192"
        ]
        assert "TTL" not in engine.executed[0]
        assert FUNDING_TABLE_NAME == "funding_rates"


class TestInsertFundingRates:
    def test_batches_of_5000_rows(self) -> None:
        engine = FakeEngine()
        inserted = insert_funding_rates(engine, _funding_rows(5001, START))

        assert inserted == 5001
        assert len(engine.executed) == 2
        assert engine.executed[0].count("('BTC/USDT'") == 5000
        assert engine.executed[1].count("('BTC/USDT'") == 1

    def test_empty_rows_insert_nothing(self) -> None:
        engine = FakeEngine()

        assert insert_funding_rates(engine, []) == 0
        assert engine.executed == []

    def test_row_format_seven_columns_event_time_equals_funding_time(self) -> None:
        engine = FakeEngine()
        insert_funding_rates(engine, _funding_rows(1, START))

        sql = engine.executed[0]
        assert sql.startswith(
            "INSERT INTO trading_events.funding_rates "
            "(instrument, venue, funding_time, funding_rate, mark_price, "
            "event_time, ingestion_time) VALUES ("
        )
        assert (
            "('BTC/USDT', 'BINANCE_FUTURES', '2025-01-01 00:00:00', 0.0001, 100.5, "
            "'2025-01-01 00:00:00', "
        ) in sql
        assert sql.endswith("')")


class TestCountFundingRates:
    def test_count_reads_first_cell(self) -> None:
        engine = FakeEngine(query_rows=[["42"]])
        assert count_funding_rates(engine, "BTC/USDT", "BINANCE_FUTURES") == 42

    def test_count_empty_result_is_zero(self) -> None:
        engine = FakeEngine()
        assert count_funding_rates(engine, "BTC/USDT", "BINANCE_FUTURES") == 0


class TestRefreshFunding:
    def test_inserts_all_instruments_and_returns_total(self) -> None:
        script = {
            "BTCUSDT": [
                httpx.Response(
                    200,
                    json=[_funding_row_dict(START_MS), _funding_row_dict(START_MS + EIGHT_HOURS_MS)],
                )
            ],
            "ETHUSDT": [httpx.Response(200, json=[_funding_row_dict(START_MS)])],
        }
        client = _client(_per_symbol_transport(script), max_retries=0)
        engine = FakeEngine()

        total = refresh_funding(
            engine, client, ("BTC/USDT", "ETH/USDT"), START, START + timedelta(hours=8)
        )

        assert total == 3
        assert len(engine.executed) == 2

    def test_retries_failing_instrument_once(self) -> None:
        script = {
            "BTCUSDT": [
                httpx.Response(500, text="boom"),
                httpx.Response(200, json=[_funding_row_dict(START_MS)]),
            ],
            "ETHUSDT": [httpx.Response(200, json=[_funding_row_dict(START_MS)])],
        }
        client = _client(_per_symbol_transport(script), max_retries=0)
        engine = FakeEngine()

        total = refresh_funding(
            engine, client, ("BTC/USDT", "ETH/USDT"), START, START + timedelta(hours=8)
        )

        assert total == 2
        assert len(engine.executed) == 2

    def test_persistent_failure_raises_with_only_failing_instrument(self) -> None:
        script = {
            "BTCUSDT": [httpx.Response(500, text="boom")] * 2,
            "ETHUSDT": [httpx.Response(200, json=[_funding_row_dict(START_MS)])],
        }
        client = _client(_per_symbol_transport(script), max_retries=0)

        with pytest.raises(RuntimeError) as excinfo:
            refresh_funding(
                FakeEngine(), client, ("BTC/USDT", "ETH/USDT"), START, START + timedelta(hours=8)
            )

        assert "BTC/USDT" in str(excinfo.value)
        assert "ETH/USDT" not in str(excinfo.value)

    def test_all_failures_listed_in_error(self) -> None:
        script = {
            "BTCUSDT": [httpx.Response(500, text="boom")] * 2,
            "ETHUSDT": [httpx.Response(500, text="boom")] * 2,
        }
        client = _client(_per_symbol_transport(script), max_retries=0)

        with pytest.raises(RuntimeError) as excinfo:
            refresh_funding(
                FakeEngine(), client, ("BTC/USDT", "ETH/USDT"), START, START + timedelta(hours=8)
            )

        assert "BTC/USDT" in str(excinfo.value)
        assert "ETH/USDT" in str(excinfo.value)


class TestFundingCli:
    """funding-Subkommando: Choice, Defaults und Arg-Validierung (Exit 2)."""

    def test_funding_choice_accepted_reaches_clickhouse_bootstrap(self) -> None:
        # Exit 1 = CH nicht erreichbar (Test-Env) → Choice wurde akzeptiert.
        assert (
            backfill_main.main(
                [
                    "funding",
                    "--start",
                    "2021-05-15",
                    "--end",
                    "2021-05-16",
                    "--ch-host",
                    "invalid-host-xyz",
                ]
            )
            == 1
        )

    def test_funding_defaults_reach_clickhouse_bootstrap(self) -> None:
        # Default-Start 2019-01-01, Default-End now, 6 Default-Instrumente
        # → valides Fenster, CH-Bootstrap ist der erste externe Schritt.
        assert backfill_main.main(["funding", "--ch-host", "invalid-host-xyz"]) == 1

    def test_start_after_end_returns_2(self) -> None:
        assert backfill_main.main(["funding", "--start", "2021-05-25", "--end", "2021-05-15"]) == 2

    def test_end_before_default_start_returns_2(self) -> None:
        # Default-Start 2019-01-01 liegt nach 2018-01-01 → Early-Validierung
        # vor dem CH-Bootstrap (kein Netzwerkzugriff).
        assert backfill_main.main(["funding", "--end", "2018-01-01"]) == 2

    def test_invalid_date_returns_2(self) -> None:
        assert backfill_main.main(["funding", "--start", "15.05.2021"]) == 2

    def test_empty_instruments_return_2(self) -> None:
        assert backfill_main.main(["funding", "--instruments", ""]) == 2


class TestFundingCliWiring:
    def test_funding_command_runs_default_instruments_and_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        class _FakeClient:
            def __enter__(self) -> _FakeClient:
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                del exc_type, exc, tb

        class _FakeEngine:
            config = ClickHouseConfig(database="trading_events")

            def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
                del sql
                return (["total"], [["3"]])

            def _execute(self, sql: str) -> None:
                del sql

        def _fake_create_ch_engine(config: ClickHouseConfig) -> _FakeEngine:
            del config
            return _FakeEngine()

        def _fake_ensure_funding_table(engine: object) -> None:
            seen["ensure"] = engine

        def _fake_refresh_funding(
            engine: object,
            client: object,
            instruments: object,
            start: object,
            end: object,
            *,
            venue: object = "BINANCE_FUTURES",
        ) -> int:
            del engine, client
            seen["refresh"] = (instruments, start, end, venue)
            return 7

        monkeypatch.setattr(backfill_main, "create_ch_engine", _fake_create_ch_engine)
        monkeypatch.setattr(backfill_main, "ensure_funding_table", _fake_ensure_funding_table)
        monkeypatch.setattr(backfill_main, "FundingRateClient", lambda: _FakeClient())
        monkeypatch.setattr(backfill_main, "refresh_funding", _fake_refresh_funding)

        return_code = backfill_main.main(["funding", "--start", "2021-05-15", "--end", "2021-05-16"])

        assert return_code == 0
        assert "ensure" in seen
        refresh = seen.get("refresh")
        assert isinstance(refresh, tuple)
        assert len(refresh) == 4
        assert refresh[0] == (
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "BNB/USDT",
            "XRP/USDT",
            "ADA/USDT",
        )
        assert refresh[1] == datetime(2021, 5, 15, tzinfo=UTC)
        assert refresh[2] == datetime(2021, 5, 16, 23, 59, tzinfo=UTC)
        assert refresh[3] == "BINANCE_FUTURES"
