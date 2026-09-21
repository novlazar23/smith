"""Tests für die 8h-Funding-Settlement-Logik des Demo-Traders.

Die Uhr wird über eine injizierte ``now``-Funktion gesteuert (kein
globales datetime-Mocken); die Funding-Rates kommen von einem Stub.
``opened_at`` liegt dabei nahe der echten Systemzeit, damit der
echte-Wanduhr-basierte Max-Haltezeit-Backstop nicht auslöst.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from apps.demo_trader.service import (
    INSERT_DEMO_TRADE,
    UPSERT_DEMO_ACCOUNT,
    ClickHouseFundingRateSource,
    DemoTrader,
    DemoTraderConfig,
    _floor_8h,
)
from packages.paper import PaperPosition
from packages.persistence.clickhouse.engine import ClickHouseConfig, ClickHouseEngine

from .conftest import (
    BTC,
    FakeConnection,
    make_funding_trader,
)

INITIAL_CASH = 50_000.0
POSITION_QTY = 2.0
MARK_PRICE = 100.0


class FakeClock:
    """Stellvertreter für die Trader-Uhr (setzbarer, deterministischer Wert)."""

    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class StubFundingSource:
    """Stellvertreter für FundingRateSource (Rates pro 8h-Grenzwert)."""

    def __init__(self, rates: dict[datetime, float] | None = None) -> None:
        self.rates: dict[datetime, float] = rates if rates is not None else {}
        self.requests: list[tuple[str, str, datetime]] = []

    def settled_rate(self, instrument: str, venue: str, funding_time: datetime) -> float | None:
        self.requests.append((instrument, venue, funding_time))
        return self.rates.get(funding_time)


class FakeCHEngine(ClickHouseEngine):
    """ClickHouseEngine-Stub (vordefinierte Zeilen oder Fehler, keine Netzwerk-Zugriffe)."""

    def __init__(self, rows: list[list[str]] | None = None, error: Exception | None = None) -> None:
        super().__init__(ClickHouseConfig(database="trading_events"))
        self._rows: list[list[str]] = rows if rows is not None else []
        self._error = error
        self.queries: list[str] = []

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        self.queries.append(sql)
        if self._error is not None:
            raise self._error
        return ["funding_rate"], self._rows


def open_position(trader: DemoTrader, opened_at: datetime) -> None:
    """Öffnet eine deterministische Long-Position (2.0 @ 100.0) auf BTC."""
    trader.account.cash = INITIAL_CASH
    trader.account.positions[BTC] = PaperPosition(
        symbol=BTC,
        quantity=POSITION_QTY,
        avg_price=MARK_PRICE,
        opened_at=opened_at,
    )


def funding_rows(conn: FakeConnection) -> list[dict]:
    """Die FUNDING-Audit-Zeilen unter den protokollierten INSERT_DEMO_TRADE-Statements."""
    return [
        params
        for (statement, params) in conn.executed
        if statement == INSERT_DEMO_TRADE and params["direction"] == "FUNDING"
    ]


def _boundary_near_now() -> tuple[datetime, datetime]:
    """(Position-Eröffnung nahe echter Systemzeit, nächster 8h-Grenzwert danach)."""
    last = _floor_8h(datetime.now(UTC))
    return last + timedelta(minutes=1), last + timedelta(hours=8)


class TestSettleFundingCycle:
    """Cash-Booking und FUNDING-Audit-Zeilen im Zyklus."""

    def test_single_boundary_books_funding_into_cash(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """(a) Position + eine überquerte 8h-Grenze → Cash kleiner, FUNDING-Zeile, neuer Upsert-Cash."""
        opened_at, boundary = _boundary_near_now()
        clock = FakeClock(boundary + timedelta(minutes=1))
        source = StubFundingSource({boundary: 0.01})
        trader = make_funding_trader(config, fake_conn, source, clock)
        open_position(trader, opened_at)

        trader.run_cycle()

        assert trader.account.cash == pytest.approx(INITIAL_CASH - POSITION_QTY * MARK_PRICE * 0.01)
        assert trader._total_funding == pytest.approx(POSITION_QTY * MARK_PRICE * 0.01)
        rows = funding_rows(fake_conn)
        assert len(rows) == 1
        row = rows[0]
        assert row["instrument"] == BTC
        assert row["quantity"] == POSITION_QTY
        assert row["price"] == pytest.approx(0.01)
        assert row["filled_price"] == pytest.approx(MARK_PRICE)
        assert row["filled_quantity"] == POSITION_QTY
        assert row["commission"] == 0.0
        assert row["slippage"] == 0.0
        assert row["status"] == "filled"
        assert row["trade_id"] == f"demo-funding-{boundary:%Y%m%dT%H%M%SZ}-{BTC}"
        assert source.requests == [(BTC, config.candle_venue, boundary)]
        assert trader._last_funding[BTC] == boundary
        upserts = [params for (stmt, params) in fake_conn.executed if stmt == UPSERT_DEMO_ACCOUNT]
        assert len(upserts) == 1
        assert upserts[0]["cash"] == pytest.approx(INITIAL_CASH - POSITION_QTY * MARK_PRICE * 0.01)

    def test_no_boundary_crossed_no_booking(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """(b) Keine überquerte 8h-Grenze → keine FUNDING-Zeile, Cash unverändert."""
        opened_at, boundary = _boundary_near_now()
        clock = FakeClock(boundary - timedelta(minutes=1))
        source = StubFundingSource({boundary: 0.01})
        trader = make_funding_trader(config, fake_conn, source, clock)
        open_position(trader, opened_at)

        trader.run_cycle()

        assert trader.account.cash == pytest.approx(INITIAL_CASH)
        assert funding_rows(fake_conn) == []
        assert source.requests == []

    def test_missing_rate_not_booked_and_not_retried(
        self,
        config: DemoTraderConfig,
        fake_conn: FakeConnection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """(c) Rate fehlt → kein Cash, keine Zeile, kein Fehler; Grenzwert wird übersprungen."""
        opened_at, boundary = _boundary_near_now()
        clock = FakeClock(boundary + timedelta(minutes=1))
        source = StubFundingSource()
        trader = make_funding_trader(config, fake_conn, source, clock)
        open_position(trader, opened_at)

        with caplog.at_level("WARNING"):
            trader.run_cycle()

        assert trader.account.cash == pytest.approx(INITIAL_CASH)
        assert funding_rows(fake_conn) == []
        assert "keine Rate in funding_rates" in caplog.text
        assert trader._last_funding[BTC] == boundary
        # Nächster Zyklus im selben Fenster: keine erneute Rate-Abfrage.
        source.requests.clear()
        trader.run_cycle()
        assert source.requests == []
        assert len(funding_rows(fake_conn)) == 0

    def test_clock_jump_17h_books_two_settlements(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """(d) 17h-Sprung → genau zwei Settlements, jeweils mit der Rate am eigenen Grenzwert."""
        last = _floor_8h(datetime.now(UTC))
        opened_at = last + timedelta(minutes=1)
        b1 = last + timedelta(hours=8)
        b2 = last + timedelta(hours=16)
        clock = FakeClock(opened_at + timedelta(hours=17))
        source = StubFundingSource({b1: 0.01, b2: 0.02})
        trader = make_funding_trader(config, fake_conn, source, clock)
        open_position(trader, opened_at)

        trader.run_cycle()

        expected = INITIAL_CASH - POSITION_QTY * MARK_PRICE * 0.01 - POSITION_QTY * MARK_PRICE * 0.02
        assert trader.account.cash == pytest.approx(expected)
        rows = funding_rows(fake_conn)
        assert [row["price"] for row in rows] == pytest.approx([0.01, 0.02])
        assert [row["trade_id"] for row in rows] == [
            f"demo-funding-{b1:%Y%m%dT%H%M%SZ}-{BTC}",
            f"demo-funding-{b2:%Y%m%dT%H%M%SZ}-{BTC}",
        ]
        assert source.requests == [(BTC, config.candle_venue, b1), (BTC, config.candle_venue, b2)]
        assert trader._last_funding[BTC] == b2

    def test_closed_position_clears_settlement_state(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """(e) Keine offene Position mehr → keine Buchung, _last_funding-Eintrag entfernt."""
        opened_at, boundary = _boundary_near_now()
        clock = FakeClock(boundary + timedelta(minutes=1))
        source = StubFundingSource({boundary: 0.01})
        trader = make_funding_trader(config, fake_conn, source, clock)
        open_position(trader, opened_at)

        trader.run_cycle()
        del trader.account.positions[BTC]
        source.requests.clear()
        trader.run_cycle()

        assert len(funding_rows(fake_conn)) == 1
        assert source.requests == []
        assert BTC not in trader._last_funding
        assert trader.account.cash == pytest.approx(INITIAL_CASH - POSITION_QTY * MARK_PRICE * 0.01)

    def test_without_funding_source_nothing_booked(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """(f) funding_source=None → keine Buchung, keine Fehler, kein _last_funding-Zustand."""
        opened_at, boundary = _boundary_near_now()
        clock = FakeClock(boundary + timedelta(minutes=1))
        trader = make_funding_trader(config, fake_conn, None, clock)
        open_position(trader, opened_at)

        trader.run_cycle()

        assert trader.account.cash == pytest.approx(INITIAL_CASH)
        assert [stmt for (stmt, _) in fake_conn.executed if stmt == INSERT_DEMO_TRADE] == []
        assert BTC not in trader._last_funding

    def test_negative_rate_increases_cash(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """(g) Negative Rate → Long erhält, Cash steigt."""
        opened_at, boundary = _boundary_near_now()
        clock = FakeClock(boundary + timedelta(minutes=1))
        source = StubFundingSource({boundary: -0.01})
        trader = make_funding_trader(config, fake_conn, source, clock)
        open_position(trader, opened_at)

        trader.run_cycle()

        assert trader.account.cash == pytest.approx(INITIAL_CASH + POSITION_QTY * MARK_PRICE * 0.01)
        assert trader._total_funding == pytest.approx(-POSITION_QTY * MARK_PRICE * 0.01)
        assert len(funding_rows(fake_conn)) == 1
        assert funding_rows(fake_conn)[0]["price"] == pytest.approx(-0.01)


class TestClickHouseFundingRateSource:
    """ClickHouse-Quelle: Parsing, Leerergebnis, Fail-Soft, Escaping."""

    def test_parses_rate_from_row(self) -> None:
        """Eine Zeile → Float; SQL enthält Tabelle, Grenzwert und Escaping."""
        engine = FakeCHEngine(rows=[["0.01"]])
        source = ClickHouseFundingRateSource(engine)
        funding_time = datetime(2026, 1, 1, 16, 0, 0, tzinfo=UTC)

        rate = source.settled_rate("O'Neil/USDT", "BINANCE_FUTURES", funding_time)

        assert rate == pytest.approx(0.01)
        assert len(engine.queries) == 1
        sql = engine.queries[0]
        assert "trading_events.funding_rates" in sql
        assert "2026-01-01 16:00:00" in sql
        assert "LIMIT 1" in sql
        assert "O\\'Neil/USDT" in sql
        assert "BINANCE_FUTURES" in sql

    def test_no_rows_returns_none(self) -> None:
        """Keine Zeilen → None (kein Fehler)."""
        source = ClickHouseFundingRateSource(FakeCHEngine(rows=[]))

        assert source.settled_rate(BTC, "BINANCE_FUTURES", datetime(2026, 1, 1, 8, 0, tzinfo=UTC)) is None

    def test_engine_failure_returns_none(self) -> None:
        """Engine-Fehler (z.B. fehlende Tabelle/CH down) → None (Fail-Soft, nie fatal)."""
        source = ClickHouseFundingRateSource(FakeCHEngine(error=RuntimeError("table not found")))

        assert source.settled_rate(BTC, "BINANCE_FUTURES", datetime(2026, 1, 1, 8, 0, tzinfo=UTC)) is None


class TestFloor8h:
    """_floor_8h rundet auf den letzten 00:00/08:00/16:00-UTC-Grenzwert."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC), datetime(2026, 1, 1, 0, 0, tzinfo=UTC)),
            (datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC), datetime(2026, 1, 1, 8, 0, tzinfo=UTC)),
            (datetime(2026, 1, 1, 3, 27, tzinfo=UTC), datetime(2026, 1, 1, 0, 0, tzinfo=UTC)),
            (datetime(2026, 1, 1, 9, 0, tzinfo=UTC), datetime(2026, 1, 1, 8, 0, tzinfo=UTC)),
            (datetime(2026, 1, 1, 17, 59, tzinfo=UTC), datetime(2026, 1, 1, 16, 0, tzinfo=UTC)),
        ],
    )
    def test_floor_cases(self, raw: datetime, expected: datetime) -> None:
        """Exakte Grenzwerte bleiben erhalten, Zeiten werden abgerundet."""
        assert _floor_8h(raw) == expected
