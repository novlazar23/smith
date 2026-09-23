"""Tests für die Account-Rehydration nach einem Neustart (demo_account-Snapshot)."""

from __future__ import annotations

from datetime import UTC, datetime

from apps.demo_trader.service import (
    LOAD_DEMO_ACCOUNT,
    LOAD_FUNDING_BOUNDARIES,
    DemoTrader,
    DemoTraderConfig,
)
from packages.paper import PaperExecutor

from .conftest import BTC, FakeDB, StubCandleSource


class _RowShim:
    def __init__(self, data: dict) -> None:
        self._mapping = data


class FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class RehydrateConnection:
    """Fake-Connection, die auf die Rehydration-Queries antwortet."""

    def __init__(
        self,
        account_row: dict | None = None,
        funding_rows: list[list[str]] | None = None,
    ) -> None:
        self._account_row = account_row
        self._funding_rows = funding_rows or []
        self.fail = False
        self.executed: list[tuple[object, dict]] = []

    def execute(self, statement: object, parameters: dict):
        if self.fail:
            raise RuntimeError("db down")
        self.executed.append((statement, parameters))
        if statement is LOAD_DEMO_ACCOUNT:
            rows = [_RowShim(self._account_row)] if self._account_row is not None else []
            return FakeResult(rows)
        if statement is LOAD_FUNDING_BOUNDARIES:
            return FakeResult(list(self._funding_rows))
        return None

    def commit(self) -> None:
        pass


def make_rehydrate_trader(
    config: DemoTraderConfig,
    conn: RehydrateConnection,
) -> DemoTrader:
    return DemoTrader(
        config=config,
        provider=StubCandleSource({}),
        db=FakeDB(conn),
        executor=PaperExecutor(initial_cash=config.initial_cash),
        pipeline_factory=lambda: None,
    )


ACCOUNT_ROW = {
    "cash": 99500.0,
    "equity": 99600.0,
    "initial_cash": 100000.0,
    "total_commission": 42.5,
    "total_trades": 7,
    "positions": [
        {
            "instrument": BTC,
            "quantity": 0.1,
            "avg_price": 50000.0,
            "opened_at": "2026-09-23T10:00:00+00:00",
        }
    ],
}


class TestRehydrateFromDb:
    """Der Neustart stellt Cash, Zähler und Positionen aus dem Snapshot her."""

    def test_rehydrates_account_from_snapshot(self, config: DemoTraderConfig) -> None:
        trader = make_rehydrate_trader(config, RehydrateConnection(account_row=ACCOUNT_ROW))
        account = trader.account

        assert account.cash == 99500.0
        assert account.initial_cash == 100000.0
        assert account.total_trades == 7
        assert account.total_commission == 42.5
        pos = account.positions[BTC]
        assert pos.quantity == 0.1
        assert pos.avg_price == 50000.0
        assert pos.opened_at == datetime(2026, 9, 23, 10, 0, tzinfo=UTC)

    def test_no_row_starts_fresh_account(self, config: DemoTraderConfig) -> None:
        trader = make_rehydrate_trader(config, RehydrateConnection(account_row=None))
        assert trader.account.cash == config.initial_cash
        assert trader.account.positions == {}

    def test_db_error_starts_fresh_account(self, config: DemoTraderConfig) -> None:
        conn = RehydrateConnection()
        conn.fail = True
        trader = make_rehydrate_trader(config, conn)
        assert trader.account.cash == config.initial_cash
        assert trader.account.positions == {}


class TestRehydrateFundingLast:
    """Der Funding-Catch-Up startet am letzten gebuchten Grenzwert (kein Doppel-Settlement)."""

    def test_last_boundary_from_funding_audit_rows(
        self, config: DemoTraderConfig
    ) -> None:
        conn = RehydrateConnection(
            account_row=ACCOUNT_ROW,
            funding_rows=[
                [f"demo-funding-20260923T000000Z-{BTC}"],
                [f"demo-funding-20260923T080000Z-{BTC}"],
            ],
        )
        trader = make_rehydrate_trader(config, conn)
        assert trader._last_funding[BTC] == datetime(2026, 9, 23, 8, 0, tzinfo=UTC)

    def test_fallback_to_opened_at_floor_without_funding_rows(
        self, config: DemoTraderConfig
    ) -> None:
        trader = make_rehydrate_trader(config, RehydrateConnection(account_row=ACCOUNT_ROW))
        # opened_at 10:00 UTC → letzter 8h-Grenzwert davor = 08:00 UTC
        assert trader._last_funding[BTC] == datetime(2026, 9, 23, 8, 0, tzinfo=UTC)
