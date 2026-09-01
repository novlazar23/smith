"""Tests für die Persistenz-Funktionen (Fake-Connection, keine echte DB)."""

from __future__ import annotations

import json

from apps.demo_trader.service import (
    INSERT_DEMO_TRADE,
    UPSERT_DEMO_ACCOUNT,
    persist_account_snapshot,
    persist_demo_trade,
)
from packages.paper import OrderType, Trade, TradeDirection

from .conftest import FakeConnection


def _make_trade() -> Trade:
    """Erzeugt einen deterministischen Paper-Trade."""
    return Trade(
        trade_id="trade-001",
        instrument="BTC/USDT",
        direction=TradeDirection.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        price=200.0,
        slippage=0.001,
        commission=2.002,
        filled_price=200.2,
        filled_quantity=10.0,
        status="filled",
    )


class TestPersistDemoTrade:
    """persist_demo_trade() schreibt eine Zeile nach demo_trades."""

    def test_executes_insert_and_commits(self) -> None:
        """Execute mit korrekten Parametern, danach Commit."""
        conn = FakeConnection()

        persist_demo_trade(conn, _make_trade())

        assert len(conn.executed) == 1
        assert conn.commits == 1
        statement, params = conn.executed[0]
        assert statement is INSERT_DEMO_TRADE
        assert params == {
            "trade_id": "trade-001",
            "instrument": "BTC/USDT",
            "direction": "BUY",
            "quantity": 10.0,
            "price": 200.0,
            "filled_price": 200.2,
            "filled_quantity": 10.0,
            "commission": 2.002,
            "slippage": 0.001,
            "status": "filled",
        }

    def test_sell_trade_direction_serialized_as_value(self) -> None:
        """Die Direction landet als String-Wert (SELL) in der Zeile."""
        conn = FakeConnection()
        trade = Trade(
            trade_id="trade-002",
            instrument="ETH/USDT",
            direction=TradeDirection.SELL,
            order_type=OrderType.MARKET,
            quantity=4.0,
            price=50.0,
            filled_price=49.95,
            filled_quantity=4.0,
            status="filled",
        )

        persist_demo_trade(conn, trade)

        assert conn.executed[0][1]["direction"] == "SELL"


class TestPersistAccountSnapshot:
    """persist_account_snapshot() upsertet die demo_account-Zeile."""

    def _snapshot(self, positions: list[dict] | None = None) -> dict:
        """Baut ein Snapshot-Dict (Form wie build_account_snapshot)."""
        return {
            "account_id": "demo",
            "cash": 98000.0,
            "equity": 99000.0,
            "initial_cash": 100000.0,
            "total_pnl": -1000.0,
            "total_commission": 5.0,
            "total_trades": 3,
            "positions": positions if positions is not None else [],
        }

    def test_executes_upsert_and_commits(self) -> None:
        """Upsert-Statement mit allen Snapshot-Feldern, danach Commit."""
        conn = FakeConnection()

        persist_account_snapshot(conn, self._snapshot())

        assert len(conn.executed) == 1
        assert conn.commits == 1
        statement, params = conn.executed[0]
        assert statement is UPSERT_DEMO_ACCOUNT
        assert params["account_id"] == "demo"
        assert params["cash"] == 98000.0
        assert params["equity"] == 99000.0
        assert params["initial_cash"] == 100000.0
        assert params["total_pnl"] == -1000.0
        assert params["total_commission"] == 5.0
        assert params["total_trades"] == 3

    def test_positions_serialized_as_json_string(self) -> None:
        """positions wird als JSON-String übergeben (→ CAST AS jsonb)."""
        conn = FakeConnection()
        positions = [
            {
                "instrument": "BTC/USDT",
                "quantity": 10.0,
                "avg_price": 200.2,
                "opened_at": "2026-09-01T10:00:00+00:00",
            }
        ]

        persist_account_snapshot(conn, self._snapshot(positions))

        assert json.loads(conn.executed[0][1]["positions"]) == positions
