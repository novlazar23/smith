"""Tests für den Account-Snapshot (JSONB-Form der demo_account-Zeile)."""

from __future__ import annotations

from datetime import datetime

import pytest
from apps.demo_trader.service import build_account_snapshot
from packages.paper import PaperAccount, PaperExecutor, TradeDirection


class TestBuildAccountSnapshot:
    """build_account_snapshot() wandelt den PaperAccount in ein Snapshot-Dict."""

    def test_snapshot_without_positions(self) -> None:
        """Leerer Account: alle Felder vorhanden, positions leer."""
        account = PaperAccount(account_id="demo", cash=100000.0, initial_cash=100000.0)

        snapshot = build_account_snapshot(account)

        assert snapshot["account_id"] == "demo"
        assert snapshot["cash"] == 100000.0
        assert snapshot["equity"] == 100000.0
        assert snapshot["initial_cash"] == 100000.0
        assert snapshot["total_pnl"] == 0.0
        assert snapshot["total_commission"] == 0.0
        assert snapshot["total_trades"] == 0
        assert snapshot["positions"] == []

    def test_snapshot_position_shape(self) -> None:
        """Jede Position hat exakt die Schlüssel instrument/quantity/avg_price/opened_at."""
        executor = PaperExecutor(initial_cash=100000.0)
        account = executor.create_account("demo")
        executor.submit_order(account, "BTC/USDT", TradeDirection.BUY, 10.0, 200.0)

        snapshot = build_account_snapshot(account)

        assert len(snapshot["positions"]) == 1
        entry = snapshot["positions"][0]
        assert set(entry) == {"instrument", "quantity", "avg_price", "opened_at"}
        assert entry["instrument"] == "BTC/USDT"
        assert entry["quantity"] == pytest.approx(10.0)
        # Slippage 0.1 % → Durchschnittspreis über 200.0
        assert entry["avg_price"] == pytest.approx(200.2)
        # opened_at ist eine ISO-8601-Zeichenkette (JSON-serialisierbar)
        parsed = datetime.fromisoformat(entry["opened_at"])
        assert parsed.tzinfo is not None

    def test_snapshot_reflects_account_state_after_trades(self) -> None:
        """cash/equity/total_trades/total_commission spiegeln den Account-Zustand."""
        executor = PaperExecutor(initial_cash=100000.0)
        account = executor.create_account("demo")
        buy = executor.submit_order(account, "BTC/USDT", TradeDirection.BUY, 10.0, 200.0)
        close = executor.close_position(account, "BTC/USDT")
        assert close is not None

        snapshot = build_account_snapshot(account)

        assert snapshot["total_trades"] == 2
        assert snapshot["cash"] == pytest.approx(account.cash)
        assert snapshot["equity"] == pytest.approx(account.equity)
        assert snapshot["cash"] < 100000.0  # Kommission + Slippage kosten Geld
        assert snapshot["total_commission"] == pytest.approx(
            buy.commission + close.commission
        )
        assert snapshot["positions"] == []
