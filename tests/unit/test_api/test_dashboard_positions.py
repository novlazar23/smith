"""Positionen-Reader des Dashboards (Quelle: demo_account, keine Trade-Ableitung)."""

from __future__ import annotations

import json
from typing import Any

import apps.api.routers.dashboard as dashboard
import pytest

MARKET = [
    {"instrument": "BTC/USDT", "last_price": 85000.0},
    {"instrument": "ETH/USDT", "last_price": 2600.0},
]

ACCOUNT_ROW = {
    "positions": [
        {
            "instrument": "BTC/USDT",
            "quantity": 0.001,
            "avg_price": 84000.0,
            "opened_at": "2026-09-24T07:35:00+00:00",
        }
    ]
}


def _patch_pg(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []

    def fake_pg_rows(sql: str, *args: Any) -> list[dict[str, Any]]:
        seen.append(sql)
        return rows

    monkeypatch.setattr(dashboard, "_pg_rows", fake_pg_rows)
    return seen


class TestFetchPositions:
    def test_reads_positions_from_account(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _patch_pg(monkeypatch, [ACCOUNT_ROW])
        positions = dashboard._fetch_positions(MARKET)

        assert len(positions) == 1
        entry = positions[0]
        assert entry["instrument"] == "BTC/USDT"
        assert entry["quantity"] == pytest.approx(0.001)
        assert entry["avg_price"] == 84000.0
        assert entry["market_price"] == 85000.0
        assert entry["unrealized_pnl"] == pytest.approx((85000.0 - 84000.0) * 0.001, rel=1e-6)
        assert entry["opened_at"] == "2026-09-24T07:35:00Z"
        # Quelle ist das Account-Snapshot, nicht die Trade-Historie
        assert "demo_trades" not in seen[0]
        assert "demo_account" in seen[0]

    def test_empty_positions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_pg(monkeypatch, [{"positions": []}])
        assert dashboard._fetch_positions(MARKET) == []

    def test_missing_account_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_pg(monkeypatch, [])
        assert dashboard._fetch_positions(MARKET) == []

    def test_positions_as_json_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        row = {"positions": json.dumps(ACCOUNT_ROW["positions"])}
        _patch_pg(monkeypatch, [row])
        assert len(dashboard._fetch_positions(MARKET)) == 1

    def test_corrupt_positions_json_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_pg(monkeypatch, [{"positions": "{kaputt"}])
        assert dashboard._fetch_positions(MARKET) == []

    def test_no_market_price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_pg(monkeypatch, [ACCOUNT_ROW])
        positions = dashboard._fetch_positions([{"instrument": "SOL/USDT", "last_price": 100.0}])
        assert positions[0]["market_price"] is None
        assert positions[0]["unrealized_pnl"] is None

    def test_skips_non_positive_quantity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        row = {"positions": [{"instrument": "BTC/USDT", "quantity": 0.0, "avg_price": 1.0}]}
        _patch_pg(monkeypatch, [row])
        assert dashboard._fetch_positions(MARKET) == []
