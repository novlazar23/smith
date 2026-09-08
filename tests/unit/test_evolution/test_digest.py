"""Tests für den Evidenz-Digest (Markdown) und Live-Paper-Graceful-Degradation."""

from __future__ import annotations

import pytest
from apps.evolution.digest import build_digest, fetch_live_paper, write_digest
from apps.evolution.state import EvolutionStore
from tests.unit.test_evolution.conftest import make_hypothesis


def test_fetch_live_paper_returns_none_without_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    monkeypatch.setenv("DB_PORT", "1")
    assert fetch_live_paper() is None


def test_build_digest_empty_store_has_placeholders(store: EvolutionStore) -> None:
    digest = build_digest(store)
    assert "# Evolutions-Digest" in digest
    assert "rsi_mean_reversion" in digest  # Baseline
    assert "(noch keine Promotionen" in digest
    assert "(noch keine Familien)" in digest
    assert "(noch keiner)" in digest
    assert "(leer)" in digest
    assert "(nicht abrufbar in dieser Umgebung)" in digest


def test_build_digest_renders_state_content(store: EvolutionStore) -> None:
    store.add_hypothesis(make_hypothesis(status="rejected"))
    store.promote(
        {
            "id": "rsi_mean_reversion-20240101-1",
            "variant": {"strategy": "rsi_mean_reversion", "params": {"period": 30.0}},
            "claim": "Promoter Claim",
            "promoted_at": "2024-01-01T00:00:00+00:00",
        }
    )
    store.set_meta(
        last_cycle={
            "finished_at": "2024-01-01T00:00:00+00:00",
            "n_tested": 2,
            "n_promoted": 1,
            "n_rejected": 1,
            "n_pending": 0,
            "n_error": 0,
            "verdicts": [{"id": "h-1", "decision": "promoted", "delta_pp": 1.5}],
        }
    )
    store.add_graveyard(
        {
            "id": "g-1",
            "variant_key": "rsi_mean_reversion::buy_below=20.5",
            "reasons": ["OOS-Marge 0.00 pp < gefordert 1.00 pp"],
        }
    )
    digest = build_digest(store)
    assert "| rsi_mean_reversion |" in digest
    assert "rsi_mean_reversion-20240101-1" in digest
    assert "2 getestet" in digest
    assert "+1.50 pp OOS vs Baseline" in digest
    assert "rsi_mean_reversion::buy_below=20.5" in digest
    assert "OOS-Marge 0.00 pp < gefordert 1.00 pp" in digest


def test_build_digest_renders_live_paper(store: EvolutionStore) -> None:
    live = {
        "account": {
            "initial_cash": 100_000.0,
            "equity": 101_000.0,
            "total_pnl": 1_000.0,
            "total_trades": 5,
            "updated_at": "2024-01-01T00:00:00+00:00",
        },
        "recent_trades": [
            {
                "created_at": "2024-01-01T00:00:00+00:00",
                "direction": "BUY",
                "instrument": "BTC/USDT",
                "filled_price": 100.0,
                "status": "filled",
            }
        ],
    }
    digest = build_digest(store, live=live)
    assert "+1.00 %" in digest
    assert "5 Trades" in digest
    assert "BTC/USDT @ 100.0 (filled)" in digest


def test_build_digest_caps_graveyard_to_last_entries(store: EvolutionStore) -> None:
    for i in range(15):
        store.add_graveyard({"id": f"g-{i}", "variant_key": f"gk-{i}", "reasons": []})
    digest = build_digest(store)
    assert "gk-14" in digest
    assert "gk-5" in digest
    assert "gk-4" not in digest
    assert "gk-0" not in digest


def test_write_digest_writes_markdown_file(store: EvolutionStore) -> None:
    digest = build_digest(store)
    path = write_digest(store, digest)
    assert path is not None
    assert path.exists()
    assert path.parent == store.root / "digests"
    assert path.read_text(encoding="utf-8") == digest
