from __future__ import annotations

import threading
import time

import pytest

from trading_harness.models import (
    PaperTrade,
    PaperTradeStatus,
    TradeProposal,
)
from trading_harness.services.paper_exchange import PaperExchange
from trading_harness.services.paper_trade_store import (
    InMemoryPaperTradeStore,
    PersistedPaperTradeStore,
)


def _make_exchange(fill_rate=0.8, fee_rate=0.001):
    policy = {
        "allowed_symbols": ["BTCUSDT", "ETHUSDT"],
        "max_risk_per_trade": 0.005,
        "max_daily_loss": 0.02,
        "max_portfolio_risk": 0.04,
        "max_leverage": 2.0,
        "max_positions": 5,
        "minimum_risk_reward": 1.8,
        "max_slippage_bps": 20,
    }
    from trading_harness.services.risk_engine import RiskEngine

    store = InMemoryPaperTradeStore()
    return PaperExchange(
        fill_rate=fill_rate,
        fee_rate=fee_rate,
        risk_engine=RiskEngine(policy),
        stores=store,
    )


def _make_proposal(**overrides):
    values = {
        "decision_id": "test-decision-1",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "equity": 10000.0,
        "entry_price": 100.0,
        "stop_price": 98.0,
        "target_price": 104.0,
        "requested_leverage": 1.0,
        "open_positions": 0,
        "current_daily_loss_fraction": 0.0,
        "current_portfolio_risk_fraction": 0.0,
        "expected_slippage_bps": 5.0,
        "requested_quantity": 100.0,
    }
    values.update(overrides)
    return TradeProposal(**values)


# ============================================================
# PaperTrade model tests
# ============================================================


def test_paper_trade_defaults():
    trade = PaperTrade(
        trade_id="d1",
        run_id="run-1",
        symbol="BTCUSDT",
        side="LONG",
        equity=10000.0,
        entry_price=100.0,
        stop_price=98.0,
    )
    assert trade.status == PaperTradeStatus.PENDING
    assert trade.actual_quantity == 0.0
    assert trade.actual_price == 0.0
    assert trade.fees == 0.0
    assert trade.reject_reason is None
    assert trade.partial_fills == []


def test_paper_trade_status_enum():
    assert PaperTradeStatus.PENDING == "PENDING"
    assert PaperTradeStatus.FILLED == "FILLED"
    assert PaperTradeStatus.PARTIALLY_FILLED == "PARTIALLY_FILLED"
    assert PaperTradeStatus.REJECTED == "REJECTED"
    assert PaperTradeStatus.CANCELLED == "CANCELLED"


# ============================================================
# InMemoryPaperTradeStore tests
# ============================================================


def test_in_memory_store_add_and_get():
    store = InMemoryPaperTradeStore()
    trade = PaperTrade(
        trade_id="d1",
        run_id="run-1",
        symbol="BTCUSDT",
        side="LONG",
        equity=10000.0,
        entry_price=100.0,
        stop_price=98.0,
    )
    store.add(trade)
    result = store.get(trade.id)
    assert result is not None
    assert result.id == trade.id
    assert result.symbol == "BTCUSDT"


def test_in_memory_store_get_nonexistent():
    store = InMemoryPaperTradeStore()
    assert store.get("nonexistent-id") is None


def test_in_memory_store_by_run():
    store = InMemoryPaperTradeStore()
    t1 = store.add(
        PaperTrade(
            trade_id="d1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            equity=10000.0,
            entry_price=100.0,
            stop_price=98.0,
        )
    )
    store.add(
        PaperTrade(
            trade_id="d2",
            run_id="run-2",
            symbol="ETHUSDT",
            side="SHORT",
            equity=10000.0,
            entry_price=50.0,
            stop_price=48.0,
        )
    )
    store.add(
        PaperTrade(
            trade_id="d3",
            run_id="run-1",
            symbol="BTCUSDT",
            side="SHORT",
            equity=10000.0,
            entry_price=100.0,
            stop_price=98.0,
        )
    )
    by_run = store.by_run("run-1")
    assert len(by_run) == 2
    ids = {t.id for t in by_run}
    assert t1.id in ids


def test_in_memory_store_by_run_empty():
    store = InMemoryPaperTradeStore()
    assert store.by_run("nonexistent") == []


def test_in_memory_store_by_symbol():
    store = InMemoryPaperTradeStore()
    store.add(
        PaperTrade(
            trade_id="d1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            equity=10000.0,
            entry_price=100.0,
            stop_price=98.0,
        )
    )
    store.add(
        PaperTrade(
            trade_id="d2",
            run_id="run-1",
            symbol="ETHUSDT",
            side="SHORT",
            equity=10000.0,
            entry_price=50.0,
            stop_price=48.0,
        )
    )
    by_sym = store.by_symbol("BTCUSDT")
    assert len(by_sym) == 1


def test_in_memory_store_all():
    store = InMemoryPaperTradeStore()
    store.add(
        PaperTrade(
            trade_id="d1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            equity=10000.0,
            entry_price=100.0,
            stop_price=98.0,
        )
    )
    store.add(
        PaperTrade(
            trade_id="d2",
            run_id="run-2",
            symbol="ETHUSDT",
            side="SHORT",
            equity=10000.0,
            entry_price=50.0,
            stop_price=48.0,
        )
    )
    all_trades = store.all()
    assert len(all_trades) == 2


# ============================================================
# PersistedPaperTradeStore tests (db=None → in-memory fallback)
# ============================================================


def test_persisted_store_fallback_add_and_get():
    store = PersistedPaperTradeStore(db=None)
    trade = PaperTrade(
        trade_id="d1",
        run_id="run-1",
        symbol="BTCUSDT",
        side="LONG",
        equity=10000.0,
        entry_price=100.0,
        stop_price=98.0,
    )
    store.add(trade)
    result = store.get(trade.id)
    assert result is not None
    assert result.id == trade.id


def test_persisted_store_fallback_by_run():
    store = PersistedPaperTradeStore(db=None)
    store.add(
        PaperTrade(
            trade_id="d1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            equity=10000.0,
            entry_price=100.0,
            stop_price=98.0,
        )
    )
    results = store.by_run("run-1")
    assert len(results) == 1


# ============================================================
# PaperExchange integration tests
# ============================================================


def test_execute_long_order():
    exchange = _make_exchange()
    proposal = _make_proposal(
        side="BUY",
        equity=10000.0,
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.status == PaperTradeStatus.FILLED
    assert trade.trade_id == "test-decision-1"
    assert trade.symbol == "BTCUSDT"
    assert trade.side == "BUY"
    # slippage = 100.0 * 5 / 10000 = 0.05
    # actual_price = 100.0 + 0.05 = 100.05
    assert trade.actual_price == 100.05
    # actual_quantity = 100.0 * 0.8 = 80.0
    assert trade.actual_quantity == 80.0
    # fees = 80.0 * 100.05 * 0.001 = 8.004
    assert trade.fees == 8.004


def test_execute_short_order():
    exchange = _make_exchange()
    proposal = _make_proposal(
        side="SELL",
        entry_price=100.0,
        stop_price=102.0,
        target_price=96.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.status == PaperTradeStatus.FILLED
    # slippage = 100.0 * 5 / 10000 = 0.05
    # actual_price = 100.0 - 0.05 = 99.95
    assert trade.actual_price == 99.95
    assert trade.actual_quantity == 80.0


def test_fill_rate_applied():
    exchange = _make_exchange(fill_rate=0.8)
    proposal = _make_proposal(
        side="BUY",
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.actual_quantity == 80.0


def test_fee_calculation():
    exchange = _make_exchange(fee_rate=0.001)
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    # slippage = 0.05, actual_price = 100.05, qty = 80, fees = 80 * 100.05 * 0.001 = 8.004
    assert trade.fees == 8.004


def test_reject_unknown_symbol():
    exchange = _make_exchange()
    proposal = _make_proposal(
        symbol="UNKNOWN",
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.status == PaperTradeStatus.REJECTED
    assert trade.reject_reason == "SYMBOL_NOT_ALLOWED"


def test_store_contains_filled_trade():
    exchange = _make_exchange()
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    stored = exchange.stores.get(trade.id)
    assert stored is not None
    assert stored.status == PaperTradeStatus.FILLED


def test_store_contains_rejected_trade():
    exchange = _make_exchange()
    proposal = _make_proposal(
        symbol="UNKNOWN",
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.status == PaperTradeStatus.REJECTED
    assert trade.reject_reason == "SYMBOL_NOT_ALLOWED"


def test_store_by_run_contains_trades():
    exchange = _make_exchange()
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    exchange.execute_order(proposal, current_price=100.0)
    by_run = exchange.stores.by_run("run-1")
    assert len(by_run) >= 1


def test_deterministic_output():
    exchange = _make_exchange()
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade1 = exchange.execute_order(proposal, current_price=100.0)
    trade2 = exchange.execute_order(proposal, current_price=100.0)
    assert trade1.actual_price == trade2.actual_price
    assert trade1.actual_quantity == trade2.actual_quantity
    assert trade1.fees == trade2.fees
    assert trade1.status == trade2.status


def test_custom_fill_rate():
    exchange = _make_exchange(fill_rate=0.5)
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=200.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.actual_quantity == 100.0  # 200 * 0.5


def test_custom_fee_rate():
    exchange = _make_exchange(fee_rate=0.002)
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    # slippage = 0.05, actual_price = 100.05, qty = 80
    expected = 80.0 * 100.05 * 0.002
    assert trade.fees == expected


def test_fill_rate_override():
    exchange = _make_exchange(fill_rate=0.8)
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(
        proposal, current_price=100.0, fill_rate_override=1.0
    )
    assert trade.actual_quantity == 100.0


def test_concurrent_execution():
    exchange = _make_exchange()
    errors = []

    def execute(n):
        try:
            proposal = _make_proposal(
                decision_id=f"d-{n}",
                side="BUY",
                entry_price=100.0,
                stop_price=98.0,
                target_price=104.0,
                requested_quantity=100.0,
            )
            exchange.execute_order(proposal, current_price=100.0)
        except Exception as exc:  # noqa: BLE001 — concurrent stress test: catch any thread error
            errors.append(exc)

    threads = [threading.Thread(target=execute, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert errors == [], f"Concurrent errors: {errors}"
    all_trades = exchange.stores.all()
    assert len(all_trades) == 20


def test_concurrent_read_write():
    exchange = _make_exchange()
    errors = []

    def writer(n):
        try:
            proposal = _make_proposal(
                decision_id=f"write-{n}",
                side="BUY",
                entry_price=100.0,
                stop_price=98.0,
                target_price=104.0,
                requested_quantity=100.0,
            )
            exchange.execute_order(proposal, current_price=100.0)
        except Exception as exc:  # noqa: BLE001 — concurrent stress test: catch any thread error
            errors.append(exc)

    def reader():
        try:
            for _ in range(50):
                exchange.stores.all()
                time.sleep(0.001)
        except Exception as exc:  # noqa: BLE001 — concurrent stress test: catch any thread error
            errors.append(exc)

    writers = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    readers = [threading.Thread(target=reader) for _ in range(3)]
    all_threads = writers + readers

    for t in all_threads:
        t.start()
    for t in all_threads:
        t.join(timeout=10)

    assert errors == [], f"Concurrent read/write errors: {errors}"

# ============================================================
# Order Lifecycle Tests
# ============================================================


def test_order_lifecycle_pending_to_filled():
    """Order wird vom Standard als FILLED gespeichert."""
    exchange = _make_exchange()
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.status == PaperTradeStatus.FILLED


def test_order_lifecycle_rejected():
    """Order wird bei ungültigem Symbol als REJECTED gespeichert."""
    exchange = _make_exchange()
    proposal = _make_proposal(
        symbol="UNKNOWN",
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.status == PaperTradeStatus.REJECTED
    assert trade.reject_reason == "SYMBOL_NOT_ALLOWED"


def test_cancel_trade_not_found():
    """Stornierung einer nicht gefundenen Order schlägt fehl."""
    exchange = _make_exchange()
    result = exchange.cancel_trade("nonexistent-id")
    assert result["success"] is False
    assert result["error"] == "TRADE_NOT_FOUND"


def test_cancel_trade_stores_not_configured():
    """Stornierung ohne Store schlägt mit error dict zurück."""
    exchange = PaperExchange(fill_rate=0.8, fee_rate=0.001, stores=None)
    result = exchange.cancel_trade("any-id")
    assert result["success"] is False
    assert result["error"] == "STORES_NOT_CONFIGURED"


def test_cancel_filled_trade_fails():
    """Eine FILLED Order kann nicht storniert werden."""
    exchange = _make_exchange()
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.status == PaperTradeStatus.FILLED
    result = exchange.cancel_trade(trade.id)
    assert result["success"] is False
    assert result["error"] == "TRADE_CANNOT_BE_CANCELLED"


def test_cancel_rejected_trade_fails():
    """Eine REJECTED Order kann nicht storniert werden."""
    exchange = _make_exchange()
    proposal = _make_proposal(
        symbol="UNKNOWN",
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.status == PaperTradeStatus.REJECTED
    result = exchange.cancel_trade(trade.id)
    assert result["success"] is False
    assert result["error"] == "TRADE_CANNOT_BE_CANCELLED"


def test_get_trade_returns_stored_trade():
    """get_trade gibt den gespeicherten Trade zurück."""
    exchange = _make_exchange()
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    retrieved = exchange.get_trade(trade.id)
    assert retrieved is not None
    assert retrieved.id == trade.id
    assert retrieved.status == PaperTradeStatus.FILLED
    assert retrieved.symbol == "BTCUSDT"


def test_get_trade_nonexistent_returns_none():
    """get_trade für nicht existierende ID gibt None zurück."""
    exchange = _make_exchange()
    assert exchange.get_trade("nonexistent") is None


def test_by_status_returns_filtered_trades():
    """by_status gibt nur Trades mit dem angegebenen Status zurück."""
    exchange = _make_exchange()
    proposal1 = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    proposal2 = _make_proposal(
        symbol="UNKNOWN",
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    exchange.execute_order(proposal1, current_price=100.0)
    exchange.execute_order(proposal2, current_price=100.0)
    filled = exchange.by_status(PaperTradeStatus.FILLED)
    rejected = exchange.by_status(PaperTradeStatus.REJECTED)
    assert len(filled) == 1
    assert len(rejected) == 1


def test_by_status_empty_for_uncommon_status():
    """by_status für seltenen Status gibt leere Liste zurück."""
    exchange = _make_exchange()
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    exchange.execute_order(proposal, current_price=100.0)
    none_trades = exchange.by_status(PaperTradeStatus.PARTIALLY_FILLED)
    assert len(none_trades) == 0


def test_multiple_orders_have_unique_ids():
    """Jede Order erhält eine eindeutige ID."""
    exchange = _make_exchange()
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade1 = exchange.execute_order(proposal, current_price=100.0)
    trade2 = exchange.execute_order(proposal, current_price=100.0)
    assert trade1.id != trade2.id
    assert trade1.id.startswith("paper-trade-")
    assert trade2.id.startswith("paper-trade-")


# ============================================================
# Fee Calculation Tests
# ============================================================


def test_fee_calculation_zero_fee_rate():
    """Bei fee_rate=0.0 sollten keine Gebühren anfallen."""
    store = InMemoryPaperTradeStore()
    exchange = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=store)
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
        expected_slippage_bps=0.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.fees == 0.0


def test_fee_calculation_default_exchange():
    """Gebühren auf dem Default-Exchange (fill_rate=0.8, fee_rate=0.001)."""
    exchange = _make_exchange()
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    # slippage = 100 * 5 / 10000 = 0.05 -> actual_price = 100.05
    # actual_quantity = 100 * 0.8 = 80.0
    # fees = 80 * 100.05 * 0.001 = 8.004
    assert trade.actual_price == pytest.approx(100.05)
    assert trade.actual_quantity == pytest.approx(80.0)
    assert trade.fees == pytest.approx(8.004)


def test_fee_calculation_high_fee_rate():
    """Hoher fee_rate führt zu proportionalem Gebührenbetrag."""
    store = InMemoryPaperTradeStore()
    exchange = PaperExchange(fill_rate=1.0, fee_rate=0.005, stores=store)
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
        expected_slippage_bps=0.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    # fees = 100 * 100 * 0.005 = 50.0
    assert trade.fees == pytest.approx(50.0)


def test_fee_calculation_short_order():
    """Gebühren für Short-Orders basieren auf Absolutbetrag."""
    store = InMemoryPaperTradeStore()
    exchange = PaperExchange(fill_rate=1.0, fee_rate=0.001, stores=store)
    proposal = _make_proposal(
        side="SELL",
        entry_price=100.0,
        stop_price=102.0,
        target_price=96.0,
        requested_quantity=50.0,
        expected_slippage_bps=0.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    # actual_price = 100.0 (slippage=0), qty=50
    # fees = 50 * 100 * 0.001 = 5.0
    assert trade.fees == pytest.approx(5.0)


def test_fee_calculation_varying_quantity():
    """Gebühren skalieren linear mit der ausgeführten Menge."""
    store1 = InMemoryPaperTradeStore()
    exchange1 = PaperExchange(fill_rate=1.0, fee_rate=0.001, stores=store1)
    store2 = InMemoryPaperTradeStore()
    exchange2 = PaperExchange(fill_rate=1.0, fee_rate=0.001, stores=store2)
    proposal_small = _make_proposal(requested_quantity=10.0, expected_slippage_bps=0.0)
    trade_small = exchange1.execute_order(proposal_small, current_price=100.0)
    proposal_large = _make_proposal(requested_quantity=100.0, expected_slippage_bps=0.0)
    trade_large = exchange2.execute_order(proposal_large, current_price=100.0)
    # fees skalieren linear: 10x Menge -> 10x fees
    assert trade_large.fees == pytest.approx(trade_small.fees * 10)


# ============================================================
# Slippage Tests
# ============================================================


def test_slippage_long_order_adds_to_price():
    """Bei LONG-Orders wird Slippage zum aktuellen Preis addiert."""
    store = InMemoryPaperTradeStore()
    exchange = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=store)
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        expected_slippage_bps=10.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    # slippage = 100 * 10 / 10000 = 0.10
    # actual_price = 100 + 0.10 = 100.10
    assert trade.actual_price == pytest.approx(100.10)
    assert trade.actual_price > 100.0


def test_slippage_short_order_subtracts_from_price():
    """Bei SHORT-Orders wird Slippage vom aktuellen Preis abgezogen."""
    store = InMemoryPaperTradeStore()
    exchange = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=store)
    proposal = _make_proposal(
        side="SELL",
        entry_price=100.0,
        stop_price=102.0,
        target_price=96.0,
        expected_slippage_bps=10.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    # slippage = 100 * 10 / 10000 = 0.10
    # actual_price = 100 - 0.10 = 99.90
    assert trade.actual_price == pytest.approx(99.90)
    assert trade.actual_price < 100.0


def test_slippage_zero_bps_no_adjustment():
    """Bei 0 bps Slippage sollte der Preis unverändert bleiben."""
    store = InMemoryPaperTradeStore()
    exchange = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=store)
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        expected_slippage_bps=0.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    assert trade.actual_price == pytest.approx(100.0)


def test_slippage_high_bps_large_adjustment():
    """Hohe Slippage in bps führt zu großer Preisanpassung."""
    store = InMemoryPaperTradeStore()
    exchange = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=store)
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        expected_slippage_bps=100.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    # slippage = 100 * 100 / 10000 = 1.0
    # actual_price = 100 + 1.0 = 101.0
    assert trade.actual_price == pytest.approx(101.0)


def test_slippage_deterministic_across_calls():
    """Slippage-Berechnung ist deterministisch."""
    store = InMemoryPaperTradeStore()
    exchange = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=store)
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        expected_slippage_bps=15.0,
    )
    prices = []
    for _ in range(10):
        trade = exchange.execute_order(proposal, current_price=100.0)
        prices.append(trade.actual_price)
    assert all(p == prices[0] for p in prices)


def test_slippage_large_price():
    """Slippage bei hohem Preis (z.B. BTC) berechnet korrekt."""
    store = InMemoryPaperTradeStore()
    exchange = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=store)
    proposal = _make_proposal(
        side="BUY",
        entry_price=50000.0,
        stop_price=48000.0,
        target_price=52000.0,
        expected_slippage_bps=5.0,
    )
    trade = exchange.execute_order(proposal, current_price=50000.0)
    # slippage = 50000 * 5 / 10000 = 25.0
    # actual_price = 50000 + 25 = 50025.0
    assert trade.actual_price == pytest.approx(50025.0)


def test_slippage_and_fill_rate_combined():
    """Slippage und Fill Rate wirken unabhängig."""
    store = InMemoryPaperTradeStore()
    exchange = PaperExchange(fill_rate=0.5, fee_rate=0.0, stores=store)
    proposal = _make_proposal(
        side="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        requested_quantity=100.0,
        expected_slippage_bps=10.0,
    )
    trade = exchange.execute_order(proposal, current_price=100.0)
    # slippage = 0.10 -> actual_price = 100.10
    # fill_rate = 0.5 -> actual_quantity = 100 * 0.5 = 50.0
    assert trade.actual_price == pytest.approx(100.10)
    assert trade.actual_quantity == pytest.approx(50.0)


def test_slippage_short_large_price():
    """Short-Slippage bei hohem Preis (z.B. ETH)."""
    store = InMemoryPaperTradeStore()
    exchange = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=store)
    proposal = _make_proposal(
        side="SELL",
        entry_price=3000.0,
        stop_price=3100.0,
        target_price=2900.0,
        expected_slippage_bps=8.0,
    )
    trade = exchange.execute_order(proposal, current_price=3000.0)
    # slippage = 3000 * 8 / 10000 = 2.4
    # actual_price = 3000 - 2.4 = 2997.6
    assert trade.actual_price == pytest.approx(2997.6)
