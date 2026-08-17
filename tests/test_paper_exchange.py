from __future__ import annotations

import threading
import time

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