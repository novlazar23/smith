from __future__ import annotations

import pytest

from trading_harness.models import PaperPositionStatus, PaperTrade
from trading_harness.services.position_manager import (
    PositionManager,
    _check_trigger,
    _realized_pnl,
    _unrealized_pnl,
)
from trading_harness.services.position_stores import InMemoryPaperPositionStore


def _make_trade(
    *,
    trade_id: str = "trade-1",
    run_id: str = "run-1",
    symbol: str = "BTC/USD",
    side: str = "LONG",
    equity: float = 100000.0,
    entry_price: float = 50000.0,
    leverage: float = 1.0,
    requested_quantity: float = 1.0,
    stop_price: float = 45000.0,
    target_price: float = 55000.0,
    fill_rate: float = 0.8,
    slippage_bps: float = 0.0,
) -> PaperTrade:
    actual_quantity = round(requested_quantity * fill_rate, 10)
    slippage_factor = 1.0 + (slippage_bps / 10000)
    if side.upper() == "SHORT":
        slippage_factor = 1.0 - (slippage_bps / 10000)
    actual_price = round(entry_price * slippage_factor, 2)
    return PaperTrade(
        trade_id=trade_id,
        run_id=run_id,
        symbol=symbol,
        side=side,
        equity=equity,
        entry_price=entry_price,
        requested_leverage=leverage,
        requested_quantity=requested_quantity,
        actual_quantity=actual_quantity,
        actual_price=actual_price,
        stop_price=stop_price,
        target_price=target_price,
        fill_rate=fill_rate,
        slippage_bps=slippage_bps,
    )


# ---------------------------------------------------------------------------
# Open position
# ---------------------------------------------------------------------------


def test_open_long_position():
    pm = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0)
    pm.open_position(trade)
    positions = pm.get_open_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.symbol == "BTC/USD"
    assert pos.side == "LONG"
    assert pos.entry_price == 50000.0
    assert pos.quantity == 1.6  # 2.0 * 0.8 fill_rate
    assert pos.status == PaperPositionStatus.OPEN


def test_open_short_position():
    pm = PositionManager()
    trade = _make_trade(
        side="SHORT",
        entry_price=50000.0,
        requested_quantity=1.0,
        stop_price=55000.0,
        target_price=45000.0,
    )
    pm.open_position(trade)
    positions = pm.get_open_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.side == "SHORT"
    assert pos.entry_price == 50000.0
    assert pos.quantity == 0.8  # 1.0 * 0.8


# ---------------------------------------------------------------------------
# Unrealized PnL — update_price
# ---------------------------------------------------------------------------


def test_update_price_long_unrealized_pnl_positive():
    pm = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0)
    pos = pm.open_position(trade)
    pm.update_price(pos.id, 52000.0)
    assert pos.unrealized_pnl == pytest.approx((52000 - 50000) * 1.6)


def test_update_price_long_unrealized_pnl_negative():
    pm = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0)
    pos = pm.open_position(trade)
    pm.update_price(pos.id, 48000.0)
    assert pos.unrealized_pnl == pytest.approx((48000 - 50000) * 1.6)


def test_update_price_short_unrealized_pnl_positive():
    pm = PositionManager()
    trade = _make_trade(
        side="SHORT", entry_price=50000.0, requested_quantity=2.0,
        stop_price=55000.0, target_price=45000.0,
    )
    pos = pm.open_position(trade)
    pm.update_price(pos.id, 48000.0)
    assert pos.unrealized_pnl == pytest.approx((50000 - 48000) * 1.6)


def test_update_price_short_unrealized_pnl_negative():
    pm = PositionManager()
    trade = _make_trade(
        side="SHORT", entry_price=50000.0, requested_quantity=2.0,
        stop_price=55000.0, target_price=45000.0,
    )
    pos = pm.open_position(trade)
    pm.update_price(pos.id, 52000.0)
    assert pos.unrealized_pnl == pytest.approx((50000 - 52000) * 1.6)


def test_update_price_closes_position_returns_none():
    pm = PositionManager()
    trade = _make_trade()
    pos = pm.open_position(trade)
    pm.close_position(pos.id, 51000.0)
    result = pm.update_price(pos.id, 52000.0)
    assert result is None


def test_update_price_missing_position_returns_none():
    pm = PositionManager()
    result = pm.update_price("nonexistent", 100.0)
    assert result is None


# ---------------------------------------------------------------------------
# Manual close — realized PnL
# ---------------------------------------------------------------------------


def test_close_long_manual_realized_pnl_positive():
    pm = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0)
    pos = pm.open_position(trade)
    closed = pm.close_position(pos.id, 52000.0, reason="MANUAL")
    assert closed is not None
    assert closed.status == PaperPositionStatus.CLOSED
    assert closed.close_price == 52000.0
    assert closed.close_reason == "MANUAL"
    expected = (52000 - 50000) * 1.6 - pos.fees
    assert closed.realized_pnl == pytest.approx(expected)


def test_close_short_manual_realized_pnl_positive():
    pm = PositionManager()
    trade = _make_trade(
        side="SHORT", entry_price=50000.0, requested_quantity=2.0,
        stop_price=55000.0, target_price=45000.0,
    )
    pos = pm.open_position(trade)
    closed = pm.close_position(pos.id, 48000.0, reason="MANUAL")
    assert closed is not None
    assert closed.status == PaperPositionStatus.CLOSED
    expected = (50000 - 48000) * 1.6 - pos.fees
    assert closed.realized_pnl == pytest.approx(expected)


def test_close_long_manual_realized_pnl_negative():
    pm = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0)
    pos = pm.open_position(trade)
    closed = pm.close_position(pos.id, 47000.0, reason="MANUAL")
    assert closed is not None
    assert closed.status == PaperPositionStatus.CLOSED
    expected = (47000 - 50000) * 1.6 - pos.fees
    assert closed.realized_pnl == pytest.approx(expected)


def test_close_missing_position_returns_none():
    pm = PositionManager()
    result = pm.close_position("nonexistent", 100.0)
    assert result is None


# ---------------------------------------------------------------------------
# Stop-loss / Target triggers
# ---------------------------------------------------------------------------


def test_stop_loss_long():
    pm = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0, stop_price=45000.0)
    pos = pm.open_position(trade)
    trigger = pm.check_stop_loss_target(pos.id, 44000.0)
    assert trigger is not None
    assert trigger["reason"] == "STOP_LOSS"
    assert trigger["action"] == "close"
    closed = pm._store.get(pos.id)
    assert closed.status == PaperPositionStatus.STOPPED_OUT
    assert closed.close_reason == "STOP_LOSS"


def test_stop_loss_short():
    pm = PositionManager()
    trade = _make_trade(
        side="SHORT", entry_price=50000.0, requested_quantity=2.0,
        stop_price=55000.0, target_price=45000.0,
    )
    pos = pm.open_position(trade)
    trigger = pm.check_stop_loss_target(pos.id, 56000.0)
    assert trigger is not None
    assert trigger["reason"] == "STOP_LOSS"
    closed = pm._store.get(pos.id)
    assert closed.status == PaperPositionStatus.STOPPED_OUT


def test_target_hit_long():
    pm = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0, target_price=55000.0)
    pos = pm.open_position(trade)
    trigger = pm.check_stop_loss_target(pos.id, 56000.0)
    assert trigger is not None
    assert trigger["reason"] == "TARGET_HIT"
    closed = pm._store.get(pos.id)
    assert closed.status == PaperPositionStatus.TARGET_HIT
    assert closed.close_reason == "TARGET_HIT"


def test_target_hit_short():
    pm = PositionManager()
    trade = _make_trade(
        side="SHORT", entry_price=50000.0, requested_quantity=2.0,
        stop_price=55000.0, target_price=45000.0,
    )
    pos = pm.open_position(trade)
    trigger = pm.check_stop_loss_target(pos.id, 44000.0)
    assert trigger is not None
    assert trigger["reason"] == "TARGET_HIT"
    closed = pm._store.get(pos.id)
    assert closed.status == PaperPositionStatus.TARGET_HIT


def test_no_trigger_within_range():
    pm = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0, stop_price=45000.0, target_price=55000.0)
    pos = pm.open_position(trade)
    trigger = pm.check_stop_loss_target(pos.id, 51000.0)
    assert trigger is None
    # Price should still be updated
    updated = pm._store.get(pos.id)
    assert updated.current_price == 51000.0


def test_trigger_on_missing_position_returns_none():
    pm = PositionManager()
    result = pm.check_stop_loss_target("nonexistent", 100.0)
    assert result is None


# ---------------------------------------------------------------------------
# Partial close
# ---------------------------------------------------------------------------


def test_partial_close_long_half():
    pm = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0)
    pos = pm.open_position(trade)
    initial_qty = pos.quantity
    closed = pm.partial_close(pos.id, 0.5, 52000.0, reason="PARTIAL_CLOSE")
    assert closed is not None
    assert closed.quantity == pytest.approx(initial_qty * 0.5)
    assert closed.realized_pnl > 0


def test_partial_close_short_half():
    pm = PositionManager()
    trade = _make_trade(
        side="SHORT", entry_price=50000.0, requested_quantity=2.0,
        stop_price=55000.0, target_price=45000.0,
    )
    pos = pm.open_position(trade)
    initial_qty = pos.quantity
    closed = pm.partial_close(pos.id, 0.5, 48000.0, reason="PARTIAL_CLOSE")
    assert closed is not None
    assert closed.quantity == pytest.approx(initial_qty * 0.5)
    assert closed.realized_pnl > 0


def test_partial_close_full_closes_position():
    pm = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0)
    pos = pm.open_position(trade)
    closed = pm.partial_close(pos.id, 1.0, 52000.0, reason="PARTIAL_CLOSE")
    assert closed is not None
    assert closed.quantity == pytest.approx(0.0)
    assert closed.status == PaperPositionStatus.CLOSED


def test_partial_close_invalid_fraction_returns_none():
    pm = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0)
    pos = pm.open_position(trade)
    result = pm.partial_close(pos.id, 1.5, 52000.0)
    assert result is None
    result = pm.partial_close(pos.id, 0.0, 52000.0)
    assert result is None


# ---------------------------------------------------------------------------
# get_open_positions
# ---------------------------------------------------------------------------


def test_get_open_positions_returns_only_open():
    pm = PositionManager()
    trade1 = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0, trade_id="t1")
    trade2 = _make_trade(side="LONG", entry_price=60000.0, requested_quantity=1.0, trade_id="t2")
    pos1 = pm.open_position(trade1)
    pm.open_position(trade2)
    pm.close_position(pos1.id, 51000.0)
    open_positions = pm.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].trade_id == "t2"


def test_get_open_positions_empty():
    pm = PositionManager()
    assert pm.get_open_positions() == []


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_open_same_outputs():
    pm1 = PositionManager()
    pm2 = PositionManager()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=2.0, trade_id="det-1")
    pos1 = pm1.open_position(trade)
    pos2 = pm2.open_position(trade)
    assert pos1.symbol == pos2.symbol
    assert pos1.side == pos2.side
    assert pos1.entry_price == pos2.entry_price
    assert pos1.quantity == pos2.quantity
    assert pos1.fees == pos2.fees
    assert pos1.status == pos2.status


def test_deterministic_pnl_calculations():
    assert _unrealized_pnl("LONG", 52000.0, 50000.0, 1.6) == pytest.approx(3200.0)
    assert _unrealized_pnl("SHORT", 48000.0, 50000.0, 1.6) == pytest.approx(3200.0)
    assert _realized_pnl("LONG", 52000.0, 50000.0, 1.6, 10.0) == pytest.approx(3190.0)
    assert _realized_pnl("SHORT", 48000.0, 50000.0, 1.6, 10.0) == pytest.approx(3190.0)


# ---------------------------------------------------------------------------
# InMemoryPaperPositionStore
# ---------------------------------------------------------------------------


def test_store_add_and_get():
    store = InMemoryPaperPositionStore()
    trade = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=1.0)
    pm = PositionManager(store)
    pos = pm.open_position(trade)
    retrieved = store.get(pos.id)
    assert retrieved is not None
    assert retrieved.id == pos.id


def test_store_get_open():
    store = InMemoryPaperPositionStore()
    pm = PositionManager(store)
    trade1 = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=1.0, trade_id="so-1")
    trade2 = _make_trade(side="LONG", entry_price=60000.0, requested_quantity=1.0, trade_id="so-2")
    pos1 = pm.open_position(trade1)
    pm.open_position(trade2)
    pm.close_position(pos1.id, 51000.0)
    open_positions = store.get_open()
    assert len(open_positions) == 1
    assert open_positions[0].trade_id == "so-2"


def test_store_get_open_with_run_filter():
    store = InMemoryPaperPositionStore()
    pm = PositionManager(store)
    trade1 = _make_trade(side="LONG", entry_price=50000.0, requested_quantity=1.0, trade_id="rf-1", run_id="run-a")
    trade2 = _make_trade(side="LONG", entry_price=60000.0, requested_quantity=1.0, trade_id="rf-2", run_id="run-b")
    pm.open_position(trade1)
    pm.open_position(trade2)
    open_a = store.get_open(run_id="run-a")
    open_b = store.get_open(run_id="run-b")
    assert len(open_a) == 1
    assert len(open_b) == 1
    assert open_a[0].trade_id == "rf-1"
    assert open_b[0].trade_id == "rf-2"


# ---------------------------------------------------------------------------
# _check_trigger edge cases
# ---------------------------------------------------------------------------


def test_check_trigger_no_stop_or_target():
    result = _check_trigger("LONG", 51000.0, 0.0, 0.0)
    assert result is None


def test_check_trigger_at_exact_stop_long():
    result = _check_trigger("LONG", 45000.0, 45000.0, 55000.0)
    assert result is not None
    assert result["reason"] == "STOP_LOSS"


def test_check_trigger_at_exact_stop_short():
    result = _check_trigger("SHORT", 55000.0, 55000.0, 45000.0)
    assert result is not None
    assert result["reason"] == "STOP_LOSS"