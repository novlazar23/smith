from __future__ import annotations

from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from trading_harness.models import PaperPosition, PaperTrade
from trading_harness.services.position_stores import (
    InMemoryPaperPositionStore,
    PaperPositionStatus,
    PaperPositionStore,
)

_FEE_RATE = 0.001


def _unrealized_pnl(side: str, current_price: float, entry_price: float, quantity: float) -> float:
    if side.upper() == "LONG":
        return (current_price - entry_price) * quantity
    else:
        return (entry_price - current_price) * quantity


def _realized_pnl(side: str, exit_price: float, entry_price: float, quantity: float, fees: float) -> float:
    if side.upper() == "LONG":
        return (exit_price - entry_price) * quantity - fees
    else:
        return (entry_price - exit_price) * quantity - fees


def _check_trigger(
    side: str,
    current_price: float,
    stop_price: float,
    target_price: float,
) -> dict[str, Any] | None:
    if side.upper() == "LONG":
        if stop_price > 0 and current_price <= stop_price:
            return {"action": "close", "price": current_price, "reason": "STOP_LOSS"}
        if target_price > 0 and current_price >= target_price:
            return {"action": "close", "price": current_price, "reason": "TARGET_HIT"}
    else:
        if stop_price > 0 and current_price >= stop_price:
            return {"action": "close", "price": current_price, "reason": "STOP_LOSS"}
        if target_price > 0 and current_price <= target_price:
            return {"action": "close", "price": current_price, "reason": "TARGET_HIT"}
    return None


class PositionManager:
    """Manages paper trading positions: open, update, close, and triggers.

    Thread-safe via RLock. Uses a PaperPositionStore for persistence
    (in-memory by default, PostgreSQL-backed when available).
    """

    def __init__(self, store: PaperPositionStore | None = None) -> None:
        self._store: PaperPositionStore
        if store is not None:
            self._store = store
        else:
            self._store = InMemoryPaperPositionStore()
        self._lock = RLock()

    def open_position(self, paper_trade: PaperTrade) -> PaperPosition:
        fees = paper_trade.actual_quantity * paper_trade.actual_price * _FEE_RATE
        position = PaperPosition(
            id=f"paper-pos-{uuid4()}",
            trade_id=paper_trade.trade_id,
            run_id=paper_trade.run_id,
            symbol=paper_trade.symbol,
            side=paper_trade.side,
            entry_price=paper_trade.actual_price,
            quantity=paper_trade.actual_quantity,
            fees=fees,
            stop_price=paper_trade.stop_price,
            target_price=paper_trade.target_price,
            status=PaperPositionStatus.OPEN,
        )
        with self._lock:
            self._store.add(position)
        return position

    def update_price(self, position_id: str, current_price: float) -> PaperPosition | None:
        with self._lock:
            position = self._store.get(position_id)
            if position is None or position.status != PaperPositionStatus.OPEN:
                return None
            position.current_price = current_price
            position.unrealized_pnl = _unrealized_pnl(
                position.side,
                current_price,
                position.entry_price,
                position.quantity,
            )
            self._store.add(position)
            return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "MANUAL",
    ) -> PaperPosition | None:
        with self._lock:
            position = self._store.get(position_id)
            if position is None or position.status != PaperPositionStatus.OPEN:
                return None

            position.realized_pnl = _realized_pnl(
                position.side,
                exit_price,
                position.entry_price,
                position.quantity,
                position.fees,
            )
            position.current_price = exit_price
            position.status = PaperPositionStatus.CLOSED
            position.close_price = exit_price
            position.close_reason = reason
            position.close_timestamp = datetime.now(UTC)

            if reason == "STOP_LOSS":
                position.status = PaperPositionStatus.STOPPED_OUT
            elif reason == "TARGET_HIT":
                position.status = PaperPositionStatus.TARGET_HIT

            self._store.add(position)
            return position

    def check_stop_loss_target(
        self,
        position_id: str,
        current_price: float,
    ) -> dict[str, Any] | None:
        with self._lock:
            position = self._store.get(position_id)
            if position is None or position.status != PaperPositionStatus.OPEN:
                return None

            trigger = _check_trigger(
                position.side,
                current_price,
                position.stop_price,
                position.target_price,
            )
            if trigger is None:
                position.current_price = current_price
                position.unrealized_pnl = _unrealized_pnl(
                    position.side,
                    current_price,
                    position.entry_price,
                    position.quantity,
                )
                self._store.add(position)
                return None

            self.close_position(position_id, trigger["price"], trigger["reason"])
            return trigger

    def partial_close(
        self,
        position_id: str,
        fraction: float,
        exit_price: float,
        reason: str = "PARTIAL_CLOSE",
    ) -> PaperPosition | None:
        with self._lock:
            position = self._store.get(position_id)
            if position is None or position.status != PaperPositionStatus.OPEN:
                return None

            if not (0 < fraction <= 1.0):
                return None

            close_quantity = position.quantity * fraction
            remaining_quantity = position.quantity - close_quantity

            fees_for_close = close_quantity * exit_price * _FEE_RATE
            realized = _realized_pnl(
                position.side,
                exit_price,
                position.entry_price,
                close_quantity,
                fees_for_close,
            )

            position.fees += fees_for_close
            position.realized_pnl += realized
            position.quantity = remaining_quantity
            position.current_price = exit_price

            if remaining_quantity <= 0:
                position.status = PaperPositionStatus.CLOSED
                position.close_price = exit_price
                position.close_reason = reason
                position.close_timestamp = datetime.now(UTC)
            else:
                position.unrealized_pnl = _unrealized_pnl(
                    position.side,
                    exit_price,
                    position.entry_price,
                    remaining_quantity,
                )

            self._store.add(position)
            return position

    def get_open_positions(self) -> list[PaperPosition]:
        return self._store.get_open()