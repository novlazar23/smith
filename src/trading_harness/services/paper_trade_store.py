from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from psycopg.types.json import Jsonb

from trading_harness.models import PaperTrade

if TYPE_CHECKING:
    from trading_harness.services.db import Database


class PaperTradeStore(Protocol):
    """Protocol for paper trade persistence stores."""

    def add(self, trade: PaperTrade) -> PaperTrade: ...
    def get(self, trade_id: str) -> PaperTrade | None: ...
    def by_run(self, run_id: str) -> list[PaperTrade]: ...
    def by_symbol(self, symbol: str) -> list[PaperTrade]: ...
    def all(self) -> list[PaperTrade]: ...


def _record_to_row(record: PaperTrade) -> dict:
    return {
        "id": record.id,
        "trade_id": record.trade_id,
        "run_id": record.run_id,
        "symbol": record.symbol,
        "side": record.side,
        "equity": record.equity,
        "entry_price": record.entry_price,
        "requested_leverage": record.requested_leverage,
        "requested_quantity": record.requested_quantity,
        "actual_quantity": record.actual_quantity,
        "actual_price": record.actual_price,
        "stop_price": record.stop_price,
        "target_price": record.target_price,
        "fill_rate": record.fill_rate,
        "slippage_bps": record.slippage_bps,
        "fees": record.fees,
        "status": record.status,
        "partial_fills": Jsonb(record.partial_fills),
        "created_at": record.created_at.isoformat(),
        "filled_at": record.filled_at.isoformat() if record.filled_at else None,
        "closed_at": record.closed_at.isoformat() if record.closed_at else None,
        "reject_reason": record.reject_reason,
    }


def _row_to_record(row: dict) -> PaperTrade:
    return PaperTrade(
        id=row["id"],
        trade_id=row["trade_id"],
        run_id=row["run_id"],
        symbol=row["symbol"],
        side=row["side"],
        equity=row["equity"],
        entry_price=row["entry_price"],
        requested_leverage=row.get("requested_leverage", 1.0),
        requested_quantity=row["requested_quantity"],
        actual_quantity=row.get("actual_quantity", 0.0),
        actual_price=row.get("actual_price", 0.0),
        stop_price=row["stop_price"],
        target_price=row.get("target_price", 0.0),
        fill_rate=row.get("fill_rate", 0.8),
        slippage_bps=row.get("slippage_bps", 0.0),
        fees=row.get("fees", 0.0),
        status=row["status"],
        partial_fills=row.get("partial_fills", []),
        created_at=datetime.fromisoformat(str(row["created_at"])).replace(tzinfo=UTC),
        filled_at=(
            datetime.fromisoformat(str(row["filled_at"])).replace(tzinfo=UTC)
            if row.get("filled_at")
            else None
        ),
        closed_at=(
            datetime.fromisoformat(str(row["closed_at"])).replace(tzinfo=UTC)
            if row.get("closed_at")
            else None
        ),
        reject_reason=row.get("reject_reason"),
    )


class InMemoryPaperTradeStore:
    """Thread-safe in-memory store for paper trades."""

    def __init__(self) -> None:
        self._trades: dict[str, PaperTrade] = {}
        self._lock = threading.RLock()

    def add(self, trade: PaperTrade) -> PaperTrade:
        with self._lock:
            self._trades[trade.id] = trade
        return trade

    def get(self, trade_id: str) -> PaperTrade | None:
        with self._lock:
            return self._trades.get(trade_id)

    def by_run(self, run_id: str) -> list[PaperTrade]:
        with self._lock:
            return [t for t in self._trades.values() if t.run_id == run_id]

    def by_symbol(self, symbol: str) -> list[PaperTrade]:
        with self._lock:
            return [t for t in self._trades.values() if t.symbol == symbol]

    def all(self) -> list[PaperTrade]:
        with self._lock:
            return list(self._trades.values())


class PersistedPaperTradeStore:
    """PostgreSQL-backed store for paper trades.

    Falls back to in-memory store when PostgreSQL is unavailable.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._fallback: dict[str, PaperTrade] = {}

    def add(self, trade: PaperTrade) -> PaperTrade:
        if self._db and self._db.is_available:
            row = _record_to_row(trade)
            cols = list(row.keys())
            placeholders = ",".join(["%s"] * len(cols))
            self._db.execute_write(
                f"INSERT INTO paper_trades ({','.join(cols)}) "
                f"VALUES ({placeholders}) "
                "ON CONFLICT (id) DO UPDATE SET "
                + ",".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id"),
                *[row[c] for c in cols],
            )
        else:
            self._fallback[trade.id] = trade
        return trade

    def get(self, trade_id: str) -> PaperTrade | None:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM paper_trades WHERE id = %s", (trade_id,)
            )
            if rows:
                return _row_to_record(rows[0])
            return None
        return self._fallback.get(trade_id)

    def by_run(self, run_id: str) -> list[PaperTrade]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM paper_trades WHERE run_id = %s ORDER BY created_at",
                (run_id,),
            )
            return [_row_to_record(r) for r in rows]
        return [t for t in self._fallback.values() if t.run_id == run_id]

    def by_symbol(self, symbol: str) -> list[PaperTrade]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM paper_trades WHERE symbol = %s ORDER BY created_at",
                (symbol,),
            )
            return [_row_to_record(r) for r in rows]
        return [t for t in self._fallback.values() if t.symbol == symbol]

    def all(self) -> list[PaperTrade]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM paper_trades ORDER BY created_at"
            )
            return [_row_to_record(r) for r in rows]
        return list(self._fallback.values())