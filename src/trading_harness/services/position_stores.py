from __future__ import annotations

from datetime import UTC, datetime
from threading import RLock
from typing import Any, Protocol

from trading_harness.models import PaperPosition, PaperPositionStatus
from trading_harness.services.db import Database


def _record_to_row(position: PaperPosition) -> dict[str, Any]:
    return {
        "id": position.id,
        "trade_id": position.trade_id,
        "run_id": position.run_id,
        "symbol": position.symbol,
        "side": position.side,
        "entry_price": position.entry_price,
        "quantity": position.quantity,
        "fees": position.fees,
        "current_price": position.current_price,
        "unrealized_pnl": position.unrealized_pnl,
        "realized_pnl": position.realized_pnl,
        "stop_price": position.stop_price,
        "target_price": position.target_price,
        "status": position.status.value if isinstance(position.status, PaperPositionStatus) else position.status,
        "open_timestamp": position.open_timestamp.isoformat(),
        "close_timestamp": (
            position.close_timestamp.isoformat() if position.close_timestamp else None
        ),
        "close_price": position.close_price,
        "close_reason": position.close_reason,
    }


def _row_to_record(row: dict[str, Any]) -> PaperPosition:
    return PaperPosition(
        id=row["id"],
        trade_id=row["trade_id"],
        run_id=row["run_id"],
        symbol=row["symbol"],
        side=row["side"],
        entry_price=float(row["entry_price"]),
        quantity=float(row["quantity"]),
        fees=float(row.get("fees", 0.0)),
        current_price=float(row.get("current_price", 0.0)),
        unrealized_pnl=float(row.get("unrealized_pnl", 0.0)),
        realized_pnl=float(row.get("realized_pnl", 0.0)),
        stop_price=float(row.get("stop_price", 0.0)),
        target_price=float(row.get("target_price", 0.0)),
        status=PaperPositionStatus(row.get("status", "OPEN")),
        open_timestamp=datetime.fromisoformat(str(row["open_timestamp"])).replace(tzinfo=UTC),
        close_timestamp=(
            datetime.fromisoformat(str(row["close_timestamp"])).replace(tzinfo=UTC)
            if row.get("close_timestamp")
            else None
        ),
        close_price=float(row["close_price"]) if row.get("close_price") else None,
        close_reason=row.get("close_reason"),
    )


# ---------------------------------------------------------------------------
# Store protocol
# ---------------------------------------------------------------------------


class PaperPositionStore(Protocol):
    """Protocol for paper position stores."""

    def add(self, position: PaperPosition) -> PaperPosition: ...
    def get(self, position_id: str) -> PaperPosition | None: ...
    def get_open(self, run_id: str | None = None) -> list[PaperPosition]: ...
    def all(self) -> list[PaperPosition]: ...


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


class InMemoryPaperPositionStore:
    """Thread-safe in-memory store for paper positions."""

    def __init__(self) -> None:
        self._positions: dict[str, PaperPosition] = {}
        self._lock = RLock()

    def add(self, position: PaperPosition) -> PaperPosition:
        with self._lock:
            self._positions[position.id] = position
        return position

    def get(self, position_id: str) -> PaperPosition | None:
        with self._lock:
            return self._positions.get(position_id)

    def get_open(self, run_id: str | None = None) -> list[PaperPosition]:
        with self._lock:
            positions = [
                p for p in self._positions.values()
                if p.status == PaperPositionStatus.OPEN
            ]
            if run_id:
                positions = [p for p in positions if p.run_id == run_id]
            return positions

    def all(self) -> list[PaperPosition]:
        with self._lock:
            return list(self._positions.values())


# ---------------------------------------------------------------------------
# Persisted store
# ---------------------------------------------------------------------------


class PersistedPaperPositionStore:
    """PostgreSQL-backed store for paper positions.

    Falls back to in-memory store when PostgreSQL is unavailable.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._fallback: dict[str, PaperPosition] = {}

    def add(self, position: PaperPosition) -> PaperPosition:
        if self._db and self._db.is_available:
            row = _record_to_row(position)
            cols = list(row.keys())
            placeholders = ",".join(["%s"] * len(cols))
            self._db.execute_write(
                f"INSERT INTO paper_positions ({','.join(cols)}) "
                f"VALUES ({placeholders}) "
                "ON CONFLICT (id) DO UPDATE SET "
                + ",".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id"),
                *[row[c] for c in cols],
            )
        else:
            self._fallback[position.id] = position
        return position

    def get(self, position_id: str) -> PaperPosition | None:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM paper_positions WHERE id = %s", (position_id,)
            )
            if rows:
                return _row_to_record(rows[0])
            return None
        return self._fallback.get(position_id)

    def get_open(self, run_id: str | None = None) -> list[PaperPosition]:
        if self._db and self._db.is_available:
            if run_id:
                rows = self._db.execute(
                    "SELECT * FROM paper_positions "
                    "WHERE status = 'OPEN' AND run_id = %s",
                    (run_id,),
                )
            else:
                rows = self._db.execute(
                    "SELECT * FROM paper_positions WHERE status = 'OPEN'"
                )
            return [_row_to_record(r) for r in rows]
        positions = [
            p for p in self._fallback.values()
            if p.status == PaperPositionStatus.OPEN
        ]
        if run_id:
            positions = [p for p in positions if p.run_id == run_id]
        return positions

    def all(self) -> list[PaperPosition]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM paper_positions ORDER BY open_timestamp"
            )
            return [_row_to_record(r) for r in rows]
        return list(self._fallback.values())