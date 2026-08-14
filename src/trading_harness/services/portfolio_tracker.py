from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

from trading_harness.models import PaperPosition, PortfolioState


class PortfolioStore:
    """Protocol for portfolio state persistence stores."""

    def add(self, state: PortfolioState) -> PortfolioState: ...

    def get(self, state_id: str) -> PortfolioState | None: ...

    def by_run(self, run_id: str) -> list[PortfolioState]: ...

    def all(self) -> list[PortfolioState]: ...


def _record_to_row(state: PortfolioState) -> dict[str, Any]:
    return {
        "id": state.id,
        "run_id": state.run_id,
        "start_equity": state.start_equity,
        "current_equity": state.current_equity,
        "total_realized_pnl": state.total_realized_pnl,
        "total_unrealized_pnl": state.total_unrealized_pnl,
        "max_drawdown": state.max_drawdown,
        "current_drawdown": state.current_drawdown,
        "peak_equity": state.peak_equity,
        "positions": state.positions,
        "symbols": state.symbols,
        "timestamp": state.timestamp.isoformat(),
    }


def _row_to_state(row: dict[str, Any]) -> PortfolioState:
    return PortfolioState(
        id=row["id"],
        run_id=row["run_id"],
        start_equity=row.get("start_equity", 100000.0),
        current_equity=row.get("current_equity", 100000.0),
        total_realized_pnl=row.get("total_realized_pnl", 0.0),
        total_unrealized_pnl=row.get("total_unrealized_pnl", 0.0),
        max_drawdown=row.get("max_drawdown", 0.0),
        current_drawdown=row.get("current_drawdown", 0.0),
        peak_equity=row.get("peak_equity", 100000.0),
        positions=row.get("positions", {}),
        symbols=row.get("symbols", []),
        timestamp=datetime.fromisoformat(str(row["timestamp"])).replace(tzinfo=UTC),
    )


class InMemoryPortfolioStore:
    """Thread-safe in-memory store for portfolio states."""

    def __init__(self) -> None:
        self._states: dict[str, PortfolioState] = {}
        self._lock = threading.RLock()

    def add(self, state: PortfolioState) -> PortfolioState:
        with self._lock:
            self._states[state.id] = state
        return state

    def get(self, state_id: str) -> PortfolioState | None:
        with self._lock:
            return self._states.get(state_id)

    def by_run(self, run_id: str) -> list[PortfolioState]:
        with self._lock:
            return [
                s for s in self._states.values() if s.run_id == run_id
            ]

    def all(self) -> list[PortfolioState]:
        with self._lock:
            return list(self._states.values())


class PersistedPortfolioStore:
    """PostgreSQL-backed store for portfolio states.

    Falls back to in-memory store when PostgreSQL is unavailable.
    """

    def __init__(self, db: Database | None = None) -> None:  # noqa: F821
        self._db = db
        self._fallback: dict[str, PortfolioState] = {}

    def add(self, state: PortfolioState) -> PortfolioState:
        if self._db and self._db.is_available:
            row = _record_to_row(state)
            cols = list(row.keys())
            placeholders = ",".join(["%s"] * len(cols))
            self._db.execute_write(
                f"INSERT INTO portfolio_states ({','.join(cols)}) "
                f"VALUES ({placeholders}) "
                "ON CONFLICT (id) DO UPDATE SET "
                + ",".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id"),
                *[row[c] for c in cols],
            )
        else:
            self._fallback[state.id] = state
        return state

    def get(self, state_id: str) -> PortfolioState | None:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM portfolio_states WHERE id = %s", (state_id,)
            )
            if rows:
                return _row_to_state(rows[0])
            return None
        return self._fallback.get(state_id)

    def by_run(self, run_id: str) -> list[PortfolioState]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM portfolio_states "
                "WHERE run_id = %s ORDER BY timestamp",
                (run_id,),
            )
            return [_row_to_state(r) for r in rows]
        return [s for s in self._fallback.values() if s.run_id == run_id]

    def all(self) -> list[PortfolioState]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM portfolio_states ORDER BY timestamp"
            )
            return [_row_to_state(r) for r in rows]
        return list(self._fallback.values())


class PortfolioTracker:
    """Calculates and persists portfolio state from paper trading positions.

    Thread-safe via RLock. Equities are tracked over time to compute drawdown.
    """

    def __init__(
        self,
        start_equity: float = 100000.0,
        store: PortfolioStore | None = None,
    ) -> None:
        self._start_equity = start_equity
        self._store = store or InMemoryPortfolioStore()
        self._equity_history: list[float] = [start_equity]
        self._lock = threading.RLock()

    @property
    def start_equity(self) -> float:
        return self._start_equity

    def update(self, positions: list[PaperPosition]) -> PortfolioState:
        """Recalculate portfolio state from current positions and persist it."""
        with self._lock:
            total_unrealized = sum(p.unrealized_pnl for p in positions)
            total_realized = sum(
                p.realized_pnl
                for p in positions
                if p.status.value != "OPEN"
            )
            current_equity = (
                self._start_equity + total_realized + total_unrealized
            )

            self._equity_history.append(current_equity)
            current_drawdown, max_drawdown = self._calculate_drawdown(current_equity)
            peak_equity = max(self._equity_history)

            positions_dict = {
                p.symbol: p.quantity for p in positions
            }
            symbols = list(positions_dict.keys())

            state = PortfolioState(
                run_id="",
                start_equity=self._start_equity,
                current_equity=current_equity,
                total_realized_pnl=total_realized,
                total_unrealized_pnl=total_unrealized,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                peak_equity=peak_equity,
                positions=positions_dict,
                symbols=symbols,
                timestamp=datetime.now(UTC),
            )
            self._store.add(state)
            return state

    def calculate_equity(self) -> float:
        return (
            self._start_equity
            + sum(p.unrealized_pnl for p in [])
            + sum(p.realized_pnl for p in [] if p.status.value != "OPEN")
        )

    def _calculate_drawdown(self, current_equity: float) -> tuple[float, float]:
        peak_equity = max(self._equity_history)
        if peak_equity > 0:
            current_dd = (peak_equity - current_equity) / peak_equity
        else:
            current_dd = 0.0

        all_drawdowns = []
        running_peak = self._equity_history[0]
        for eq in self._equity_history:
            running_peak = max(running_peak, eq)
            if running_peak > 0:
                all_drawdowns.append((running_peak - eq) / running_peak)
            else:
                all_drawdowns.append(0.0)

        max_dd = max(all_drawdowns) if all_drawdowns else 0.0
        return current_dd, max_dd

    def calculate_exposure(self, positions: list[PaperPosition]) -> dict[str, float]:
        exposure: dict[str, float] = {}
        for p in positions:
            notional = abs(p.quantity) * p.entry_price
            exposure[p.symbol] = exposure.get(p.symbol, 0.0) + notional
        return exposure

    def get_state(self, run_id: str) -> PortfolioState | None:
        states = self._store.by_run(run_id)
        if not states:
            return None
        return max(states, key=lambda s: s.timestamp)

    def get_history(self, run_id: str) -> list[PortfolioState]:
        return self._store.by_run(run_id)