"""Deterministic live PnL tracker.

Computes realized PnL from opposite-side fills and optional unrealized
PnL from injected mark prices.  Pure computation — no exchange calls,
no network access, no invented estimates.

Position tracking is keyed by ``(venue, symbol)``:

- signed position quantity (buy = positive, sell = negative)
- average entry price for the current position
- realized PnL accumulated when an opposite-side fill closes a position
- closed-trade PnL list (for win rate / profit factor)

PnL of a closed quantity:

- long closed by sell:  ``closed_qty * (exit_price - avg_entry_price)``
- short closed by buy:  ``closed_qty * (avg_entry_price - exit_price)``

Only fills with ``filled_quantity > 0`` and a non-``None`` ``fill_price``
are processed.  Every :class:`~packages.live_execution.gateway.OrderResult`
is processed once per order with its cumulative ``filled_quantity``.

# ponytail: repeated partial-fill updates for the same order are not
# supported yet — process the cumulative snapshot once per order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from packages.live_execution.gateway import OrderResult

#: Trading periods per year used to annualize daily ratios.
_ANNUAL_PERIODS = 252.0


class PnlSummary(TypedDict):
    """Result of :meth:`LivePnlTracker.summary`."""

    realized: float
    unrealized: float
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    win_rate: float
    profit_factor: float | None


class DailyPnlRow(TypedDict):
    """Result of :meth:`LivePnlTracker.daily` (one row per date)."""

    date: str
    pnl: float
    realized: float
    unrealized: float


@dataclass
class _Position:
    """Open position: signed quantity and average entry price.

    ``quantity > 0`` is a long, ``quantity < 0`` is a short.
    """

    quantity: float
    avg_price: float


class LivePnlTracker:
    """Deterministic PnL tracker for live order fills.

    Feed one :class:`~packages.live_execution.gateway.OrderResult` per
    fill via :meth:`process_order`, then query :meth:`summary` or
    :meth:`daily`.  Mark prices are injected explicitly by the caller —
    the tracker never fetches prices itself.
    """

    def __init__(self) -> None:
        self._positions: dict[tuple[str, str], _Position] = {}
        self._realized_by_day: dict[str, float] = {}
        self._closed_trades: list[float] = []
        self._latest_day: str | None = None

    # ── input ────────────────────────────────────────────────────────

    def process_order(self, result: OrderResult) -> None:
        """Apply one order result (cumulative fill snapshot).

        Orders with ``filled_quantity <= 0`` or ``fill_price is None``
        are ignored — they carry no actionable fill.
        """
        if result.filled_quantity <= 0 or result.fill_price is None:
            return

        # ponytail: real fill timestamps are not available yet —
        # submitted_at is used as the fill-time proxy for daily buckets.
        day = result.submitted_at.date().isoformat()
        if self._latest_day is None or day > self._latest_day:
            self._latest_day = day

        key = (result.venue, result.symbol)
        signed = result.filled_quantity if result.side == "buy" else -result.filled_quantity
        fill_price = result.fill_price

        position = self._positions.get(key)
        if position is None:
            self._positions[key] = _Position(quantity=signed, avg_price=fill_price)
            return

        if position.quantity * signed >= 0:
            # Same side — blend the average entry price.
            total = position.quantity + signed
            position.avg_price = (
                abs(position.quantity) * position.avg_price + abs(signed) * fill_price
            ) / abs(total)
            position.quantity = total
            return

        # Opposite side — realize PnL on the closed portion.
        close_qty = min(abs(signed), abs(position.quantity))
        if position.quantity > 0:
            pnl = close_qty * (fill_price - position.avg_price)
        else:
            pnl = close_qty * (position.avg_price - fill_price)
        self._closed_trades.append(pnl)
        self._realized_by_day[day] = self._realized_by_day.get(day, 0.0) + pnl

        previous = position.quantity
        position.quantity = previous + signed
        if position.quantity == 0:
            del self._positions[key]
        elif (position.quantity > 0) != (previous > 0):
            # Fill larger than the position — the remainder is a fresh
            # opposite-side position entered at the fill price.
            position.avg_price = fill_price

    # ── queries ──────────────────────────────────────────────────────

    def summary(
        self,
        mark_prices: Mapping[tuple[str, str], float] | None = None,
    ) -> PnlSummary:
        """Return the PnL summary with risk-adjusted metrics.

        Keys: ``realized``, ``unrealized``, ``sharpe``, ``sortino``,
        ``max_drawdown``, ``win_rate``, ``profit_factor``.

        Args:
            mark_prices: Optional ``(venue, symbol) -> mark price``
                mapping for unrealized PnL.  Positions without a mark
                price contribute ``0.0`` — never an estimate.
        """
        daily = self._daily_realized()
        total_trades = len(self._closed_trades)
        win_rate = (
            sum(1 for trade in self._closed_trades if trade > 0) / total_trades
            if total_trades
            else 0.0
        )
        gross_profit = sum(trade for trade in self._closed_trades if trade > 0)
        gross_loss = -sum(trade for trade in self._closed_trades if trade < 0)
        profit_factor: float | None = (
            gross_profit / gross_loss if gross_loss > 0 else None
        )

        return {
            "realized": sum(self._realized_by_day.values()),
            "unrealized": self._unrealized(mark_prices),
            "sharpe": _sharpe(daily),
            "sortino": _sortino(daily),
            "max_drawdown": _max_drawdown(daily),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    def daily(
        self,
        mark_prices: Mapping[tuple[str, str], float] | None = None,
    ) -> list[DailyPnlRow]:
        """Return daily PnL rows sorted by date descending.

        Each row has ``date`` (``YYYY-MM-DD``), ``pnl`` (net),
        ``realized`` and ``unrealized``.  Realized PnL is bucketed by the
        fill-time proxy (see :meth:`process_order`).  The current
        unrealized PnL is reported on the date of the most recently
        processed order only.
        """
        rows: dict[str, dict[str, float]] = {
            day: {"realized": value, "unrealized": 0.0}
            for day, value in self._realized_by_day.items()
        }

        unrealized = self._unrealized(mark_prices)
        if unrealized != 0.0 and self._latest_day is not None:
            # ponytail: historical unrealized PnL is not available without
            # historical mark prices — current unrealized is attributed to
            # the latest order date only.
            row = rows.setdefault(self._latest_day, {"realized": 0.0, "unrealized": 0.0})
            row["unrealized"] = unrealized

        return [
            {
                "date": day,
                "pnl": row["realized"] + row["unrealized"],
                "realized": row["realized"],
                "unrealized": row["unrealized"],
            }
            for day, row in sorted(rows.items(), reverse=True)
        ]

    # ── internals ────────────────────────────────────────────────────

    def _unrealized(
        self, mark_prices: Mapping[tuple[str, str], float] | None
    ) -> float:
        """Mark-to-market PnL of open positions; 0.0 without mark prices."""
        if mark_prices is None:
            return 0.0
        total = 0.0
        for key, position in self._positions.items():
            mark = mark_prices.get(key)
            if mark is None:
                continue
            if position.quantity > 0:
                total += position.quantity * (mark - position.avg_price)
            else:
                total += -position.quantity * (position.avg_price - mark)
        return total

    def _daily_realized(self) -> list[float]:
        """Realized PnL per calendar day, sorted by date ascending."""
        return [value for _, value in sorted(self._realized_by_day.items())]


# ── Risk metrics ─────────────────────────────────────────────────────────────


def _sharpe(daily: list[float]) -> float | None:
    """Annualized Sharpe ratio (zero risk-free rate).

    ``None`` when fewer than 2 points or the daily std is zero.
    """
    if len(daily) < 2:
        return None
    mean = sum(daily) / len(daily)
    variance = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return None
    return (mean / std) * math.sqrt(_ANNUAL_PERIODS)


def _sortino(daily: list[float], target: float = 0.0) -> float | None:
    """Annualized Sortino ratio using downside deviation vs. *target*.

    ``None`` when fewer than 2 points or there is no downside.
    """
    if len(daily) < 2:
        return None
    mean = sum(daily) / len(daily)
    downside_var = sum(min(x - target, 0.0) ** 2 for x in daily) / len(daily)
    downside_std = math.sqrt(downside_var)
    if downside_std == 0:
        return None
    return (mean / downside_std) * math.sqrt(_ANNUAL_PERIODS)


def _max_drawdown(daily: list[float]) -> float:
    """Peak-to-trough drawdown ratio (0.0-1.0) over cumulative daily PnL."""
    peak = 0.0
    cumulative = 0.0
    max_dd = 0.0
    for value in daily:
        cumulative += value
        if cumulative > peak:
            peak = cumulative
        if peak > 0:
            max_dd = max(max_dd, (peak - cumulative) / peak)
    return max_dd
