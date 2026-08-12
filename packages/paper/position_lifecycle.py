"""Position lifecycle management — Open → Partial Fill → Full Fill → Close → PnL."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class PositionStatus(StrEnum):
    """Lifecycle status of a position."""

    OPEN = "OPEN"
    PARTIAL_FILL = "PARTIAL_FILL"
    FULL_FILL = "FULL_FILL"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    CLOSED_PROFIT = "CLOSED_PROFIT"
    CLOSED_LOSS = "CLOSED_LOSS"


@dataclass
class Fill:
    """Represents a single fill (execution) of a position."""

    fill_id: str
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def notional(self) -> float:
        """Calculate notional value of this fill."""
        return self.quantity * self.price

    @property
    def total_cost(self) -> float:
        """Total cost including commission and slippage."""
        return self.notional + self.commission + (self.quantity * self.slippage)


@dataclass
class PositionLifecycle:
    """Tracks the full lifecycle of a trading position."""

    symbol: str
    target_quantity: float = 0.0
    current_quantity: float = 0.0
    avg_entry_price: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    fills: list[Fill] = field(default_factory=list)
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    max_drawdown: float = 0.0
    _total_invested: float = 0.0

    def add_fill(self, fill: Fill) -> dict[str, Any]:
        """Add a fill to the position and update state.

        Args:
            fill: The fill to add.

        Returns:
            Dict with updated position state including status and PnL.
        """
        # Determine if this is a buy or sell fill based on sign
        is_buy = fill.quantity > 0

        if self.opened_at is None:
            self.opened_at = fill.timestamp

        if is_buy:
            # Update average entry price (volume-weighted)
            total_qty = self.current_quantity + fill.quantity
            if total_qty > 0:
                old_cost = self.avg_entry_price * self.current_quantity
                new_cost = fill.price * fill.quantity
                self.avg_entry_price = (old_cost + new_cost) / total_qty
            self.current_quantity += fill.quantity

            # Track invested amount
            self._total_invested += fill.notional

        else:
            # Sell fill — calculate realized PnL
            # Long position: PnL = (sell_price - avg_entry) * qty
            sell_qty = abs(fill.quantity)
            price_diff = fill.price - self.avg_entry_price
            realized = price_diff * sell_qty
            self.realized_pnl += realized
            self.current_quantity -= sell_qty

        # Track costs
        self.total_commission += fill.commission
        self.total_slippage += fill.quantity * fill.slippage

        self.fills.append(fill)

        # Update status based on lifecycle
        self._update_status()

        # Calculate current equity and drawdown
        if self.current_quantity > 0:
            self.current_equity = self.current_quantity * self.avg_entry_price
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
            if self.peak_equity > 0:
                dd = (self.peak_equity - self.current_equity) / self.peak_equity
                self.max_drawdown = max(self.max_drawdown, dd)

        return self.get_state()

    def _update_status(self) -> None:
        """Update position status based on current state."""
        if self.current_quantity <= 0:
            if self.realized_pnl > 0:
                self.status = PositionStatus.CLOSED_PROFIT
            elif self.realized_pnl < 0:
                self.status = PositionStatus.CLOSED_LOSS
            else:
                self.status = PositionStatus.CLOSED
            self.closed_at = datetime.now(UTC)
        elif self.current_quantity < abs(self.target_quantity):
            self.status = PositionStatus.PARTIAL_FILL
        elif self.current_quantity >= abs(self.target_quantity):
            self.status = PositionStatus.FULL_FILL

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL based on current market price.

        Args:
            current_price: Current market price of the instrument.

        Returns:
            Unrealized PnL amount.
        """
        if self.current_quantity <= 0:
            return 0.0
        # For long positions: (current - avg_entry) * quantity
        pnl = (current_price - self.avg_entry_price) * self.current_quantity
        self.unrealized_pnl = pnl
        return pnl

    def close_position(self, close_price: float) -> Fill:
        """Close the entire position at the given price.

        Args:
            close_price: Price at which to close the position.

        Returns:
            The closing fill with realized PnL.
        """
        if self.current_quantity <= 0:
            raise ValueError("No open position to close")

        fill = Fill(
            fill_id="close",
            quantity=-self.current_quantity,
            price=close_price,
            commission=0.0,
        )
        self.add_fill(fill)
        return fill

    def get_state(self) -> dict[str, Any]:
        """Get current position state as a dictionary.

        Returns:
            Dict with all position state information.
        """
        return {
            "symbol": self.symbol,
            "target_quantity": self.target_quantity,
            "current_quantity": self.current_quantity,
            "avg_entry_price": self.avg_entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "status": str(self.status),
            "num_fills": len(self.fills),
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }

    def get_exposure(self, current_price: float) -> dict[str, float]:
        """Calculate position exposure metrics.

        Args:
            current_price: Current market price.

        Returns:
            Dict with notional, risk, and leverage exposure.
        """
        self.calculate_unrealized_pnl(current_price)
        notional = self.current_quantity * current_price
        return {
            "notional": notional,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "commission_drag": self.total_commission,
            "max_drawdown": self.max_drawdown,
        }

    def get_summary(self, current_price: float) -> str:
        """Get a human-readable summary of the position.

        Args:
            current_price: Current market price for PnL calculation.

        Returns:
            Formatted string summary.
        """
        self.calculate_unrealized_pnl(current_price)
        state = self.get_state()
        lines = [
            f"Position: {self.symbol}",
            f"  Status:      {state['status']}",
            f"  Qty:         {state['current_quantity']:.4f} / {state['target_quantity']:.4f}",
            f"  Avg Entry:   {state['avg_entry_price']:.4f}",
            f"  Realized PnL: {state['realized_pnl']:.2f}",
            f"  Unrealized:  {state['unrealized_pnl']:.2f}",
            f"  Total PnL:   {state['total_pnl']:.2f}",
            f"  Commission:  {state['total_commission']:.2f}",
            f"  Drawdown:    {state['max_drawdown']:.2%}",
        ]
        return "\n".join(lines)
