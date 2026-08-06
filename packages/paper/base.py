"""Base types for paper trading: Trade, PaperPosition, PaperAccount."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class TradeDirection(StrEnum):
    """Direction of a trade order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Type of order to execute."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass
class Trade:
    """Represents a single filled trade."""

    trade_id: str
    instrument: str
    direction: TradeDirection
    order_type: OrderType
    quantity: float
    price: float
    slippage: float = 0.0
    commission: float = 0.0
    filled_price: float = 0.0
    filled_quantity: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, filled, cancelled

    @property
    def notional_value(self) -> float:
        """Calculate notional value from filled trade data."""
        return (
            abs(self.filled_quantity * self.filled_price)
            if self.filled_quantity > 0
            else 0.0
        )


@dataclass
class PaperPosition:
    """Represents a position held in a paper account."""

    symbol: str
    quantity: float
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    opened_at: datetime | None = None
    closed_at: datetime | None = None

    @property
    def market_value(self) -> float:
        """Calculate market value from position data."""
        return abs(self.quantity) * self.avg_price if self.avg_price > 0 else 0.0


@dataclass
class PaperAccount:
    """Represents a paper trading account with positions and PnL tracking."""

    account_id: str
    cash: float
    initial_cash: float
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    total_trades: int = 0
    total_commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def equity(self) -> float:
        """Total equity = cash + all position market values."""
        pos_value = sum(p.market_value for p in self.positions.values())
        return self.cash + pos_value

    @property
    def unrealized_pnl(self) -> float:
        """Sum of all unrealized P&L across positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        """Sum of all realized P&L across positions."""
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        """Total P&L = realized + unrealized."""
        return self.realized_pnl + self.unrealized_pnl
