"""Core data structures for the backtesting engine.

Defines BacktestConfig, BacktestResult, BacktestState, PortfolioSnapshot,
OrderBookSimulator, and the candle/quote data model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

# ─── Data Model ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Candle:
    """One OHLCV candle from historical data."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def mid_price(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)


@dataclass
class Bar:
    """Aggregated bar with computed indicators attached."""

    candle: Candle
    sma_20: float | None = None
    sma_50: float | None = None
    ema_12: float | None = None
    ema_26: float | None = None
    rsi_14: float | None = None
    atr_14: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None


# ─── Portfolio State ────────────────────────────────────────────────────────


@dataclass
class Position:
    """An open position tracked during backtest."""

    instrument: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0

    def close_price(self, current_price: float) -> float:
        """Calculate realized PnL for closing at `current_price`."""
        if self.side == "long":
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity

    def current_pnl(self, current_price: float) -> float:
        """Unrealized PnL at `current_price`."""
        return self.close_price(current_price)


@dataclass
class Trade:
    """A completed trade (fill)."""

    trade_id: str
    instrument: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    pnl: float = 0.0

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of portfolio state."""

    timestamp: datetime
    initial_capital: float
    cash: float
    market_value: float
    total_equity: float
    positions: list[Position] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)

    @property
    def drawdown(self) -> float:
        """Current drawdown as fraction of peak equity."""
        if self.total_equity <= 0:
            return 0.0
        # Drawdown will be set externally after tracking peaks
        return self._drawdown if hasattr(self, "_drawdown") else 0.0

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.quantity * _current_price(p)) for p in self.positions)

    @property
    def net_exposure(self) -> float:
        total = 0.0
        for p in self.positions:
            if p.side == "long":
                total += p.quantity * _current_price(p)
            else:
                total -= p.quantity * _current_price(p)
        return total

    def snapshot_copy(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            timestamp=self.timestamp,
            initial_capital=self.initial_capital,
            cash=self.cash,
            market_value=self.market_value,
            total_equity=self.total_equity,
            positions=list(self.positions),
            trades=list(self.trades),
        )


# ─── Backtest Config ────────────────────────────────────────────────────────


class BacktestConfig(BaseModel):
    """Configuration for a backtest run."""

    initial_capital: float = Field(default=100_000.0, gt=0)
    commission_model: str = "fixed"
    slippage_model: str = "percentage"
    commission_rate: float = Field(default=0.001, ge=0, le=0.1)
    slippage_rate: float = Field(default=0.0005, ge=0, le=0.1)
    commission_fixed: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0, le=1000)
    max_position_size: float = Field(default=0.25, gt=0, le=1)
    max_total_exposure: float = Field(default=1.0, gt=0, le=1)
    initial_cash_ratio: float = Field(default=0.1, gt=0, le=1)
    allow_short: bool = True
    date_start: datetime | None = None
    date_end: datetime | None = None
    warmup_bars: int = Field(default=50, ge=0)
    rebalance_interval_seconds: int = Field(default=3600, ge=60)
    symbol: str = "BTC/USDT"
    timeframe: str = "1d"
    data_provider: str = "csv"
    data_path: str = ""
    metrics: list[str] = field(
        default_factory=lambda: [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "total_trades",
            "avg_trade_return",
            "avg_win_return",
            "avg_loss_return",
            "best_trade",
            "worst_trade",
        ]
    )

    @model_validator(mode="after")
    def validate(self) -> BacktestConfig:
        if self.initial_cash_ratio > 1.0:
            raise ValueError("initial_cash_ratio must be <= 1.0")
        if self.max_position_size > 1.0:
            raise ValueError("max_position_size must be <= 1.0")
        return self


# ─── Backtest Result ────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """Full result of a backtest run."""

    config: BacktestConfig
    candles: list[Candle]
    snapshots: list[PortfolioSnapshot]
    trades: list[Trade]
    metrics: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        if self.snapshots:
            return self.snapshots[-1].total_equity
        return self.config.initial_capital

    @property
    def total_return(self) -> float:
        initial = self.config.initial_capital
        if initial <= 0:
            return 0.0
        return (self.final_equity - initial) / initial

    @property
    def total_trades(self) -> int:
        return len(self.trades)


# ─── Order Book Simulation ──────────────────────────────────────────────────


@dataclass
class LimitOrder:
    """A single limit order in the simulated book."""

    price: float
    quantity: float
    side: str  # "buy" or "sell"
    timestamp: datetime


class OrderBookSimulator:
    """Simple order book simulator for limit/stop-limit orders.

    Maintains a bid/ask queue and fills orders at the first matching
    price in the book.
    """

    def __init__(self) -> None:
        self.bids: list[LimitOrder] = []
        self.asks: list[LimitOrder] = []

    def add_order(self, order: LimitOrder) -> None:
        if order.side == "buy":
            self.bids.append(order)
            self.bids.sort(key=lambda o: o.price, reverse=True)
        else:
            self.asks.append(order)
            self.asks.sort(key=lambda o: o.price)

    def get_best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    def get_best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    def get_mid_price(self) -> float | None:
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None

    def try_fill_buy(self, price: float, quantity: float) -> tuple[float, float]:
        """Try to fill a buy order. Returns (filled_price, filled_quantity)."""
        filled_price = price
        filled_qty = quantity

        remaining = quantity
        for ask in self.asks[:]:
            if remaining <= 0:
                break
            if ask.price <= price:
                fill_qty = min(remaining, ask.quantity)
                filled_price = (
                    (filled_price * (quantity - remaining + fill_qty)
                     + ask.price * fill_qty)
                    / (quantity - remaining + fill_qty)
                    if (quantity - remaining + fill_qty) > 0
                    else price
                )
                ask.quantity -= fill_qty
                remaining -= fill_qty
                if ask.quantity <= 0:
                    self.asks.remove(ask)

        if remaining > 0:
            filled_qty = quantity - remaining
            filled_price = (filled_price * (quantity - remaining) + price * remaining) / quantity

        return filled_price, filled_qty

    def try_fill_sell(self, price: float, quantity: float) -> tuple[float, float]:
        """Try to fill a sell order. Returns (filled_price, filled_quantity)."""
        filled_price = price
        filled_qty = quantity
        remaining = quantity

        for bid in self.bids[:]:
            if remaining <= 0:
                break
            if bid.price >= price:
                fill_qty = min(remaining, bid.quantity)
                filled_price = (
                    (filled_price * (quantity - remaining + fill_qty)
                     + bid.price * fill_qty)
                    / (quantity - remaining + fill_qty)
                    if (quantity - remaining + fill_qty) > 0
                    else price
                )
                bid.quantity -= fill_qty
                remaining -= fill_qty
                if bid.quantity <= 0:
                    self.bids.remove(bid)

        if remaining > 0:
            filled_qty = quantity - remaining
            filled_price = (filled_price * (quantity - remaining) + price * remaining) / quantity

        return filled_price, filled_qty

    def clean(self, timestamp: datetime) -> None:
        """Remove stale orders (older than 5 minutes)."""
        cutoff = timestamp.timestamp() - 300
        self.bids = [b for b in self.bids if b.timestamp.timestamp() > cutoff]
        self.asks = [a for a in self.asks if a.timestamp.timestamp() > cutoff]


# ─── Helpers ────────────────────────────────────────────────────────────────


def _current_price(position: Position) -> float:
    """Return current price for a position (from external source)."""
    return position.entry_price  # Fallback; updated by engine
