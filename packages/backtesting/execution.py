"""Execution models and fill simulation for backtesting.

Provides market, limit, and stop-limit execution with realistic
slippage, partial fills, and order rejections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .core import Candle

# ─── Enums ──────────────────────────────────────────────────────────────────


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class FillStatus(Enum):
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


# ─── Fill Data ──────────────────────────────────────────────────────────────


@dataclass
class SimulatedFill:
    """Represents a single fill (order execution)."""

    fill_id: str
    instrument: str
    side: Side
    order_quantity: float
    filled_quantity: float
    fill_price: float
    commission: float
    slippage_cost: float
    timestamp: datetime
    status: FillStatus = FillStatus.FILLED
    reason: str = ""

    @property
    def notional(self) -> float:
        return self.filled_quantity * self.fill_price

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "instrument": self.instrument,
            "side": self.side.value,
            "order_quantity": self.order_quantity,
            "filled_quantity": self.filled_quantity,
            "fill_price": self.fill_price,
            "commission": self.commission,
            "slippage_cost": self.slippage_cost,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "reason": self.reason,
        }


# ─── Order ──────────────────────────────────────────────────────────────────


@dataclass
class BacktestOrder:
    """An order placed during backtest."""

    order_id: str
    instrument: str
    side: Side
    quantity: float
    price: float | None  # None for market orders
    order_type: OrderType
    timestamp: datetime
    stop_price: float | None = None  # For stop-limit / stop-market
    status: FillStatus = FillStatus.FILLED
    fill: SimulatedFill | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── Commission Models ──────────────────────────────────────────────────────


class CommissionModel:
    """Base class for commission calculation."""

    def calculate(self, notional: float, side: Side) -> float:
        raise NotImplementedError


class FixedCommissionModel(CommissionModel):
    """Fixed fee per trade, regardless of notional."""

    def __init__(self, fee: float = 0.0) -> None:
        self.fee = fee

    def calculate(self, notional: float, side: Side) -> float:
        return self.fee


class PercentageCommissionModel(CommissionModel):
    """Percentage-based commission (e.g., 0.1% of notional)."""

    def __init__(self, rate: float = 0.001) -> None:
        if not (0 <= rate <= 0.1):
            raise ValueError(f"Commission rate must be in [0, 0.1], got {rate}")
        self.rate = rate

    def calculate(self, notional: float, side: Side) -> float:
        return notional * self.rate


class TieredCommissionModel(CommissionModel):
    """Tiered commission based on trade volume.

    Lower commission for larger notional values (maker/taker style).
    """

    def __init__(self, tiers: list[tuple[float, float]] | None = None) -> None:
        """Tiers as list of (min_notional, rate). Sorted ascending by notional.

        Example: [(0, 0.001), (10_000, 0.0008), (100_000, 0.0005)]
        """
        self.tiers = tiers or [(0, 0.001), (10_000, 0.0008), (100_000, 0.0005)]

    def calculate(self, notional: float, side: Side) -> float:
        rate = self.tiers[0][1]
        for min_notional, tier_rate in self.tiers:
            if notional >= min_notional:
                rate = tier_rate
        return notional * rate


# ─── Slippage Models ────────────────────────────────────────────────────────


class SlippageModel:
    """Base class for slippage calculation."""

    def calculate(self, price: float, side: Side, volume: float = 0.0) -> float:
        """Returns the slippage cost (added to price for buy, subtracted for sell)."""
        raise NotImplementedError


class FixedSlippageModel(SlippageModel):
    """Fixed amount of slippage per unit."""

    def __init__(self, amount: float = 0.01) -> None:
        self.amount = amount

    def calculate(self, price: float, side: Side, volume: float = 0.0) -> float:
        return self.amount


class PercentageSlippageModel(SlippageModel):
    """Percentage-based slippage (e.g., 5 bps)."""

    def __init__(self, bps: float = 5.0) -> None:
        if not (0 <= bps <= 1000):
            raise ValueError(f"BPS must be in [0, 1000], got {bps}")
        self.bps = bps

    def calculate(self, price: float, side: Side, volume: float = 0.0) -> float:
        return price * (self.bps / 10_000)


class VolumeBasedSlippageModel(SlippageModel):
    """Slippage increases with trade volume relative to market depth."""

    def __init__(self, base_bps: float = 2.0, depth: float = 1_000_000) -> None:
        self.base_bps = base_bps
        self.depth = depth

    def calculate(self, price: float, side: Side, volume: float = 0.0) -> float:
        ratio = volume / self.depth if self.depth > 0 else 0
        bps = self.base_bps * (1 + ratio)
        return price * (bps / 10_000)


# ─── Execution Models ───────────────────────────────────────────────────────


class ExecutionModel:
    """Base execution model. Simulates order fill behavior."""

    def __init__(
        self,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
    ) -> None:
        self.commission_model = commission_model or FixedCommissionModel(0.0)
        self.slippage_model = slippage_model or PercentageSlippageModel(5.0)

    def execute(
        self,
        order: BacktestOrder,
        candle: Candle,
    ) -> SimulatedFill:
        raise NotImplementedError


class MarketExecutionModel(ExecutionModel):
    """Market order fills at next bar open with slippage."""

    def execute(
        self,
        order: BacktestOrder,
        candle: Candle,
    ) -> SimulatedFill:
        # Market orders fill at next bar open
        fill_price = candle.open

        # Apply slippage
        slippage = self.slippage_model.calculate(
            fill_price, order.side, volume=candle.volume
        )

        if order.side == Side.BUY:
            fill_price += slippage
        else:
            fill_price -= slippage

        commission = self.commission_model.calculate(
            order.quantity * fill_price, order.side
        )

        return SimulatedFill(
            fill_id=order.order_id,
            instrument=order.instrument,
            side=order.side,
            order_quantity=order.quantity,
            filled_quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slippage * order.quantity,
            timestamp=order.timestamp,
            status=FillStatus.FILLED,
        )


class LimitExecutionModel(ExecutionModel):
    """Limit order fills only if price reaches limit.

    For BUY: fills at or below limit price.
    For SELL: fills at or above limit price.
    """

    def execute(
        self,
        order: BacktestOrder,
        candle: Candle,
    ) -> SimulatedFill:
        limit_price = order.price
        if limit_price is None:
            return SimulatedFill(
                fill_id=order.order_id,
                instrument=order.instrument,
                side=order.side,
                order_quantity=0,
                filled_quantity=0,
                fill_price=0,
                commission=0,
                slippage_cost=0,
                timestamp=order.timestamp,
                status=FillStatus.REJECTED,
                reason="no_limit_price",
            )

        if order.side == Side.BUY:
            can_fill = candle.low <= limit_price
            fill_price = min(candle.open, limit_price)
        else:
            can_fill = candle.high >= limit_price
            fill_price = max(candle.open, limit_price)

        if not can_fill:
            return SimulatedFill(
                fill_id=order.order_id,
                instrument=order.instrument,
                side=order.side,
                order_quantity=order.quantity,
                filled_quantity=0,
                fill_price=0,
                commission=0,
                slippage_cost=0,
                timestamp=order.timestamp,
                status=FillStatus.REJECTED,
                reason="price_not_reached",
            )

        slippage = self.slippage_model.calculate(fill_price, order.side)
        if order.side == Side.BUY:
            fill_price += slippage
        else:
            fill_price -= slippage

        commission = self.commission_model.calculate(
            order.quantity * fill_price, order.side
        )

        return SimulatedFill(
            fill_id=order.order_id,
            instrument=order.instrument,
            side=order.side,
            order_quantity=order.quantity,
            filled_quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slippage * order.quantity,
            timestamp=order.timestamp,
            status=FillStatus.FILLED,
        )


class StopLimitExecutionModel(ExecutionModel):
    """Stop-limit order: triggers at stop price, then becomes limit order."""

    def execute(
        self,
        order: BacktestOrder,
        candle: Candle,
    ) -> SimulatedFill:
        if order.stop_price is None:
            return SimulatedFill(
                fill_id=order.order_id,
                instrument=order.instrument,
                side=order.side,
                order_quantity=0,
                filled_quantity=0,
                fill_price=0,
                commission=0,
                slippage_cost=0,
                timestamp=order.timestamp,
                status=FillStatus.REJECTED,
                reason="no_stop_price",
            )

        # Check if stop was triggered
        stop_triggered = False
        if order.side == Side.BUY:
            stop_triggered = candle.high >= order.stop_price
        else:
            stop_triggered = candle.low <= order.stop_price

        if not stop_triggered:
            return SimulatedFill(
                fill_id=order.order_id,
                instrument=order.instrument,
                side=order.side,
                order_quantity=order.quantity,
                filled_quantity=0,
                fill_price=0,
                commission=0,
                slippage_cost=0,
                timestamp=order.timestamp,
                status=FillStatus.REJECTED,
                reason="stop_not_triggered",
            )

        # Stop triggered — becomes limit order at limit price
        limit_price = order.price
        if limit_price is None:
            limit_price = candle.close

        if order.side == Side.BUY:
            fill_price = min(candle.close, limit_price)
        else:
            fill_price = max(candle.close, limit_price)

        slippage = self.slippage_model.calculate(fill_price, order.side)
        if order.side == Side.BUY:
            fill_price += slippage
        else:
            fill_price -= slippage

        commission = self.commission_model.calculate(
            order.quantity * fill_price, order.side
        )

        return SimulatedFill(
            fill_id=order.order_id,
            instrument=order.instrument,
            side=order.side,
            order_quantity=order.quantity,
            filled_quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slippage * order.quantity,
            timestamp=order.timestamp,
            status=FillStatus.FILLED,
        )
