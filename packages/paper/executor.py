"""Paper trading executor — handles order submission, execution, and position management."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from .base import OrderType, PaperAccount, PaperPosition, Trade, TradeDirection


class PaperExecutor:
    """Simulates trade execution with slippage and commissions."""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        default_slippage_pct: float = 0.001,
        default_commission_pct: float = 0.001,
        max_position_size_pct: float = 0.10,
    ) -> None:
        """Initialize the executor with default parameters.

        Args:
            initial_cash: Starting cash for new accounts.
            default_slippage_pct: Fraction of price added for market impact (0.1% = 10bps).
            default_commission_pct: Fraction of notional value charged per trade.
            max_position_size_pct: Maximum position as fraction of equity (10%).
        """
        self.initial_cash = initial_cash
        self.default_slippage_pct = default_slippage_pct
        self.default_commission_pct = default_commission_pct
        self.max_position_size_pct = max_position_size_pct
        self._accounts: dict[str, PaperAccount] = {}

    def create_account(self, account_id: str) -> PaperAccount:
        """Create a new paper trading account.

        Args:
            account_id: Unique identifier for the account.

        Returns:
            The newly created PaperAccount.
        """
        account = PaperAccount(
            account_id=account_id,
            cash=self.initial_cash,
            initial_cash=self.initial_cash,
        )
        self._accounts[account_id] = account
        return account

    def _get_max_allowed_quantity(self, account: PaperAccount, price: float) -> float:
        """Calculate the maximum allowed quantity based on position size limit."""
        max_notional = account.equity * self.max_position_size_pct
        if price <= 0:
            return 0.0
        return max_notional / price

    def submit_order(
        self,
        account: PaperAccount,
        instrument: str,
        direction: TradeDirection,
        quantity: float,
        price: float | None = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> Trade:
        """Submit a trade order for execution.

        For MARKET orders the price is the current market price.
        Slippage is applied as a fraction of the market price.
        Commission is calculated on the filled notional value.

        Args:
            account: The paper account to execute against.
            instrument: Symbol of the instrument to trade.
            direction: BUY or SELL.
            quantity: Number of units to trade.
            price: Current market price (required for MARKET orders).
            order_type: Type of order.

        Returns:
            The filled Trade object.

        Raises:
            ValueError: If there is insufficient cash (BUY) or position (SELL).
        """
        if price is None or price <= 0:
            raise ValueError("Market price must be provided and positive")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        trade_id = str(uuid.uuid4())
        slippage = self.default_slippage_pct

        # Calculate filled price with slippage
        if direction == TradeDirection.BUY:
            filled_price = price * (1 + slippage)
        else:
            filled_price = price * (1 - slippage)

        commission = filled_price * quantity * self.default_commission_pct

        trade = Trade(
            trade_id=trade_id,
            instrument=instrument,
            direction=direction,
            order_type=order_type,
            quantity=quantity,
            price=price,
            slippage=slippage,
            commission=commission,
            filled_price=filled_price,
            filled_quantity=quantity,
            status="filled",
        )

        if direction == TradeDirection.BUY:
            total_cost = filled_price * quantity + commission
            if account.cash < total_cost:
                raise ValueError(
                    f"Insufficient cash: need {total_cost:.2f}, have {account.cash:.2f}"
                )

            # Check max position size
            max_qty = self._get_max_allowed_quantity(account, filled_price)
            effective_qty = min(quantity, max_qty)
            if effective_qty < quantity:
                # Adjust trade for capped quantity
                effective_commission = filled_price * effective_qty * self.default_commission_pct
                trade.quantity = effective_qty
                trade.filled_quantity = effective_qty
                trade.commission = effective_commission
                total_cost = filled_price * effective_qty + effective_commission

            account.cash -= total_cost
            account.total_commission += trade.commission
            account.total_trades += 1

            # Update or create position
            if instrument in account.positions:
                pos = account.positions[instrument]
                total_qty = pos.quantity + trade.filled_quantity
                total_cost_basis = (pos.avg_price * pos.quantity) + (
                    filled_price * trade.filled_quantity
                )
                pos.avg_price = total_cost_basis / total_qty if total_qty > 0 else 0.0
                pos.quantity = total_qty
                pos.total_commission += trade.commission
            else:
                account.positions[instrument] = PaperPosition(
                    symbol=instrument,
                    quantity=trade.filled_quantity,
                    avg_price=filled_price,
                    total_commission=trade.commission,
                    opened_at=datetime.now(UTC),
                )

        elif direction == TradeDirection.SELL:
            if instrument not in account.positions:
                raise ValueError(f"No position to sell: {instrument}")

            pos = account.positions[instrument]
            available_qty = pos.quantity

            if available_qty < quantity:
                raise ValueError(
                    f"Insufficient position: need {quantity}, have {available_qty}"
                )

            # Sell portion of position — update PnL
            price_diff = pos.avg_price - filled_price
            realized_pnl = price_diff * quantity

            account.cash += filled_price * quantity - commission
            account.total_commission += commission
            account.total_trades += 1

            # Update position
            pos.quantity -= quantity
            pos.realized_pnl += realized_pnl
            pos.total_commission += commission

            if pos.quantity <= 0:
                pos.closed_at = datetime.now(UTC)
                pos.quantity = 0.0
                del account.positions[instrument]

        return trade

    def close_position(
        self, account: PaperAccount, instrument: str
    ) -> Trade | None:
        """Close the entire position for the given instrument.

        Args:
            account: The paper account holding the position.
            instrument: Symbol of the position to close.

        Returns:
            The closing Trade, or None if no position exists.
        """
        if instrument not in account.positions:
            return None

        pos = account.positions[instrument]
        if pos.quantity <= 0:
            return None

        # Use a dummy market price for the close (same as avg_price for simulation)
        market_price = pos.avg_price
        quantity = pos.quantity

        trade_id = str(uuid.uuid4())
        slippage = self.default_slippage_pct

        # Sell slippage reduces proceeds
        filled_price = market_price * (1 - slippage)
        commission = filled_price * quantity * self.default_commission_pct

        realized_pnl = (market_price - pos.avg_price) * quantity

        trade = Trade(
            trade_id=trade_id,
            instrument=instrument,
            direction=TradeDirection.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=market_price,
            slippage=slippage,
            commission=commission,
            filled_price=filled_price,
            filled_quantity=quantity,
            status="filled",
        )

        account.cash += filled_price * quantity - commission
        account.total_commission += commission
        account.total_trades += 1

        pos.realized_pnl += realized_pnl
        pos.total_commission += commission
        pos.closed_at = datetime.now(UTC)
        pos.quantity = 0.0
        del account.positions[instrument]

        return trade

    def get_account_summary(self, account: PaperAccount) -> dict[str, Any]:
        """Return a summary of the account state.

        Args:
            account: The account to summarize.

        Returns:
            Dict with cash, equity, pnl breakdowns, trade counts, and positions.
        """
        position_list = []
        for sym, pos in account.positions.items():
            position_list.append(
                {
                    "symbol": sym,
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                    "total_commission": pos.total_commission,
                }
            )

        return {
            "account_id": account.account_id,
            "cash": account.cash,
            "equity": account.equity,
            "unrealized_pnl": account.unrealized_pnl,
            "realized_pnl": account.realized_pnl,
            "total_pnl": account.total_pnl,
            "total_trades": account.total_trades,
            "total_commission": account.total_commission,
            "num_positions": len(account.positions),
            "position_list": position_list,
        }
