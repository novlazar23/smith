"""Paper trading executor — handles order submission, execution, and position management."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from .base import OrderType, PaperAccount, PaperPosition, Trade, TradeDirection
from .fill_model import FillModel, FillStatus
from .latency_simulator import LatencySimulator


class PaperExecutor:
    """Simulates trade execution with slippage, commissions, and optional
    advanced fill / latency models."""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        default_slippage_pct: float = 0.001,
        default_commission_pct: float = 0.001,
        max_position_size_pct: float = 0.10,
        fill_model: FillModel | None = None,
        latency_simulator: LatencySimulator | None = None,
    ) -> None:
        """Initialize the executor with default parameters.

        Args:
            initial_cash: Starting cash for new accounts.
            default_slippage_pct: Fraction of price added for market impact (0.1% = 10bps).
            default_commission_pct: Fraction of the notional value charged per trade.
            max_position_size_pct: Maximum position as fraction of equity (10%).
            fill_model: Optional stochastic fill model.  When *None* all orders
                        execute immediately at full quantity.
            latency_simulator: Optional latency simulator.  When *None* no
                               artificial delay is applied.
        """
        self.initial_cash = initial_cash
        self.default_slippage_pct = default_slippage_pct
        self.default_commission_pct = default_commission_pct
        self.max_position_size_pct = max_position_size_pct
        self.fill_model = fill_model
        self.latency_simulator = latency_simulator
        self._accounts: dict[str, PaperAccount] = {}
        # Pending orders keyed by order_id
        self._pending_orders: dict[str, _PendingOrder] = {}

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

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _get_max_allowed_quantity(self, account: PaperAccount, price: float) -> float:
        """Calculate the maximum allowed quantity based on position size limit."""
        max_notional = account.equity * self.max_position_size_pct
        if price <= 0:
            return 0.0
        return max_notional / price

    def _execute_buy(
        self,
        account: PaperAccount,
        instrument: str,
        quantity: float,
        filled_price: float,
        commission: float,
        slippage: float,
        order_type: OrderType,
        status: str,
    ) -> Trade:
        """Internal helper that executes a BUY trade and updates account state.

        Args:
            account: The account to debit.
            instrument: Symbol to buy.
            quantity: Filled quantity.
            filled_price: Actual fill price.
            commission: Commission amount.
            slippage: Slippage fraction.
            order_type: The original order type.
            status: Fill status string.

        Returns:
            The filled Trade object.
        """
        trade_id = str(uuid.uuid4())
        total_cost = filled_price * quantity + commission

        if account.cash < total_cost:
            raise ValueError(
                f"Insufficient cash: need {total_cost:.2f}, have {account.cash:.2f}"
            )

        # Check max position size
        max_qty = self._get_max_allowed_quantity(account, filled_price)
        effective_qty = min(quantity, max_qty)
        if effective_qty < quantity:
            effective_commission = filled_price * effective_qty * self.default_commission_pct
            total_cost = filled_price * effective_qty + effective_commission
        else:
            effective_commission = commission
            effective_qty = quantity

        trade = Trade(
            trade_id=trade_id,
            instrument=instrument,
            direction=TradeDirection.BUY,
            order_type=order_type,
            quantity=quantity,
            price=filled_price,
            slippage=slippage,
            commission=effective_commission,
            filled_price=filled_price,
            filled_quantity=effective_qty,
            status=status,
        )

        account.cash -= total_cost
        account.total_commission += effective_commission
        account.total_trades += 1

        # Update or create position
        if instrument in account.positions:
            pos = account.positions[instrument]
            total_qty = pos.quantity + effective_qty
            total_cost_basis = (pos.avg_price * pos.quantity) + (
                filled_price * effective_qty
            )
            pos.avg_price = total_cost_basis / total_qty if total_qty > 0 else 0.0
            pos.quantity = total_qty
            pos.total_commission += effective_commission
        else:
            account.positions[instrument] = PaperPosition(
                symbol=instrument,
                quantity=effective_qty,
                avg_price=filled_price,
                total_commission=effective_commission,
                opened_at=datetime.now(UTC),
            )

        return trade

    def _execute_sell(
        self,
        account: PaperAccount,
        instrument: str,
        quantity: float,
        filled_price: float,
        commission: float,
        slippage: float,
        order_type: OrderType,
        status: str,
    ) -> Trade:
        """Internal helper that executes a SELL trade and updates account state.

        Args:
            account: The account to credit.
            instrument: Symbol to sell.
            quantity: Filled quantity.
            filled_price: Actual fill price.
            commission: Commission amount.
            slippage: Slippage fraction.
            order_type: The original order type.
            status: Fill status string.

        Returns:
            The filled Trade object.

        Raises:
            ValueError: If there is no position or insufficient quantity.
        """
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

        trade_id = str(uuid.uuid4())
        trade = Trade(
            trade_id=trade_id,
            instrument=instrument,
            direction=TradeDirection.SELL,
            order_type=order_type,
            quantity=quantity,
            price=filled_price,
            slippage=slippage,
            commission=commission,
            filled_price=filled_price,
            filled_quantity=quantity,
            status=status,
        )

        return trade

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        account: PaperAccount,
        instrument: str,
        direction: TradeDirection,
        quantity: float,
        price: float | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trigger_price: float | None = None,
    ) -> Trade:
        """Submit a trade order for execution.

        For MARKET orders the *price* argument is the current market price.
        Slippage is applied as a fraction of the market price.
        Commission is calculated on the filled notional value.

        When an advanced ``fill_model`` is configured, order filling is
        stochastic: partial fills, rejections, and pending states are
        possible.  Latency is also simulated when a
        ``latency_simulator`` is present.

        Args:
            account: The paper account to execute against.
            instrument: Symbol of the instrument to trade.
            direction: BUY or SELL.
            quantity: Number of units to trade.
            price: Current market price (required for MARKET orders).
            order_type: Type of order (MARKET, LIMIT, STOP).
            limit_price: Limit price for LIMIT orders.
            stop_price: Stop price for STOP orders.
            trigger_price: Current trigger price used to evaluate STOP orders.

        Returns:
            The filled Trade object (status may be "partial" or "pending").

        Raises:
            ValueError: If there is insufficient cash (BUY) or position (SELL),
                        or required prices are missing for LIMIT/STOP orders.
        """
        if price is None or price <= 0:
            raise ValueError("Market price must be provided and positive")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        slippage = self.default_slippage_pct
        fill_status = "filled"

        # --- Stochastic fill evaluation (if model configured) ---
        if self.fill_model is not None:
            result = self.fill_model.calculate_partial_fill(
                order_quantity=quantity,
                order_type=order_type,
                market_price=price,
                limit_price=limit_price,
                stop_price=stop_price,
                trigger_price=trigger_price,
            )
            filled_quantity, filled_price, raw_status = result

            if raw_status == FillStatus.REJECTED:
                # Build a rejected trade
                trade_id = str(uuid.uuid4())
                return Trade(
                    trade_id=trade_id,
                    instrument=instrument,
                    direction=direction,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    slippage=slippage,
                    commission=0.0,
                    filled_price=price,
                    filled_quantity=0.0,
                    status="rejected",
                )

            if raw_status == FillStatus.PENDING:
                # Store as pending order for later resolution
                order_id = str(uuid.uuid4())
                self._pending_orders[order_id] = _PendingOrder(
                    order_id=order_id,
                    account_id=account.account_id,
                    instrument=instrument,
                    direction=direction,
                    quantity=quantity,
                    price=price,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    trigger_price=trigger_price,
                    created_at=datetime.now(UTC),
                )
                # Return a "pending" trade as a placeholder
                return Trade(
                    trade_id=order_id,
                    instrument=instrument,
                    direction=direction,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    slippage=slippage,
                    commission=0.0,
                    filled_price=0.0,
                    filled_quantity=0.0,
                    status="pending",
                )

            # Partial or full fill — use returned quantity
            filled_quantity = max(filled_quantity, quantity * self.fill_model.partial_fill_pct)
            fill_status = "partial" if raw_status == FillStatus.PARTIAL else "filled"

            # --- Queue-position slippage adjustment ---
            size_ratio = (filled_quantity * filled_price) / self.fill_model.typical_liquidity
            queue_slippage = 1.0 + (self.fill_model.queue_position_factor * size_ratio * 0.01)
            filled_price = filled_price * queue_slippage

        else:
            # --- Legacy path (backward-compatible) ---
            filled_price = price * (1 + slippage) if direction == TradeDirection.BUY else price * (1 - slippage)
            fill_status = "filled"
            filled_quantity = quantity

        # --- Latency simulation ---
        if self.latency_simulator is not None:
            self.latency_simulator.simulate_latency_for_order(order_type)

        # --- Build commission ---
        commission = filled_price * filled_quantity * self.default_commission_pct

        # --- Execute ---
        if direction == TradeDirection.BUY:
            return self._execute_buy(
                account=account,
                instrument=instrument,
                quantity=filled_quantity,
                filled_price=filled_price,
                commission=commission,
                slippage=slippage,
                order_type=order_type,
                status=fill_status,
            )

        return self._execute_sell(
            account=account,
            instrument=instrument,
            quantity=filled_quantity,
            filled_price=filled_price,
            commission=commission,
            slippage=slippage,
            order_type=order_type,
            status=fill_status,
        )

    # ------------------------------------------------------------------
    # Pending-order resolution
    # ------------------------------------------------------------------

    def advance_time(
        self,
        account: PaperAccount,
        instrument: str,
        market_price: float,
    ) -> list[Trade]:
        """Process all pending orders whose conditions are now met.

        This is the mechanism by which LIMIT / STOP orders transition from
        ``"pending"`` to ``"filled"`` / ``"partial"`` once the market has
        moved to satisfy their price constraints.

        Args:
            account: The account holding the pending orders.
            instrument: Symbol whose pending orders should be evaluated.
            market_price: Current market price after time has advanced.

        Returns:
            List of newly-executed Trade objects.
        """
        executed: list[Trade] = []
        orders_to_remove: list[str] = []

        for order_id, order in self._pending_orders.items():
            if order.account_id != account.account_id:
                continue
            if order.instrument != instrument:
                continue

            # Re-evaluate using the fill model
            if self.fill_model is None:
                continue

            result = self.fill_model.calculate_partial_fill(
                order_quantity=order.quantity,
                order_type=order.order_type,
                market_price=market_price,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                trigger_price=market_price,  # use current market as trigger proxy
            )
            _, _, raw_status = result

            if raw_status == FillStatus.PENDING:
                continue  # Still not ready

            if raw_status == FillStatus.REJECTED:
                orders_to_remove.append(order_id)
                trade_id = str(uuid.uuid4())
                executed.append(Trade(
                    trade_id=trade_id,
                    instrument=instrument,
                    direction=order.direction,
                    order_type=order.order_type,
                    quantity=order.quantity,
                    price=market_price,
                    slippage=self.default_slippage_pct,
                    commission=0.0,
                    filled_price=market_price,
                    filled_quantity=0.0,
                    status="rejected",
                ))
                continue

            # Execute the fill — call submit_order-like logic directly
            filled_quantity = max(
                result[0], order.quantity * self.fill_model.partial_fill_pct
            )
            filled_price = result[1]
            size_ratio = (filled_quantity * filled_price) / self.fill_model.typical_liquidity
            queue_slippage = 1.0 + (self.fill_model.queue_position_factor * size_ratio * 0.01)
            filled_price = filled_price * queue_slippage

            commission = filled_price * filled_quantity * self.default_commission_pct

            if order.direction == TradeDirection.BUY:
                trade = self._execute_buy(
                    account=account,
                    instrument=instrument,
                    quantity=filled_quantity,
                    filled_price=filled_price,
                    commission=commission,
                    slippage=self.default_slippage_pct,
                    order_type=order.order_type,
                    status="filled",
                )
            else:
                trade = self._execute_sell(
                    account=account,
                    instrument=instrument,
                    quantity=filled_quantity,
                    filled_price=filled_price,
                    commission=commission,
                    slippage=self.default_slippage_pct,
                    order_type=order.order_type,
                    status="filled",
                )

            executed.append(trade)
            orders_to_remove.append(order_id)

        for oid in orders_to_remove:
            self._pending_orders.pop(oid, None)

        return executed

    # ------------------------------------------------------------------
    # Position closing
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Account summary
    # ------------------------------------------------------------------

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
            "num_pending_orders": len(self._pending_orders),
            "position_list": position_list,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _PendingOrder:
    """Internal container for orders that are pending fill conditions."""

    def __init__(
        self,
        order_id: str,
        account_id: str,
        instrument: str,
        direction: TradeDirection,
        quantity: float,
        price: float,
        order_type: OrderType,
        limit_price: float | None,
        stop_price: float | None,
        trigger_price: float | None,
        created_at: datetime,
    ) -> None:
        self.order_id = order_id
        self.account_id = account_id
        self.instrument = instrument
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.trigger_price = trigger_price
        self.created_at = created_at
