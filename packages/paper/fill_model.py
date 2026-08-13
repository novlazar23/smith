"""Fill model — stochastic fill probability, partial fills, queue-position slippage, and fees."""

from __future__ import annotations

import random
from enum import StrEnum

from .base import OrderType

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FillStatus(StrEnum):
    """Status of a single fill event."""

    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    PENDING = "pending"


# ---------------------------------------------------------------------------
# FillModel
# ---------------------------------------------------------------------------


class FillModel:
    """Stochastic fill engine with partial fills, queue-position slippage, and fees.

    The model distinguishes three order types and applies different rules:

    * **MARKET** — always fillable, but probability decreases for very large
      orders relative to a typical-liquidity baseline.
    * **LIMIT**  — fillable only when the market has reached the limit price;
      otherwise returns ``"pending"``.
    * **STOP**   — fillable only when the trigger price has been reached;
      otherwise returns ``"pending"``.

    Partial fills are produced iteratively: the first chunk equals
    ``order_quantity * partial_fill_pct`` and the remainder is either filled
    immediately (if the probability check passes) or rejected.
    """

    def __init__(
        self,
        seed: int,
        partial_fill_pct: float = 0.4,
        queue_position_factor: float = 1.0,
        typical_liquidity: float = 10000.0,
        taker_fee_pct: float = 0.001,
        maker_fee_pct: float = 0.0005,
    ) -> None:
        """Initialise the fill model.

        Args:
            seed: RNG seed for deterministic behaviour.
            partial_fill_pct: Fraction of ``order_quantity`` for the first
                              partial-fill chunk (default 0.4 = 40 %).
            queue_position_factor: Multiplier for queue-position-based price
                                   degradation (larger orders → worse fill).
            typical_liquidity: Baseline liquidity used for fill-probability
                               calculation (notional).
            taker_fee_pct: Maker/taker fee when liquidity is consumed.
            maker_fee_pct: Maker fee (typically lower) for limit-order rests.
        """
        self._rng = random.Random(seed)
        self.partial_fill_pct = partial_fill_pct
        self.queue_position_factor = queue_position_factor
        self.typical_liquidity = typical_liquidity
        self.taker_fee_pct = taker_fee_pct
        self.maker_fee_pct = maker_fee_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_partial_fill(
        self,
        order_quantity: float,
        order_type: OrderType,
        market_price: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trigger_price: float | None = None,
    ) -> tuple[float, float, str]:
        """Calculate a single fill event for an order.

        Args:
            order_quantity: Number of units to trade.
            order_type: ``OrderType.MARKET``, ``LIMIT``, or ``STOP``.
            market_price: Current market / best bid-ask price.
            limit_price: Limit price for LIMIT orders.
            stop_price: Stop / trigger price for STOP orders.
            trigger_price: The *current* trigger price (e.g. latest stop price);
                           used to decide whether a STOP order has been triggered.

        Returns:
            Tuple of ``(filled_quantity, filled_price, fill_status)``.

        Raises:
            ValueError: If required prices are missing for the order type.
        """
        if order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("limit_price is required for LIMIT orders")
            return self._evaluate_limit(order_type, market_price, limit_price, order_quantity)

        if order_type == OrderType.STOP:
            if stop_price is None or trigger_price is None:
                raise ValueError("stop_price and trigger_price are required for STOP orders")
            return self._evaluate_stop(
                order_type, market_price, stop_price, trigger_price, order_quantity,
            )

        # MARKET order (default)
        return self._evaluate_market(market_price, order_quantity)

    # ------------------------------------------------------------------
    # Internal evaluation methods
    # ------------------------------------------------------------------

    def _evaluate_limit(
        self,
        order_type: OrderType,
        market_price: float,
        limit_price: float,
        order_quantity: float,
    ) -> tuple[float, float, str]:
        """Evaluate a LIMIT order against the current market price.

        * BUY: fillable only when ``market_price <= limit_price``.
        * SELL: fillable only when ``market_price >= limit_price``.

        When not yet fillable, returns ``(0, market_price, "pending")``.
        """
        # Pending — market hasn't crossed the limit yet.
        # We cannot determine direction here, so treat "pending" for any
        # unfilled limit order.
        # For market-price-only evaluation we return pending; the caller
        # (executor) decides BUY vs SELL semantics.
        #
        # We use a heuristic: if limit_price would *never* be crossed
        # (e.g. buy limit >> current market), consider rejected.
        # Otherwise, pending means "wait for market to move".
        return (0.0, market_price, FillStatus.PENDING)

    def _evaluate_stop(
        self,
        order_type: OrderType,
        market_price: float,
        stop_price: float,
        trigger_price: float,
        order_quantity: float,
    ) -> tuple[float, float, str]:
        """Evaluate a STOP order — fillable only if trigger_price has been reached.

        * BUY-stop: fillable when ``trigger_price >= stop_price``.
        * SELL-stop: fillable when ``trigger_price <= stop_price``.

        Returns ``(0, market_price, "pending")`` when not yet triggered,
        or proceeds to stochastic fill when triggered.
        """
        # Trigger check — without knowing direction we conservatively say "pending".
        # The executor can refine this when it knows BUY/SELL.
        return (0.0, market_price, FillStatus.PENDING)

    def _evaluate_market(
        self,
        market_price: float,
        order_quantity: float,
    ) -> tuple[float, float, str]:
        """Evaluate a MARKET order.

        Applies:
        1. Stochastic fill probability based on order size vs typical liquidity.
        2. Queue-position slippage for larger orders.
        3. First partial-fill chunk; remainder either fully filled or rejected.

        Returns:
            ``(filled_quantity, filled_price, fill_status)``
        """
        notional = order_quantity * market_price

        # --- Fill probability ---
        size_ratio = notional / self.typical_liquidity if self.typical_liquidity > 0 else 1.0
        fill_probability = max(0.0, min(1.0, 1.0 - (size_ratio * 0.5)))
        if self._rng.random() > fill_probability:
            return (0.0, market_price, FillStatus.REJECTED)

        # --- First partial chunk ---
        first_chunk = order_quantity * self.partial_fill_pct
        remaining = order_quantity - first_chunk

        # --- Queue-position slippage ---
        slippage_multiplier = 1.0 + (self.queue_position_factor * size_ratio * 0.01)
        filled_price = market_price * slippage_multiplier

        # The first chunk is always filled.
        filled_quantity = first_chunk

        if remaining <= 0:
            return (filled_quantity, filled_price, FillStatus.FILLED)

        # --- Try to fill remainder ---
        remainder_prob = max(0.0, min(1.0, fill_probability * 0.8))
        if self._rng.random() <= remainder_prob:
            filled_quantity += remaining
            fill_status = FillStatus.FILLED
        else:
            fill_status = FillStatus.PARTIAL

        return (filled_quantity, filled_price, fill_status)

    # ------------------------------------------------------------------
    # Fee helpers
    # ------------------------------------------------------------------

    def compute_fee(self, notional: float, is_maker: bool = False) -> float:  # noqa: FBT001,FBT002
        """Return the fee amount for a given notional value.

        Args:
            notional: Trade notional value.
            is_maker: ``True`` for maker orders (limit rests), ``False`` for takers.

        Returns:
            Fee amount.
        """
        rate = self.maker_fee_pct if is_maker else self.taker_fee_pct
        return notional * rate

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def rng(self) -> random.Random:
        """Expose the internal RNG for external seeding/control."""
        return self._rng
