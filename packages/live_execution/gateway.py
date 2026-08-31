"""Live execution gateway — CCXT-based order execution entry point.

The gateway is the single point of entry for submitting, cancelling,
querying, and tracking live orders.  It uses `CCXT <https://docs.ccxt.org>`_
as the unified exchange API layer.

Workflow
--------
1. ``submit_order()`` validates the order through :class:`OrderValidator`.
2. A fresh idempotency key is generated (or uses the caller-supplied key).
3. The order state machine is created in the ``NEW`` state.
4. The gateway checks the venue's rate limiter and waits if necessary.
5. The order is sent to the exchange via CCXT.
6. The state machine transitions through ``NEW -> PENDING -> FILLED/CANCELLED/REJECTED/EXPIRED``.
7. On any unexpected failure, the state machine transitions to ``ERROR``.

Supported order types (CCXT standard):
    - ``market`` — executed at the best available market price.
    - ``limit`` — executed at the specified price or better.
    - ``stop_limit`` — triggers at ``stop_price``, then becomes a limit order.
    - ``stop_market`` — triggers at ``stop_price``, then executes as a market order.

Usage
-----

.. code-block:: python

    gateway = LiveExecutionGateway(
        ccxt_config={
            "binance": {
                "apiKey": "YOUR_KEY",
                "secret": "YOUR_SECRET",
                "enableRateLimit": True,
            }
        },
        venues=["binance"],
    )

    result = await gateway.submit_order(
        venue="binance",
        symbol="BTC/USDT",
        side="buy",
        order_type="limit",
        amount=0.1,
        price=45000.0,
    )
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from packages.live_execution.idempotency import IdempotencyStore
from packages.live_execution.order_state_machine import (
    OrderState,
    OrderStateMachine,
    StateTransitionError,
)
from packages.live_execution.rate_limiter import RateLimiter
from packages.live_execution.validator import OrderValidator, ValidationError

logger = logging.getLogger(__name__)

# CCXT is an optional runtime dependency — import lazily.
_ccxt: Any = None


def _get_ccxt() -> Any:
    """Lazily import CCXT.  Raises ImportError if not installed."""
    global _ccxt
    if _ccxt is None:
        import ccxt

        _ccxt = ccxt
    return _ccxt


# ─── Data Types ─────────────────────────────────────────────────────────────


class OrderResult:
    """Result of an order operation.

    Attributes:
        order_id: Venue-assigned order identifier.
        idempotency_key: The idempotency key used for this submission.
        symbol: Trading pair.
        venue: Venue identifier.
        side: ``"buy"`` or ``"sell"``.
        order_type: Type of order.
        quantity: Ordered quantity.
        price: Limit price (``None`` for market orders).
        state: Current :class:`OrderState`.
        status: Human-readable status string.
        raw_response: Raw exchange response (for debugging).
        error: Error message if the operation failed.
        submitted_at: Timestamp of submission.
    """

    def __init__(
        self,
        idempotency_key: str,
        symbol: str,
        venue: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
    ) -> None:
        self.idempotency_key = idempotency_key
        self.symbol = symbol
        self.venue = venue
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.state = OrderState.NEW
        self.order_id: str = ""
        self.status: str = "submitted"
        self.raw_response: dict[str, Any] = {}
        self.error: str = ""
        self.submitted_at: datetime = datetime.now(UTC)
        self.filled_quantity: float = 0.0
        self.fill_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "idempotency_key": self.idempotency_key,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "venue": self.venue,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "state": self.state.name,
            "status": self.status,
            "error": self.error,
            "submitted_at": self.submitted_at.isoformat(),
            "filled_quantity": self.filled_quantity,
            "fill_price": self.fill_price,
        }


# ─── Gateway Exceptions ─────────────────────────────────────────────────────


class GatewayError(Exception):
    """Base exception for gateway errors."""


class GatewayValidationError(GatewayError):
    """Raised when order validation fails."""

    def __init__(self, errors: list[ValidationError]) -> None:
        self.errors = errors
        super().__init__(
            f"Order validation failed with {len(errors)} error(s): "
            + "; ".join(e.message for e in errors)
        )


class GatewayExecutionError(GatewayError):
    """Raised when order submission fails at the exchange level."""

    def __init__(self, message: str, raw: dict[str, Any] | None = None) -> None:
        self.raw = raw or {}
        super().__init__(message)


class GatewayIdempotencyError(GatewayError):
    """Raised when a duplicate idempotency key is detected."""

    def __init__(
        self, idempotency_key: str, cached_result: dict[str, Any]
    ) -> None:
        self.idempotency_key = idempotency_key
        self.cached_result = cached_result
        super().__init__(
            f"Duplicate idempotency key: {idempotency_key}"
        )


# ─── Gateway ────────────────────────────────────────────────────────────────


class LiveExecutionGateway:
    """CCXT-based live execution gateway.

    Acts as the entry point for all live order operations.  Manages order
    state machines, idempotency, rate limiting, and validation.

    Args:
        ccxt_config: Per-venue CCXT configuration.  Keys are venue names.
        venues: List of active venue identifiers.
        validator: Custom validator instance.  A default is created if ``None``.
        rate_limiter: Custom rate limiter.  A default is created if ``None``.
        idempotency_store: Custom idempotency store.  A default is created if ``None``.
    """

    def __init__(
        self,
        ccxt_config: dict[str, dict[str, Any]] | None = None,
        venues: list[str] | None = None,
        validator: OrderValidator | None = None,
        rate_limiter: RateLimiter | None = None,
        idempotency_store: IdempotencyStore | None = None,
    ) -> None:
        self._ccxt_config = ccxt_config or {}
        self._venues = venues or ["binance"]
        self._validators: dict[str, OrderValidator] = {}
        self._rate_limiter = rate_limiter or RateLimiter()
        self._idempotency_store = idempotency_store or IdempotencyStore()

        # Register venues in the rate limiter (sync, creates internal state)
        for venue in self._venues:
            venue_cfg = self._ccxt_config.get(venue, {})
            capacity = venue_cfg.get("rate_limit_capacity", 10)
            refill = venue_cfg.get("rate_limit_refill", 5)
            self._rate_limiter._register_sync(
                venue, capacity=capacity, refill=refill
            )

        # Register per-venue validators
        if validator is not None:
            for venue in self._venues:
                self._validators[venue] = validator

        # Order state machines keyed by order_id
        self._state_machines: dict[str, OrderStateMachine] = {}
        logger.info(
            "LiveExecutionGateway initialized for venues: %s", self._venues,
        )

    # ── order submission ─────────────────────────────────────────────────

    async def submit_order(
        self,
        venue: str,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        stop_price: float | None = None,
        idempotency_key: str | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        """Submit a new order to the specified venue.

        Full pipeline: validation -> idempotency check -> rate limiting ->
        CCXT submission -> state transition.

        Args:
            venue: Venue identifier (e.g. ``"binance"``).
            symbol: Trading pair (e.g. ``"BTC/USDT"``).
            side: ``"buy"`` or ``"sell"``.
            order_type: One of ``"market"``, ``"limit"``, ``"stop_limit"``,
                ``"stop_market"``.
            amount: Order quantity.
            price: Limit price (required for limit / stop-limit orders).
            stop_price: Stop price (required for stop-limit / stop-market).
            idempotency_key: Custom key.  Generated if ``None``.
            client_order_id: Client-assigned order ID for the exchange.

        Returns:
            An :class:`OrderResult` with the submission outcome.

        Raises:
            GatewayValidationError: If validation fails.
            GatewayIdempotencyError: If the key was already used.
            GatewayExecutionError: If CCXT submission fails.
        """
        # 1. Generate or validate idempotency key
        if idempotency_key is None:
            idempotency_key = str(uuid.uuid4())

        # 2. Check idempotency store first
        existing = await self._idempotency_store.get(idempotency_key, venue)
        if existing is not None:
            result = OrderResult(
                idempotency_key=idempotency_key,
                symbol=symbol,
                venue=venue,
                side=side,
                order_type=order_type,
                quantity=amount,
                price=price,
            )
            result.state = OrderState.PENDING
            result.status = "duplicate_cached"
            result.raw_response = existing
            result.error = ""
            return result

        # 3. Validate the order
        validator = self._validators.get(venue)
        if validator is None:
            validator = OrderValidator()
            self._validators[venue] = validator

        validation_errors = await validator.validate(
            symbol=symbol,
            venue=venue,
            side=side,
            quantity=amount,
            price=price,
            order_type=order_type,
            stop_price=stop_price,
        )
        if validation_errors:
            raise GatewayValidationError(validation_errors)

        # 4. Create order result and state machine
        result = OrderResult(
            idempotency_key=idempotency_key,
            symbol=symbol,
            venue=venue,
            side=side,
            order_type=order_type,
            quantity=amount,
            price=price,
        )
        result.status = "submitted"

        # 5. Create state machine
        order_id = client_order_id or f"live_{uuid.uuid4().hex[:12]}"
        state_machine = OrderStateMachine(
            order_id=order_id,
            symbol=symbol,
            venue=venue,
            side=side,
            quantity=amount,
            price=price,
        )
        self._state_machines[order_id] = state_machine

        # 6. Transition to PENDING
        try:
            state_machine.transition_to(
                OrderState.PENDING, event="submitted"
            )
        except StateTransitionError as exc:
            state_machine.transition_from_error(
                event="submit_transition_error"
            )
            raise GatewayExecutionError(
                f"Failed to transition to PENDING: {exc}"
            ) from exc

        # 7. Acquire rate limit tokens
        try:
            await self._rate_limiter.acquire(venue, tokens=1)
        except Exception as exc:
            result.error = f"Rate limit acquire failed: {exc}"
            state_machine.transition_from_error(event="rate_limit_error")
            raise GatewayExecutionError(
                f"Rate limit error: {exc}"
            ) from exc

        # 8. Submit to exchange via CCXT
        try:
            exchange = self._create_exchange(venue)
            ccxt_order = await self._execute_ccxt_order(
                exchange=exchange,
                symbol=symbol,
                side=side,
                order_type=order_type,
                amount=amount,
                price=price,
                stop_price=stop_price,
                client_order_id=client_order_id,
            )

            # Extract venue order ID
            venue_order_id = ccxt_order.get("id", "")
            if venue_order_id:
                result.order_id = str(venue_order_id)

            result.raw_response = ccxt_order

            # Determine fill status from CCXT response
            ccxt_status = (ccxt_order.get("status") or "").lower()
            filled_qty = ccxt_order.get("filled", 0) or 0
            avg_price = ccxt_order.get("average")

            if ccxt_status in ("closed", "filled"):
                state_machine.transition_to(
                    OrderState.FILLED,
                    event="filled",
                    filled_quantity=filled_qty,
                )
                result.state = OrderState.FILLED
                result.status = "filled"
                result.filled_quantity = filled_qty
                if avg_price is not None:
                    result.fill_price = float(avg_price)
            elif ccxt_status in ("open", "pending"):
                if filled_qty > 0:
                    state_machine.transition_to(
                        OrderState.PARTIALLY_FILLED,
                        event="partial_fill",
                        filled_quantity=filled_qty,
                    )
                    result.state = OrderState.PARTIALLY_FILLED
                    result.status = "partial_fill"
                    result.filled_quantity = filled_qty
                else:
                    result.state = OrderState.PENDING
                    result.status = "pending"
            elif ccxt_status in ("canceled", "cancelled"):
                state_machine.transition_to(
                    OrderState.CANCELLED,
                    event="cancelled",
                )
                result.state = OrderState.CANCELLED
                result.status = "cancelled"
            elif ccxt_status == "expired":
                state_machine.transition_to(
                    OrderState.EXPIRED,
                    event="expired",
                )
                result.state = OrderState.EXPIRED
                result.status = "expired"
            else:
                result.state = OrderState.PENDING
                result.status = "unknown"

            # Record in idempotency store
            await self._idempotency_store.record(
                idempotency_key=idempotency_key,
                venue=venue,
                result=result.to_dict(),
            )

            # Record success in rate limiter
            await self._rate_limiter.record_success(venue)

        except Exception as exc:
            # Handle CCXT errors
            status_code = getattr(exc, "status_code", None)
            message = str(exc)

            # Record rate-limit error if applicable
            if status_code == 429:
                await self._rate_limiter.record_rate_limit_error(venue)

            logger.error(
                "CCXT submission failed for %s/%s %s: %s",
                venue, symbol, side, exc,
            )

            # Try to determine if it was a rejection or error
            if (
                "rejected" in message.lower()
                or "invalid" in message.lower()
            ):
                state_machine.transition_to(
                    OrderState.REJECTED,
                    event="rejected",
                )
                result.state = OrderState.REJECTED
                result.status = "rejected"
            else:
                state_machine.transition_from_error(event="ccxt_error")
                result.state = OrderState.ERROR
                result.status = "error"

            result.error = message
            result.raw_response = self._error_response(exc)

            # Record error in idempotency store (so duplicate submits
            # return error)
            await self._idempotency_store.record(
                idempotency_key=idempotency_key,
                venue=venue,
                result=result.to_dict(),
            )

        return result

    # ── order management ─────────────────────────────────────────────────

    async def cancel_order(
        self,
        venue: str,
        order_id: str,
    ) -> OrderResult:
        """Cancel an existing order.

        Args:
            venue: Venue identifier.
            order_id: Venue-assigned order ID.

        Returns:
            :class:`OrderResult` with cancellation outcome.
        """
        result = OrderResult(
            idempotency_key="",
            symbol="",
            venue=venue,
            side="",
            order_type="",
            quantity=0,
            price=None,
        )
        result.order_id = order_id
        result.status = "cancelling"

        try:
            exchange = self._create_exchange(venue)
            await self._rate_limiter.acquire(venue, tokens=1)

            ccxt_result = await exchange.cancel_order(
                order_id,
                symbol=result.symbol or "UNKNOWN",
            )

            ccxt_status = (ccxt_result.get("status") or "").lower()
            if ccxt_status in ("canceled", "cancelled", "closed"):
                result.state = OrderState.CANCELLED
                result.status = "cancelled"
            else:
                result.state = OrderState.PENDING
                result.status = "cancel_pending"

            result.raw_response = ccxt_result

            # Update state machine if it exists
            sm = self._state_machines.get(order_id)
            if sm is not None:
                sm.transition_to(
                    OrderState.CANCELLED,
                    event="cancel_order",
                )

            await self._rate_limiter.record_success(venue)

        except Exception as exc:
            result.error = str(exc)
            result.state = OrderState.ERROR
            result.status = "cancel_error"
            logger.error(
                "Cancel order failed for %s: %s", order_id, exc
            )

        return result

    async def get_order_status(
        self,
        venue: str,
        order_id: str,
    ) -> dict[str, Any]:
        """Query the current status of an order from the exchange.

        Args:
            venue: Venue identifier.
            order_id: Venue-assigned order ID.

        Returns:
            Raw exchange response dict.
        """
        exchange = self._create_exchange(venue)
        await self._rate_limiter.acquire(venue, tokens=1)
        try:
            order_data = await exchange.fetch_order(
                order_id, symbol="UNKNOWN"
            )
            await self._rate_limiter.record_success(venue)
            return order_data
        except Exception as exc:
            logger.error("Fetch order status failed: %s", exc)
            raise

    async def get_open_orders(
        self,
        venue: str,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch open orders for a venue.

        Args:
            venue: Venue identifier.
            symbol: Filter by symbol (optional).

        Returns:
            List of open order dicts.
        """
        exchange = self._create_exchange(venue)
        await self._rate_limiter.acquire(venue, tokens=1)
        try:
            orders = await exchange.fetch_open_orders(
                symbol or ""
            )
            await self._rate_limiter.record_success(venue)
            return orders
        except Exception as exc:
            logger.error("Fetch open orders failed: %s", exc)
            raise

    # ── order tracking ─────────────────────────────────────────────────

    def get_state_machine(
        self, order_id: str
    ) -> OrderStateMachine | None:
        """Retrieve the state machine for an order, or ``None``."""
        return self._state_machines.get(order_id)

    def get_order_state(self, order_id: str) -> OrderState | None:
        """Get the current state of an order, or ``None`` if not found."""
        sm = self._state_machines.get(order_id)
        if sm is not None:
            return sm.state
        return None

    def get_all_order_states(self) -> dict[str, str]:
        """Return a mapping of order_id -> state_name for all tracked orders."""
        return {
            oid: sm.state.name
            for oid, sm in self._state_machines.items()
        }

    # ── internal ─────────────────────────────────────────────────────────

    def _create_exchange(self, venue: str) -> Any:
        """Create a CCXT exchange instance for a venue.

        Args:
            venue: Venue identifier (maps to CCXT exchange ID).

        Returns:
            Initialized CCXT exchange instance.
        """
        ccxt_module = _get_ccxt()
        venue_lower = venue.lower()

        # Map known venue names to CCXT exchange IDs
        venue_map: dict[str, str] = {
            "binance": "binance",
            "binanceus": "binanceus",
            "bybit": "bybit",
            "okx": "okx",
            "coinbase": "coinbase",
            "dummy": "dummy",
        }
        exchange_id = venue_map.get(venue_lower, venue_lower)

        exchange_class = getattr(ccxt_module, exchange_id, None)
        if exchange_class is None:
            raise GatewayExecutionError(
                f"Unknown exchange: {exchange_id}. "
                f"Available: {list(ccxt_module.exchanges)}"
            )

        config = self._ccxt_config.get(venue, {})
        exchange = exchange_class(config)

        # Enable rate limiting
        if "enableRateLimit" not in config:
            exchange.enableRateLimit = True

        return exchange

    async def _execute_ccxt_order(
        self,
        exchange: Any,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None,
        stop_price: float | None,
        client_order_id: str | None,
    ) -> dict[str, Any]:
        """Execute an order via the CCXT exchange API.

        Args:
            exchange: CCXT exchange instance.
            symbol: Trading pair.
            side: ``"buy"`` or ``"sell"``.
            order_type: CCXT order type.
            amount: Order quantity.
            price: Limit price.
            stop_price: Stop price.
            client_order_id: Client-side order ID.

        Returns:
            CCXT order response dict.
        """
        params: dict[str, Any] = {}

        if client_order_id:
            params["clientOrderId"] = client_order_id

        # Stop-limit and stop-market orders need stopPrice
        if (
            order_type in ("stop_limit", "stop_market")
            and stop_price is not None
        ):
            params["stopPrice"] = str(stop_price)

        if order_type == "market":
            ccxt_order = await exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=str(amount),
                params=params,
            )
        elif order_type == "limit":
            if price is None:
                raise GatewayExecutionError(
                    "Limit order requires a price"
                )
            ccxt_order = await exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=str(amount),
                price=str(price),
                params=params,
            )
        else:
            # Stop-limit / stop-market
            if price is None:
                raise GatewayExecutionError(
                    f"{order_type} order requires a price"
                )
            ccxt_order = await exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=str(amount),
                price=str(price),
                params=params,
            )

        return ccxt_order

    @staticmethod
    def _error_response(exc: Exception) -> dict[str, Any]:
        """Extract useful info from an exception for error responses."""
        result: dict[str, Any] = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            result["http_status"] = status_code

        headers = getattr(exc, "headers", None)
        if headers is not None:
            result["headers"] = (
                dict(headers) if isinstance(headers, dict) else {}
            )

        body = getattr(exc, "body", None)
        if body is not None:
            result["raw_body"] = body

        return result
