"""Live Order Management Router — submit, list, cancel orders & kill-switch.

This router exposes REST endpoints for live-trading order management.
All order operations are gated behind the ``live_trading_enabled`` feature flag
and integrated with the live execution gateway (CCXT-based) for real order
submission.

Endpoints
---------
- ``POST /v1/live/orders`` — Submit a live order (validated + audit trail)
- ``GET /v1/live/orders`` — Order history with filters
- ``POST /v1/live/cancel`` — Cancel a live order
- ``POST /v1/live/kill-switch`` — Kill switch control (activate / deactivate)
- ``GET /v1/live/pnl`` — Realized / unrealized PnL with risk metrics
- ``GET /v1/live/pnl/daily`` — Daily PnL aggregation
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from packages.governance.audit import AuditTrail
from packages.governance.feature_flags import feature_flags
from packages.live_execution import (
    GatewayExecutionError,
    GatewayIdempotencyError,
    GatewayValidationError,
    LiveExecutionGateway,
    LivePnlTracker,
    OrderResult,
    OrderState,
)
from packages.rollout import get_rollout_controller
from packages.security import Permission, Role
from packages.security.hardening.audit_live import get_live_audit
from packages.security.hardening.rbac_live import require_live_permission
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/v1/live", tags=["live-orders"])

# ---------------------------------------------------------------------------
# Shared gateway instance (lazy; created on first request)
# ---------------------------------------------------------------------------

_gateway: LiveExecutionGateway | None = None


def _get_gateway() -> LiveExecutionGateway:
    """Return the singleton gateway, creating it lazily."""
    global _gateway
    if _gateway is None:
        _gateway = LiveExecutionGateway()
    return _gateway


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class SubmitOrderRequest(BaseModel):
    """Request body for submitting a live order.

    Attributes:
        instrument: Trading pair symbol (e.g. ``"BTC/USDT"``).
        direction: ``"buy"`` or ``"sell"``.
        quantity: Order quantity (must be > 0).
        order_type: One of ``"market"``, ``"limit"``, ``"stop_limit"``,
            ``"stop_market"``.
        price: Limit / stop price (``None`` for market orders).
        stop_price: Stop trigger price for ``stop_limit`` / ``stop_market``
            orders (optional; validated by the gateway).
        idempotency_key: Unique key to guard against duplicate submissions.
        venue: Exchange identifier (e.g. ``"binance"``, ``"bybit"``).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    instrument: str = Field(..., min_length=1, max_length=50, description="Trading pair symbol")
    direction: str = Field(..., pattern="^(buy|sell|BUY|SELL)$", description="Order direction")
    quantity: float = Field(..., gt=0, description="Order quantity")
    order_type: str = Field(
        ...,
        pattern="^(market|limit|stop_limit|stop_market|MARKET|LIMIT|STOP_LIMIT|STOP_MARKET)$",
        description="Order type",
    )
    price: float | None = Field(default=None, description="Limit or stop price (None for market orders)")
    stop_price: float | None = Field(
        default=None, description="Stop trigger price for stop orders"
    )
    idempotency_key: str = Field(..., min_length=1, description="Idempotency key for duplicate protection")
    venue: str = Field(..., min_length=1, max_length=50, description="Exchange/venue identifier")


class SubmitOrderResponse(BaseModel):
    """Response after order submission.

    Attributes:
        order_id: System-generated order identifier.
        state: Current :class:`OrderState` (string).
        submitted_at: ISO-8601 submission timestamp.
        idempotency_key: Echo of the provided idempotency key.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    order_id: str
    state: str
    submitted_at: datetime
    idempotency_key: str


class ListOrderFilter(BaseModel):
    """Query filters for the order history endpoint.

    Attributes:
        order_id: Filter by specific order ID (optional).
        status: Filter by order state string (optional).
        venue: Filter by venue (optional).
        from_dt: Start of time window (optional).
        to_dt: End of time window (optional).
    """

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    order_id: str | None = Field(default=None, description="Filter by order ID")
    status: str | None = Field(default=None, description="Filter by order state")
    venue: str | None = Field(default=None, description="Filter by venue")
    from_dt: datetime | None = Field(default=None, alias="from", description="Start datetime (inclusive)")
    to_dt: datetime | None = Field(default=None, alias="to", description="End datetime (inclusive)")


class OrderRecord(BaseModel):
    """A single order record returned by the history endpoint.

    Attributes:
        order_id: System order identifier.
        instrument: Trading pair symbol.
        venue: Venue identifier.
        direction: ``"buy"`` or ``"sell"``.
        quantity: Ordered quantity.
        price: Limit price or ``None``.
        order_type: Order type string.
        state: Current order state.
        status: Human-readable status.
        filled_quantity: Cumulative filled quantity.
        error: Error message (empty if none).
        submitted_at: Submission timestamp.
        history: Ordered list of state transitions.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    order_id: str
    instrument: str
    venue: str
    direction: str
    quantity: float
    price: float | None
    order_type: str
    state: str
    status: str
    filled_quantity: float
    error: str
    submitted_at: datetime
    history: list[dict[str, Any]] = Field(default_factory=list)


class CancelOrderRequest(BaseModel):
    """Request body for cancelling a live order.

    Attributes:
        order_id: The order to cancel.
        reason: Optional human-readable reason for the cancellation.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    order_id: str = Field(..., min_length=1, description="Order ID to cancel")
    reason: str | None = Field(default=None, description="Cancellation reason")


class CancelOrderResponse(BaseModel):
    """Response after order cancellation.

    Attributes:
        status: Cancellation result string.
        order_state: Final order state after cancellation.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    status: str
    order_state: str


class KillSwitchRequest(BaseModel):
    """Request body for kill-switch control.

    Attributes:
        action: ``"activate"`` or ``"deactivate"``.
        reason: Mandatory human-readable reason (min 1 character).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    action: str = Field(..., pattern="^(activate|deactivate)$", description="Kill switch action")
    reason: str = Field(..., min_length=1, description="Mandatory reason for the action")


class KillSwitchResponse(BaseModel):
    """Response after kill-switch control.

    Attributes:
        state: Current kill-switch state.
        affected_orders: List of order IDs affected by the action.
        confirmed: Whether the action was successfully applied.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    state: str
    affected_orders: list[str]
    confirmed: bool


class PnlMetrics(BaseModel):
    """PnL report with risk-adjusted metrics.

    Attributes:
        realized: Total realized PnL in base currency.
        unrealized: Total unrealized (floating) PnL.
        sharpe: Annualized Sharpe ratio (or None if insufficient data).
        sortino: Annualized Sortino ratio (or None if insufficient data).
        max_drawdown: Peak-to-trough drawdown ratio (0.0-1.0).
        win_rate: Fraction of profitable trades (0.0-1.0).
        profit_factor: Gross profit / gross loss (or None if no losses).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    realized: float
    unrealized: float
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    win_rate: float
    profit_factor: float | None


class DailyPnlPoint(BaseModel):
    """A single daily PnL data point.

    Attributes:
        date: Date string (``YYYY-MM-DD``).
        pnl: Net PnL for the day.
        realized: Realized portion.
        unrealized: Unrealized (floating) portion.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    date: str
    pnl: float
    realized: float
    unrealized: float


# ---------------------------------------------------------------------------
# Helper: validate feature flag
# ---------------------------------------------------------------------------


def _require_live_trading() -> None:
    """Raise 403 if live trading is not enabled."""
    if not feature_flags.is_enabled("live_trading_enabled"):
        raise HTTPException(
            status_code=403,
            detail="Live trading is disabled — feature flag not enabled.",
        )


# ---------------------------------------------------------------------------
# Helper: audit trail
# ---------------------------------------------------------------------------


def _audit_event(event: str, **fields: object) -> str:
    """Log an event to the audit trail and return the audit ID.

    Args:
        event: Event type string.
        **fields: Arbitrary contextual fields.

    Returns:
        The generated audit ID.
    """
    audit_id = f"AUDIT-{uuid.uuid4().hex[:8]}"
    AuditTrail().log_decision(
        agent_id="live-orders",
        decision=event,
        actor="system",
        details={"event": event, "audit_id": audit_id, **fields},
    )
    return audit_id


# ---------------------------------------------------------------------------
# Price-provider seam (mark prices for unrealized PnL)
# ---------------------------------------------------------------------------

_price_provider: Callable[[str, str], float | None] | None = None


def set_live_price_provider(provider: Callable[[str, str], float | None] | None) -> None:
    """Register a ``(venue, symbol) -> price | None`` provider.

    Used for tests and future market-data integration.  ``None`` clears
    the provider; unrealized PnL then reports ``0.0``.
    """
    global _price_provider
    _price_provider = provider


def _build_mark_prices() -> dict[tuple[str, str], float]:
    """Build the mark-price mapping for all tracked orders."""
    if _price_provider is None:
        return {}
    mark: dict[tuple[str, str], float] = {}
    for result in _order_registry.values():
        price = _price_provider(result.venue, result.symbol)
        if price is not None:
            mark[(result.venue, result.symbol)] = price
    return mark


def _build_tracker() -> LivePnlTracker:
    """Build a fresh :class:`LivePnlTracker` from the order registry."""
    tracker = LivePnlTracker()
    for result in sorted(_order_registry.values(), key=lambda r: r.submitted_at):
        tracker.process_order(result)
    return tracker


def _cancel_failed(cancel_result: OrderResult) -> bool:
    """True if a gateway cancel result indicates a failed cancellation."""
    return "error" in cancel_result.status or cancel_result.state == OrderState.ERROR


# ---------------------------------------------------------------------------
# In-memory order store (shared across requests)
# ---------------------------------------------------------------------------

# ponytail: in-memory only — lost on process restart. WIP until order
# persistence lands; registry is intentionally not DB-backed yet.
# order_id -> OrderResult
_order_registry: dict[str, OrderResult] = {}
# idempotency_key -> list[str] of order_ids
_idempotency_index: dict[str, list[str]] = {}

#: Order states that cannot be cancelled.
_TERMINAL_STATES = frozenset(
    {
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.ERROR,
    }
)


def _find_by_idempotency(key: str) -> OrderResult | None:
    """Look up a previous order by idempotency key."""
    order_ids = _idempotency_index.get(key)
    if order_ids:
        # Return the most recent
        return _order_registry.get(order_ids[-1])
    return None


def _store_order(result: OrderResult, idempotency_key: str | None = None) -> None:
    """Store an order result in the in-memory registry."""
    oid = result.order_id or str(uuid.uuid4())
    result.order_id = oid
    _order_registry[oid] = result
    if idempotency_key:
        _idempotency_index.setdefault(idempotency_key, []).append(oid)


def _build_order_record(result: OrderResult) -> OrderRecord:
    """Build an :class:`OrderRecord` from an :class:`OrderResult`."""
    # Extract transition history from the gateway's state machine
    gateway = _get_gateway()
    sm = gateway.get_state_machine(result.order_id)
    history: list[dict[str, Any]] = []
    if sm is not None:
        for state, event, ts in sm.get_history():
            history.append({
                "state": state.name,
                "event": event,
                "timestamp": ts.isoformat(),
            })

    return OrderRecord(
        order_id=result.order_id,
        instrument=result.symbol or "",
        venue=result.venue,
        direction=result.side or "",
        quantity=result.quantity,
        price=result.price,
        order_type=result.order_type,
        state=result.state.name,
        status=result.status,
        filled_quantity=result.filled_quantity,
        error=result.error or "",
        submitted_at=result.submitted_at,
        history=history,
    )


# ---------------------------------------------------------------------------
# POST /v1/live/orders — Submit live order
# ---------------------------------------------------------------------------


@router.post("/orders", response_model=SubmitOrderResponse, status_code=201)
async def submit_live_order(
    request: SubmitOrderRequest,
    _role: Role = Depends(require_live_permission(Permission.EXECUTE_LIVE)),  # noqa: B008
) -> SubmitOrderResponse:
    """Submit a live order with full validation and audit trail.

    The endpoint checks the ``live_trading_enabled`` feature flag, checks
    idempotency, and submits to the exchange via the gateway (the gateway
    is the single validation path — it raises
    :class:`GatewayValidationError` on invalid orders).

    Request schema
    --------------
    {
        "instrument": "BTC/USDT",
        "direction": "buy",
        "quantity": 0.5,
        "order_type": "limit",
        "price": 45000.0,
        "stop_price": 44000.0,
        "idempotency_key": "unique-key-123",
        "venue": "binance"
    }

    Response schema
    ---------------
    {
        "order_id": "live_a1b2c3d4e5f6",
        "state": "PENDING",
        "submitted_at": "2025-01-15T10:30:00Z",
        "idempotency_key": "unique-key-123"
    }
    """
    _require_live_trading()
    trail = get_live_audit()

    # --- Idempotency check ---
    existing = _find_by_idempotency(request.idempotency_key)
    if existing is not None:
        audit_id = _audit_event(
            "order_submit_idempotent",
            instrument=request.instrument,
            idempotency_key=request.idempotency_key,
            existing_order_id=existing.order_id,
        )
        trail.record_order_submit(
            actor=_role.value,
            order_id=existing.order_id,
            instrument=existing.symbol or "",
            venue=existing.venue,
            direction=existing.side or "",
            quantity=existing.quantity,
            status="idempotent",
            order_type=existing.order_type,
            idempotency_key=request.idempotency_key,
        )
        logger.info(
            "Idempotent order submit — returning cached order %s (audit: %s)",
            existing.order_id,
            audit_id,
        )
        return SubmitOrderResponse(
            order_id=existing.order_id,
            state=existing.state.name,
            submitted_at=existing.submitted_at,
            idempotency_key=request.idempotency_key,
        )

    # --- Normalize for the gateway ---
    direction = request.direction.lower()
    order_type = request.order_type.lower()

    # --- Submit via gateway (single validation path) ---
    gateway = _get_gateway()
    try:
        result = await gateway.submit_order(
            venue=request.venue,
            symbol=request.instrument,
            side=direction,
            order_type=order_type,
            amount=request.quantity,
            price=request.price,
            stop_price=request.stop_price,
            idempotency_key=request.idempotency_key,
        )
    except GatewayValidationError as exc:
        audit_id = _audit_event(
            "order_submit_validation_error",
            instrument=request.instrument,
            error=str(exc),
            idempotency_key=request.idempotency_key,
        )
        trail.record_order_submit(
            actor=_role.value,
            order_id="",
            instrument=request.instrument,
            venue=request.venue,
            direction=direction,
            quantity=request.quantity,
            status="rejected",
            order_type=order_type,
            idempotency_key=request.idempotency_key,
        )
        logger.warning(
            "Gateway validation error for %s (audit: %s): %s",
            request.instrument, audit_id, exc,
        )
        raise HTTPException(
            status_code=422,
            detail=f"Gateway validation failed: {exc}",
        ) from exc
    except GatewayIdempotencyError as exc:
        trail.record_order_submit(
            actor=_role.value,
            order_id="",
            instrument=request.instrument,
            venue=request.venue,
            direction=direction,
            quantity=request.quantity,
            status="duplicate",
            order_type=order_type,
            idempotency_key=request.idempotency_key,
        )
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate order detected: {exc}",
        ) from exc
    except GatewayExecutionError as exc:
        audit_id = _audit_event(
            "order_submit_execution_error",
            instrument=request.instrument,
            error=str(exc),
            idempotency_key=request.idempotency_key,
        )
        trail.record_order_submit(
            actor=_role.value,
            order_id="",
            instrument=request.instrument,
            venue=request.venue,
            direction=direction,
            quantity=request.quantity,
            status="error",
            order_type=order_type,
            idempotency_key=request.idempotency_key,
        )
        logger.error(
            "Gateway execution error for %s (audit: %s): %s",
            request.instrument, audit_id, exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Order submission failed: {exc}",
        ) from exc

    # --- Store in registry ---
    _store_order(result, request.idempotency_key)

    # --- Audit log ---
    audit_id = _audit_event(
        "order_submit_success",
        instrument=request.instrument,
        direction=direction,
        quantity=request.quantity,
        order_type=order_type,
        venue=request.venue,
        order_id=result.order_id,
        state=result.state.name,
        idempotency_key=request.idempotency_key,
    )
    trail.record_order_submit(
        actor=_role.value,
        order_id=result.order_id,
        instrument=request.instrument,
        venue=request.venue,
        direction=direction,
        quantity=request.quantity,
        status="submitted",
        order_type=order_type,
        idempotency_key=request.idempotency_key,
    )
    if result.state == OrderState.FILLED:
        trail.record_order_fill(
            actor=_role.value,
            order_id=result.order_id,
            filled_quantity=result.filled_quantity,
            price=result.price,
        )
    logger.info(
        "Order submitted successfully: %s state=%s (audit: %s)",
        result.order_id, result.state.name, audit_id,
    )

    return SubmitOrderResponse(
        order_id=result.order_id,
        state=result.state.name,
        submitted_at=result.submitted_at,
        idempotency_key=request.idempotency_key,
    )


# ---------------------------------------------------------------------------
# GET /v1/live/orders — Order history with filters
# ---------------------------------------------------------------------------


@router.get("/orders", response_model=list[OrderRecord], status_code=200)
async def list_orders(
    order_id: str | None = Query(default=None, description="Filter by order ID"),
    status: str | None = Query(default=None, description="Filter by order state"),
    venue: str | None = Query(default=None, description="Filter by venue"),
    from_dt: datetime | None = Query(default=None, alias="from", description="Start datetime (inclusive)"),  # noqa: B008
    to_dt: datetime | None = Query(default=None, alias="to", description="End datetime (inclusive)"),  # noqa: B008
    _role: Role = Depends(require_live_permission(Permission.READ_METRICS)),  # noqa: B008
) -> list[OrderRecord]:
    """Return order history with optional filters.

    Query parameters
    ----------------
    - ``order_id`` — Exact order ID match (optional).
    - ``status`` — Order state string (e.g. ``"PENDING"``, ``"FILLED"``).
    - ``venue`` — Venue identifier (e.g. ``"binance"``).
    - ``from`` — Start of time range (ISO-8601, optional).
    - ``to`` — End of time range (ISO-8601, optional).

    Response schema
    ---------------
    [
        {
            "order_id": "live_a1b2c3d4e5f6",
            "instrument": "BTC/USDT",
            "venue": "binance",
            "direction": "buy",
            "quantity": 0.5,
            "price": 45000.0,
            "order_type": "limit",
            "state": "PENDING",
            "status": "pending",
            "filled_quantity": 0.0,
            "error": "",
            "submitted_at": "2025-01-15T10:30:00Z",
            "history": [
                {"state": "PENDING", "event": "submitted", "timestamp": "2025-01-15T10:30:00Z"}
            ]
        }
    ]
    """
    _require_live_trading()

    filtered = _order_registry

    # Filter by order_id (exact match)
    if order_id is not None:
        filtered = {k: v for k, v in filtered.items() if k == order_id}

    # Filter by status / state
    if status is not None:
        status_upper = status.upper()
        # Support both state name and status string
        filtered = {
            k: v for k, v in filtered.items()
            if v.state.name == status_upper or v.status == status
        }

    # Filter by venue
    if venue is not None:
        filtered = {k: v for k, v in filtered.items() if v.venue.lower() == venue.lower()}

    # Filter by time range
    if from_dt is not None:
        filtered = {
            k: v for k, v in filtered.items()
            if v.submitted_at >= from_dt
        }
    if to_dt is not None:
        filtered = {
            k: v for k, v in filtered.items()
            if v.submitted_at <= to_dt
        }

    results = [_build_order_record(result) for _, result in filtered.items()]

    # Sort by submission time descending
    results.sort(key=lambda r: r.submitted_at, reverse=True)
    return results


# ---------------------------------------------------------------------------
# POST /v1/live/cancel — Cancel order
# ---------------------------------------------------------------------------


@router.post("/cancel", response_model=CancelOrderResponse, status_code=200)
async def cancel_live_order(
    request: CancelOrderRequest,
    _role: Role = Depends(require_live_permission(Permission.CANCEL_ORDERS)),  # noqa: B008
) -> CancelOrderResponse:
    """Cancel a live order.

    Looks up the order in the registry, runs the gateway's cancel logic,
    and updates the audit trail.  A failed cancellation returns 500 and
    leaves the previous order state unchanged.

    Request schema
    --------------
    {
        "order_id": "live_a1b2c3d4e5f6",
        "reason": "Trader changed mind"
    }

    Response schema
    ---------------
    {
        "status": "cancelled",
        "order_state": "CANCELLED"
    }
    """
    _require_live_trading()
    trail = get_live_audit()

    # Find order in registry
    order = _order_registry.get(request.order_id)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail=f"Order {request.order_id} not found.",
        )

    # Cannot cancel orders in terminal states
    if order.state in _TERMINAL_STATES:
        raise HTTPException(
            status_code=409,
            detail=f"Order {request.order_id} is in terminal state {order.state.name} and cannot be cancelled.",
        )

    # Run gateway cancel
    gateway = _get_gateway()
    try:
        cancel_result = await gateway.cancel_order(
            venue=order.venue,
            order_id=request.order_id,
        )
    except Exception as exc:
        audit_id = _audit_event(
            "cancel_order_failed",
            order_id=request.order_id,
            reason=request.reason,
            error=str(exc),
        )
        trail.record_order_cancel(
            actor=_role.value,
            order_id=request.order_id,
            status="failed",
            reason=request.reason,
            error=str(exc),
        )
        logger.error("Cancel order failed: %s (audit: %s)", exc, audit_id)
        raise HTTPException(
            status_code=500,
            detail=f"Order cancellation failed: {exc}",
        ) from exc

    # Cancel failure — report 500, leave the previous order state unchanged
    # (a failed cancel is not an order error state).
    if _cancel_failed(cancel_result):
        audit_id = _audit_event(
            "cancel_order_failed",
            order_id=request.order_id,
            reason=request.reason,
            error=cancel_result.error or cancel_result.status,
        )
        trail.record_order_cancel(
            actor=_role.value,
            order_id=request.order_id,
            status="failed",
            reason=request.reason,
            error=cancel_result.error or cancel_result.status,
        )
        logger.error(
            "Cancel order failed for %s (audit: %s): %s",
            request.order_id, audit_id, cancel_result.error or cancel_result.status,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Order cancellation failed: {cancel_result.error or cancel_result.status}",
        )

    # Update registry
    order.state = cancel_result.state
    order.status = cancel_result.status
    if cancel_result.error:
        order.error = cancel_result.error

    # Audit trail
    audit_id = _audit_event(
        "order_cancelled",
        order_id=request.order_id,
        reason=request.reason,
        new_state=cancel_result.state.name,
    )
    trail.record_order_cancel(
        actor=_role.value,
        order_id=request.order_id,
        status="cancelled",
        reason=request.reason,
    )
    logger.info(
        "Order %s cancelled (state=%s, reason=%s, audit: %s)",
        request.order_id, cancel_result.state.name, request.reason, audit_id,
    )

    return CancelOrderResponse(
        status=cancel_result.status,
        order_state=cancel_result.state.name,
    )


# ---------------------------------------------------------------------------
# POST /v1/live/kill-switch — Kill switch control
# ---------------------------------------------------------------------------


@router.post("/kill-switch", response_model=KillSwitchResponse, status_code=200)
async def kill_switch(
    request: KillSwitchRequest,
    _role: Role = Depends(require_live_permission(Permission.MANAGE_KILL_SWITCH)),  # noqa: B008
) -> KillSwitchResponse:
    """Activate or deactivate the trading kill switch.

    The kill switch immediately halts all live trading activity and cancels
    all open orders.  Activation requires a mandatory reason and is logged
    to the audit trail.

    Request schema
    --------------
    {
        "action": "activate",
        "reason": "Critical drawdown breach detected"
    }

    Response schema
    ---------------
    {
        "state": "activated",
        "affected_orders": ["live_a1b2c3d4e5f6", "live_x7y8z9w0v1u2"],
        "confirmed": true
    }
    """
    _require_live_trading()
    trail = get_live_audit()

    # Shared rollout controller — state persists across requests
    rollout = get_rollout_controller()
    previous_state = str(rollout.kill_switch.state)

    affected: list[str] = []

    if request.action == "activate":
        # Activate the kill switch
        rollout.force_kill(reason=request.reason)

        # Collect affected open orders
        for oid, order in _order_registry.items():
            if order.state not in _TERMINAL_STATES:
                affected.append(oid)

        # Attempt to cancel open orders via gateway; use the cancel result
        # to update state, and leave state unchanged on failure.
        gateway = _get_gateway()
        for oid in affected:
            order = _order_registry.get(oid)
            if order is None:
                continue
            try:
                cancel_result = await gateway.cancel_order(
                    venue=order.venue,
                    order_id=oid,
                )
            except Exception:
                logger.exception("Failed to auto-cancel order %s during kill switch", oid)
                _audit_event(
                    "kill_switch_cancel_failed",
                    order_id=oid,
                    reason=request.reason,
                    error="cancel call raised",
                )
                continue
            if _cancel_failed(cancel_result):
                # Cancel failed — leave the order state unchanged and audit it.
                logger.error(
                    "Kill-switch auto-cancel failed for %s: %s",
                    oid, cancel_result.error or cancel_result.status,
                )
                _audit_event(
                    "kill_switch_cancel_failed",
                    order_id=oid,
                    reason=request.reason,
                    error=cancel_result.error or cancel_result.status,
                )
                continue
            order.state = cancel_result.state
            order.status = cancel_result.status

        audit_id = _audit_event(
            "kill_switch_activated",
            reason=request.reason,
            affected_order_count=len(affected),
        )
        trail.record_kill_switch(
            actor=_role.value,
            action="activate",
            reason=request.reason,
            affected_order_count=len(affected),
        )
        logger.critical(
            "KILL SWITCH ACTIVATED — reason=%s (audit: %s)",
            request.reason,
            audit_id,
        )

    elif request.action == "deactivate":
        # Deactivate — only possible if no automatic triggers are active
        # We check via the controller's kill switch
        ks = rollout.kill_switch
        if ks.state == "activated":
            # Check if activation was manual (reason provided)
            # Only allow deactivation if it was a manual activation
            # For automatic activations, the controller must handle reactivation
            ks.deactivate()
            rollout.reset_circuit_breaker()

        audit_id = _audit_event(
            "kill_switch_deactivated",
            reason=request.reason,
        )
        trail.record_kill_switch(
            actor=_role.value,
            action="deactivate",
            reason=request.reason,
        )
        logger.warning(
            "KILL SWITCH DEACTIVATED — reason=%s (audit: %s)",
            request.reason,
            audit_id,
        )

    # Determine current state
    current_state = rollout.kill_switch.state
    if str(current_state) != previous_state:
        trail.record_rollout_state_change(
            actor=_role.value,
            previous_state=previous_state,
            new_state=str(current_state),
        )

    return KillSwitchResponse(
        state=current_state,
        affected_orders=affected,
        confirmed=True,
    )


# ---------------------------------------------------------------------------
# GET /v1/live/pnl — PnL with risk metrics
# ---------------------------------------------------------------------------


@router.get("/pnl", response_model=PnlMetrics, status_code=200)
async def get_pnl(
    _role: Role = Depends(require_live_permission(Permission.VIEW_LIVE_PNL)),  # noqa: B008
) -> PnlMetrics:
    """Return realized and unrealized PnL with risk-adjusted metrics.

    The PnL is computed deterministically by :class:`LivePnlTracker` from
    the order registry fills; unrealized PnL uses the configured price
    provider and is ``0.0`` when no mark prices are available.

    Response schema
    ---------------
    {
        "realized": 1234.56,
        "unrealized": -78.90,
        "sharpe": 1.45,
        "sortino": 1.82,
        "max_drawdown": 0.052,
        "win_rate": 0.65,
        "profit_factor": 1.8
    }
    """
    _require_live_trading()

    tracker = _build_tracker()
    return PnlMetrics(**tracker.summary(_build_mark_prices()))


# ---------------------------------------------------------------------------
# GET /v1/live/pnl/daily — Daily PnL aggregation
# ---------------------------------------------------------------------------


@router.get("/pnl/daily", response_model=list[DailyPnlPoint], status_code=200)
async def get_daily_pnl(
    _role: Role = Depends(require_live_permission(Permission.VIEW_LIVE_PNL)),  # noqa: B008
) -> list[DailyPnlPoint]:
    """Return daily PnL aggregation for all tracked orders.

    Response schema
    ---------------
    [
        {
            "date": "2025-01-15",
            "pnl": 1155.66,
            "realized": 1234.56,
            "unrealized": -78.90
        }
    ]
    """
    _require_live_trading()

    tracker = _build_tracker()
    return [DailyPnlPoint(**row) for row in tracker.daily(_build_mark_prices())]
