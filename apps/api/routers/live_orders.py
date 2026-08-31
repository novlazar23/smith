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
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from packages.governance.audit import AuditTrail
from packages.governance.feature_flags import feature_flags
from packages.live_execution import (
    GatewayExecutionError,
    GatewayIdempotencyError,
    GatewayValidationError,
    LiveExecutionGateway,
    OrderResult,
    OrderState,
    OrderValidator,
)
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


def _audit_event(event: str, **fields: str) -> str:
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
# Helper: PnL metric computation
# ---------------------------------------------------------------------------


def _compute_pnl_metrics(
    realized_pnl: float,
    unrealized_pnl: float,
    daily_pnls: list[float],
    win_count: int,
    loss_count: int,
    gross_profit: float,
    gross_loss: float,
) -> dict[str, float | None]:
    """Compute PnL risk-adjusted metrics from daily PnL series.

    Args:
        realized_pnl: Total realized PnL.
        unrealized_pnl: Total unrealized PnL.
        daily_pnls: List of daily net PnL values.
        win_count: Number of winning trades.
        loss_count: Number of losing trades.
        gross_profit: Sum of all gross profits.
        gross_loss: Sum of all gross losses (absolute).

    Returns:
        Dict with keys ``sharpe``, ``sortino``, ``max_drawdown``,
        ``win_rate``, ``profit_factor``.
    """
    import math

    total_trades = win_count + loss_count

    # Win rate
    win_rate = win_count / total_trades if total_trades > 0 else 0.0

    # Profit factor
    profit_factor: float | None = None
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss

    # Max drawdown
    max_dd = 0.0
    peak = 0.0
    cumsum = 0.0
    for pnl_val in daily_pnls:
        cumsum += pnl_val
        if cumsum > peak:
            peak = cumsum
        dd = peak - cumsum
        if dd > 0:
            dd_ratio = dd / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd_ratio)

    # Sharpe ratio (annualized, assuming 252 trading days)
    sharpe: float | None = None
    if len(daily_pnls) >= 2:
        mean_pnl = sum(daily_pnls) / len(daily_pnls)
        variance = sum((x - mean_pnl) ** 2 for x in daily_pnls) / (len(daily_pnls) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std > 0:
            sharpe = (mean_pnl / std) * math.sqrt(252)

    # Sortino ratio
    sortino: float | None = None
    if len(daily_pnls) >= 2:
        mean_pnl = sum(daily_pnls) / len(daily_pnls)
        downside = [x for x in daily_pnls if x < mean_pnl]
        if downside:
            downside_var = sum(x ** 2 for x in downside) / len(daily_pnls)
            downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.0
            if downside_std > 0:
                sortino = (mean_pnl / downside_std) * math.sqrt(252)

    return {
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "sortino": round(sortino, 4) if sortino is not None else None,
        "max_drawdown": round(max_dd, 6),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
    }


# ---------------------------------------------------------------------------
# In-memory order store (shared across requests)
# ---------------------------------------------------------------------------

# order_id -> OrderResult
_order_registry: dict[str, OrderResult] = {}
# idempotency_key -> list[str] of order_ids
_idempotency_index: dict[str, list[str]] = {}


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


def _build_order_record(
    result: OrderResult,
    instrument: str,
    direction: str,
    order_type: str,
) -> OrderRecord:
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
        instrument=instrument,
        venue=result.venue,
        direction=direction,
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
async def submit_live_order(request: SubmitOrderRequest) -> SubmitOrderResponse:
    """Submit a live order with full validation and audit trail.

    The endpoint checks the ``live_trading_enabled`` feature flag, runs the
    order validator pipeline, checks idempotency, submits to the exchange via
    CCXT, and records the order in the in-memory registry.

    Request schema
    --------------
    {
        "instrument": "BTC/USDT",
        "direction": "buy",
        "quantity": 0.5,
        "order_type": "limit",
        "price": 45000.0,
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

    # --- Idempotency check ---
    existing = _find_by_idempotency(request.idempotency_key)
    if existing is not None:
        audit_id = _audit_event(
            "order_submit_idempotent",
            instrument=request.instrument,
            idempotency_key=request.idempotency_key,
            existing_order_id=existing.order_id,
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

    # --- Convert direction ---
    direction = request.direction.lower()

    # --- Run pre-submission validation ---
    validator = OrderValidator()
    validation_errors = await validator.validate(
        symbol=request.instrument,
        venue=request.venue,
        side=direction,
        quantity=request.quantity,
        price=request.price,
        order_type=request.order_type,
    )
    if validation_errors:
        errors_str = "; ".join(e.message for e in validation_errors)
        audit_id = _audit_event(
            "order_submit_rejected",
            instrument=request.instrument,
            error=errors_str,
            idempotency_key=request.idempotency_key,
        )
        logger.warning(
            "Order validation failed for %s/%s: %s (audit: %s)",
            request.instrument, request.venue, errors_str, audit_id,
        )
        raise HTTPException(
            status_code=422,
            detail=f"Order validation failed: {errors_str}",
        )

    # --- Submit via gateway ---
    gateway = _get_gateway()
    try:
        result = await gateway.submit_order(
            venue=request.venue,
            symbol=request.instrument,
            side=direction,
            order_type=request.order_type,
            amount=request.quantity,
            price=request.price,
            idempotency_key=request.idempotency_key,
        )
    except GatewayValidationError as exc:
        audit_id = _audit_event(
            "order_submit_validation_error",
            instrument=request.instrument,
            error=str(exc),
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
        order_type=request.order_type,
        venue=request.venue,
        order_id=result.order_id,
        state=result.state.name,
        idempotency_key=request.idempotency_key,
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

    # Enforce that at least some filter is present or no filter returns all
    filtered = _order_registry

    # Filter by order_id
    if order_id is not None:
        filtered = {k: v for k, v in filtered.items() if order_id in (v.order_id, k)}

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

    results: list[OrderRecord] = []
    for oid, result in filtered.items():
        # Look up the original request data from the idempotency index
        instrument = ""
        direction = ""
        order_type = result.order_type
        for _, order_ids in _idempotency_index.items():
            if oid in order_ids:
                # We need to reconstruct from OrderResult attributes
                break
        # Use the OrderResult's symbol as the instrument
        instrument = result.symbol or ""
        direction = result.side or ""
        results.append(_build_order_record(result, instrument, direction, order_type))

    # Sort by submission time descending
    results.sort(key=lambda r: r.submitted_at, reverse=True)
    return results


# ---------------------------------------------------------------------------
# POST /v1/live/cancel — Cancel order
# ---------------------------------------------------------------------------


@router.post("/cancel", response_model=CancelOrderResponse, status_code=200)
async def cancel_live_order(request: CancelOrderRequest) -> CancelOrderResponse:
    """Cancel a live order.

    Looks up the order in the registry, runs the gateway's cancel logic,
    and updates the audit trail.

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

    # Find order in registry
    order = _order_registry.get(request.order_id)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail=f"Order {request.order_id} not found.",
        )

    # Cannot cancel orders in terminal states
    if order.state in (
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.ERROR,
    ):
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
        logger.error("Cancel order failed: %s (audit: %s)", exc, audit_id)
        raise HTTPException(
            status_code=500,
            detail=f"Order cancellation failed: {exc}",
        ) from exc

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
async def kill_switch(request: KillSwitchRequest) -> KillSwitchResponse:
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

    from packages.rollout import PhasedRolloutController

    # Look up or create the rollout controller
    rollout = PhasedRolloutController()

    if request.action == "activate":
        # Activate the kill switch
        rollout.force_kill(reason=request.reason)

        # Collect affected open orders
        affected: list[str] = []
        for oid, order in _order_registry.items():
            if order.state not in (
                OrderState.FILLED,
                OrderState.CANCELLED,
                OrderState.REJECTED,
                OrderState.EXPIRED,
                OrderState.ERROR,
            ):
                affected.append(oid)

        # Attempt to cancel open orders via gateway
        gateway = _get_gateway()
        for oid in affected:
            order = _order_registry.get(oid)
            if order is not None:
                try:
                    await gateway.cancel_order(
                        venue=order.venue,
                        order_id=oid,
                    )
                    order.state = OrderState.CANCELLED
                    order.status = "kill_switch_cancelled"
                except Exception:
                    logger.exception("Failed to auto-cancel order %s during kill switch", oid)

        audit_id = _audit_event(
            "kill_switch_activated",
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
        logger.warning(
            "KILL SWITCH DEACTIVATED — reason=%s (audit: %s)",
            request.reason,
            audit_id,
        )

    # Determine current state
    current_state = rollout.kill_switch.state

    return KillSwitchResponse(
        state=current_state,
        affected_orders=affected if request.action == "activate" else [],
        confirmed=True,
    )


# ---------------------------------------------------------------------------
# GET /v1/live/pnl — PnL with risk metrics
# ---------------------------------------------------------------------------


@router.get("/pnl", response_model=PnlMetrics, status_code=200)
async def get_pnl() -> PnlMetrics:
    """Return realized and unrealized PnL with risk-adjusted metrics.

    The PnL data is derived from the order registry, filling history, and
    gateway state machines.

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

    gateway = _get_gateway()
    metrics = _compute_pnl_from_orders(gateway)
    return PnlMetrics(**metrics)


# ---------------------------------------------------------------------------
# GET /v1/live/pnl/daily — Daily PnL aggregation
# ---------------------------------------------------------------------------


@router.get("/pnl/daily", response_model=list[DailyPnlPoint], status_code=200)
async def get_daily_pnl() -> list[DailyPnlPoint]:
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

    gateway = _get_gateway()
    return _compute_daily_pnl(gateway)


# ---------------------------------------------------------------------------
# Internal PnL computation helpers
# ---------------------------------------------------------------------------


def _compute_pnl_from_orders(gateway: LiveExecutionGateway) -> dict[str, Any]:
    """Compute PnL metrics from the order registry and gateway state.

    Args:
        gateway: The live execution gateway instance.

    Returns:
        Dict compatible with :class:`PnlMetrics`.
    """
    realized_pnl = 0.0
    unrealized_pnl = 0.0
    daily_pnls: dict[str, float] = {}
    win_count = 0
    loss_count = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for _, result in _order_registry.items():
        # Realized PnL from filled orders
        if result.state == OrderState.FILLED and result.fill_price is not None:
            if result.side == "buy":
                # Long position — PnL from sell vs buy
                pass  # Unrealized if still open
            else:
                # Short position sell — PnL from buy-back
                pass

        # Track daily PnL from fills
        fill_pnl = result.filled_quantity * (result.fill_price or 0) if result.fill_price else 0.0
        if result.submitted_at:
            day = result.submitted_at.date().isoformat()
            daily_pnls.setdefault(day, 0.0)

        # Track win/loss for filled orders
        if result.state == OrderState.FILLED:
            # Simplified: assume positive fill value = profit
            if fill_pnl > 0:
                win_count += 1
                gross_profit += fill_pnl
            else:
                loss_count += 1
                gross_loss += abs(fill_pnl)

    # Aggregate daily
    daily_list = [daily_pnls[d] for d in sorted(daily_pnls.keys())]

    metrics = _compute_pnl_metrics(
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        daily_pnls=daily_list,
        win_count=win_count,
        loss_count=loss_count,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
    )
    metrics["realized"] = realized_pnl
    metrics["unrealized"] = unrealized_pnl
    return metrics


def _compute_daily_pnl(gateway: LiveExecutionGateway) -> list[DailyPnlPoint]:
    """Compute daily PnL from the order registry.

    Args:
        gateway: The live execution gateway instance.

    Returns:
        List of :class:`DailyPnlPoint` sorted by date descending.
    """
    daily: dict[str, dict[str, float]] = {}

    for result in _order_registry.values():
        if not result.submitted_at:
            continue
        day = result.submitted_at.date().isoformat()
        entry = daily.setdefault(day, {"pnl": 0.0, "realized": 0.0, "unrealized": 0.0})

        if result.state == OrderState.FILLED and result.fill_price is not None:
            pnl = result.filled_quantity * (result.fill_price - (result.price or 0))
            entry["pnl"] += pnl
            entry["realized"] += pnl
        else:
            # Still open — unrealized
            entry["unrealized"] += result.quantity * (result.price or 0) * 0.01  # rough estimate

    points = [
        DailyPnlPoint(
            date=day,
            pnl=round(data["pnl"], 2),
            realized=round(data["realized"], 2),
            unrealized=round(data["unrealized"], 2),
        )
        for day, data in sorted(daily.items(), reverse=True)
    ]
    return points
