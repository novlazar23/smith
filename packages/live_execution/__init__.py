"""Live execution package for real-order trading.

This package provides the infrastructure for submitting, tracking, and
managing live orders across multiple cryptocurrency venues using CCXT
as the unified exchange API.

Sub-modules
-----------
- ``gateway`` — :class:`.LiveExecutionGateway`: CCXT-based order submission.
- ``order_state_machine`` — :class:`.OrderStateMachine`: 8-state lifecycle.
- ``router`` — :class:`.OrderRouter`: single → multi-venue routing.
- ``rate_limiter`` — :class:`.RateLimiter`: token-bucket per venue.
- ``idempotency`` — :class:`.IdempotencyStore`: duplicate submit prevention.
- ``validator`` — :class:`.OrderValidator`: pre-submission risk gates.

Quick start
-----------

.. code-block:: python

    from packages.live_execution import (
        LiveExecutionGateway,
        OrderValidator,
        RateLimiter,
    )

    gateway = LiveExecutionGateway(
        ccxt_config={
            "binance": {
                "apiKey": "KEY",
                "secret": "SECRET",
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

from packages.live_execution.gateway import (
    GatewayError,
    GatewayExecutionError,
    GatewayIdempotencyError,
    GatewayValidationError,
    LiveExecutionGateway,
    OrderResult,
)
from packages.live_execution.idempotency import (
    IdempotencyError,
    IdempotencyStore,
)
from packages.live_execution.order_state_machine import (
    OrderSnapshot,
    OrderState,
    OrderStateMachine,
    StateTransitionError,
    Transition,
)
from packages.live_execution.rate_limiter import (
    RateLimiter,
    RateLimitExceededError,
)
from packages.live_execution.router import (
    RouteStrategy,
    SplitAllocation,
    SplitOrderResult,
    VenueInfo,
)
from packages.live_execution.validator import (
    OrderValidator,
    RiskGateResult,
    ValidationConfig,
    ValidationError,
)

__all__ = [
    "GatewayError",
    "GatewayExecutionError",
    "GatewayIdempotencyError",
    "GatewayValidationError",
    "IdempotencyError",
    "IdempotencyStore",
    "LiveExecutionGateway",
    "OrderResult",
    "OrderSnapshot",
    "OrderState",
    "OrderStateMachine",
    "OrderValidator",
    "RateLimitExceededError",
    "RateLimiter",
    "RiskGateResult",
    "RouteStrategy",
    "SplitAllocation",
    "SplitOrderResult",
    "StateTransitionError",
    "Transition",
    "ValidationConfig",
    "ValidationError",
    "VenueInfo",
]
