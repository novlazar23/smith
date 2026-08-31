"""Order State Machine for live execution.

Implements a strict 8-state finite-state machine governing the lifecycle of
every order from creation through fill, cancellation, or error.

States
------
``NEW``          — Order just created, not yet submitted.
``PENDING``      — Order submitted to the venue, awaiting confirmation.
``PARTIALLY_FILLED`` — A portion of the order has been filled.
``FILLED``       — The order is completely filled.
``CANCELLED``    — The order was cancelled (voluntary or forced).
``REJECTED``     — The venue rejected the order (e.g. invalid price).
``EXPIRED``      — The order expired (time-in-force expiry, no fill).
``ERROR``        — An unexpected failure occurred at any point.

Valid Transitions
-----------------
::

    NEW ──────────► PENDING
      │               │
      │               ├──► PARTIALLY_FILLED ──┐
      │               │        │               │
      │               │        ▼               │
      │               │   FILLED ◄──────────────┤
      │               │   CANCELLED ────────────┤
      │               │   REJECTED ─────────────┤
      │               │   EXPIRED ──────────────┤
      │               │
      │               ▼
      │           ERROR ◄──────────────────────┘
      │
      └────────────► ERROR

Every state has an outgoing ``► ERROR`` transition to handle unexpected
failures (network outage, exchange API crash, unexpected response).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)

# ─── State Enum ─────────────────────────────────────────────────────────────


class OrderState(Enum):
    """Lifecycle state of a live order.

    Every state transition is guarded by :class:`OrderStateMachine`.
    """

    #: Order created but not yet sent to the venue.
    NEW = auto()

    #: Sent to the venue, awaiting acknowledgement/fill.
    PENDING = auto()

    #: Partial fill received; remaining quantity still active.
    PARTIALLY_FILLED = auto()

    #: Fully filled.
    FILLED = auto()

    #: Cancelled by the trader or venue.
    CANCELLED = auto()

    #: Venue rejected the order (bad price, insufficient balance, etc.).
    REJECTED = auto()

    #: Order expired by time-in-force (IOC/FOK/GTC expiry).
    EXPIRED = auto()

    #: Unexpected failure — always an absorbing state.
    ERROR = auto()


# ─── Transition Definition ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Transition:
    """Defines a permitted state transition.

    Attributes:
        from_state: Source state.
        to_state: Target state.
        event: Human-readable event that triggered the transition.
    """

    from_state: OrderState
    to_state: OrderState
    event: str


# ─── Allowed Transitions ────────────────────────────────────────────────────

# Every permitted transition is listed here.  If a transition is not in this
# set, :class:`OrderStateMachine` will reject it.
_ALLOWED_TRANSITIONS: frozenset[tuple[OrderState, OrderState]] = frozenset(
    [
        # NEW → PENDING (submitted to venue)
        (OrderState.NEW, OrderState.PENDING),
        # NEW → REJECTED (venue rejects immediately)
        (OrderState.NEW, OrderState.REJECTED),
        # NEW → ERROR
        (OrderState.NEW, OrderState.ERROR),
        # PENDING → PARTIALLY_FILLED
        (OrderState.PENDING, OrderState.PARTIALLY_FILLED),
        # PENDING → FILLED
        (OrderState.PENDING, OrderState.FILLED),
        # PENDING → CANCELLED
        (OrderState.PENDING, OrderState.CANCELLED),
        # PENDING → REJECTED
        (OrderState.PENDING, OrderState.REJECTED),
        # PENDING → EXPIRED
        (OrderState.PENDING, OrderState.EXPIRED),
        # PENDING → ERROR
        (OrderState.PENDING, OrderState.ERROR),
        # PARTIALLY_FILLED → FILLED (remaining filled)
        (OrderState.PARTIALLY_FILLED, OrderState.FILLED),
        # PARTIALLY_FILLED → CANCELLED (remaining cancelled)
        (OrderState.PARTIALLY_FILLED, OrderState.CANCELLED),
        # PARTIALLY_FILLED → EXPIRED (remaining expired)
        (OrderState.PARTIALLY_FILLED, OrderState.EXPIRED),
        # PARTIALLY_FILLED → ERROR
        (OrderState.PARTIALLY_FILLED, OrderState.ERROR),
        # FILLED — absorbing, no outgoing transitions.
        # CANCELLED — absorbing.
        # REJECTED — absorbing.
        # EXPIRED — absorbing.
        # ERROR — absorbing.
        # Any state → ERROR on unexpected failure
        (OrderState.NEW, OrderState.ERROR),
        (OrderState.PENDING, OrderState.ERROR),
        (OrderState.PARTIALLY_FILLED, OrderState.ERROR),
        (OrderState.FILLED, OrderState.ERROR),
        (OrderState.CANCELLED, OrderState.ERROR),
        (OrderState.REJECTED, OrderState.ERROR),
        (OrderState.EXPIRED, OrderState.ERROR),
    ]
)


# ─── Exception ──────────────────────────────────────────────────────────────


class StateTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""

    def __init__(self, from_state: OrderState, to_state: OrderState) -> None:
        super().__init__(
            f"Cannot transition from {from_state.name} to {to_state.name}."
        )
        self.from_state = from_state
        self.to_state = to_state


# ─── Order Snapshot ─────────────────────────────────────────────────────────


@dataclass
class OrderSnapshot:
    """Immutable snapshot of an order's state at a point in time.

    Attributes:
        order_id: Unique order identifier.
        symbol: Trading pair, e.g. ``"BTC/USDT"``.
        venue: Venue identifier, e.g. ``"binance"``.
        side: ``"buy"`` or ``"sell"``.
        quantity: Original order quantity.
        filled_quantity: Quantity filled so far.
        price: Limit price (``None`` for market orders).
        state: Current :class:`OrderState`.
        state_changed_at: Timestamp of the last state transition.
        event: Event that caused the current state.
        metadata: Arbitrary extra data.
    """

    order_id: str
    symbol: str
    venue: str
    side: str
    quantity: float
    filled_quantity: float
    price: float | None
    state: OrderState
    state_changed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    event: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── State Machine ──────────────────────────────────────────────────────────


class OrderStateMachine:
    """Strict 8-state finite-state machine for live orders.

    Every transition is validated against the allowed-transition set.
    Transitions that are not explicitly allowed are rejected with
    :exc:`StateTransitionError`.

    Args:
        order_id: Unique order identifier.
        symbol: Trading pair, e.g. ``"BTC/USDT"``.
        venue: Venue identifier.
        side: ``"buy"`` or ``"sell"``.
        quantity: Original order quantity.
        price: Limit price (``None`` for market orders).
    """

    # All terminal / absorbing states.
    ABSORBING: frozenset[OrderState] = frozenset(
        [
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.ERROR,
        ]
    )

    def __init__(
        self,
        order_id: str,
        symbol: str,
        venue: str,
        side: str,
        quantity: float,
        price: float | None = None,
    ) -> None:
        self.order_id = order_id
        self.symbol = symbol
        self.venue = venue
        self.side = side
        self.quantity = quantity
        self.price = price
        self._state: OrderState = OrderState.NEW
        self.filled_quantity: float = 0.0
        self._history: list[tuple[OrderState, str, datetime]] = []
        self._snapshot = OrderSnapshot(
            order_id=order_id,
            symbol=symbol,
            venue=venue,
            side=side,
            quantity=quantity,
            filled_quantity=0.0,
            price=price,
            state=OrderState.NEW,
        )

    @property
    def state(self) -> OrderState:
        """Current order state."""
        return self._state

    @property
    def is_terminal(self) -> bool:
        """``True`` if the order is in an absorbing state."""
        return self._state in self.ABSORBING

    # ── public transition methods ─────────────────────────────────────────

    def transition_to(
        self,
        target: OrderState,
        event: str = "",
        filled_quantity: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrderSnapshot:
        """Transition the order to a new state.

        Args:
            target: The target :class:`OrderState`.
            event: Human-readable description of the event.
            filled_quantity: Updated filled quantity (optional).
            metadata: Extra data to attach to the snapshot.

        Returns:
            An :class:`OrderSnapshot` of the order after the transition.

        Raises:
            StateTransitionError: If the transition is not allowed.
        """
        if target not in (
            OrderState.PENDING,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.ERROR,
        ):
            raise StateTransitionError(self._state, target)

        if (self._state, target) not in _ALLOWED_TRANSITIONS:
            raise StateTransitionError(self._state, target)

        self._state = target
        timestamp = datetime.now(UTC)
        self._history.append((self._state, event, timestamp))

        if filled_quantity is not None:
            self.filled_quantity = filled_quantity

        if event:
            self._snapshot.event = event

        if metadata:
            self._snapshot.metadata.update(metadata)

        self._snapshot = OrderSnapshot(
            order_id=self.order_id,
            symbol=self.symbol,
            venue=self.venue,
            side=self.side,
            quantity=self.quantity,
            filled_quantity=self.filled_quantity,
            price=self.price,
            state=self._state,
            state_changed_at=timestamp,
            event=event,
            metadata=dict(self._snapshot.metadata),
        )

        logger.info(
            "Order %s: %s → %s [%s]",
            self.order_id,
            self._snapshot.state.name,
            target.name,
            event or "unknown",
        )

        return self._snapshot

    def update_fill(
        self,
        filled_quantity: float,
        fill_price: float,
        event: str = "fill",
    ) -> OrderSnapshot:
        """Record a fill event.

        Handles both partial fills and full fills.  If the remaining
        quantity is zero, transitions to ``FILLED``.

        Args:
            filled_quantity: Cumulative filled quantity.
            fill_price: Price of this fill.
            event: Event description.

        Returns:
            Updated :class:`OrderSnapshot`.
        """
        self.filled_quantity = filled_quantity

        remaining = self.quantity - filled_quantity

        if remaining <= 0:
            # Order is fully filled
            return self.transition_to(
                OrderState.FILLED,
                event=event or "filled",
                filled_quantity=filled_quantity,
            )
        elif self._state == OrderState.PENDING:
            return self.transition_to(
                OrderState.PARTIALLY_FILLED,
                event=event or "partial_fill",
                filled_quantity=filled_quantity,
            )
        else:
            # Already partially filled, just update quantity
            self._snapshot.filled_quantity = filled_quantity
            self._snapshot.metadata["fill_price"] = fill_price
            return self._snapshot

    def transition_from_error(self, event: str = "error") -> OrderSnapshot:
        """Force transition to ERROR from any state.

        Convenience method for unexpected failures.

        Args:
            event: Description of the error event.

        Returns:
            Updated :class:`OrderSnapshot`.
        """
        return self.transition_to(OrderState.ERROR, event=event)

    def reset_to_new(self) -> None:
        """Reset the state machine back to ``NEW``.

        Useful for order retries.  Clears history and resets fill count.
        """
        self._state = OrderState.NEW
        self.filled_quantity = 0.0
        self._history = []
        self._snapshot = OrderSnapshot(
            order_id=self.order_id,
            symbol=self.symbol,
            venue=self.venue,
            side=self.side,
            quantity=self.quantity,
            filled_quantity=0.0,
            price=self.price,
            state=OrderState.NEW,
        )

    def to_snapshot(self) -> OrderSnapshot:
        """Return the current :class:`OrderSnapshot`."""
        self._snapshot.state_changed_at = datetime.now(UTC)
        return self._snapshot

    def get_history(self) -> list[tuple[OrderState, str, datetime]]:
        """Return the full transition history."""
        return list(self._history)

    def is_transition_allowed(
        self, from_state: OrderState, to_state: OrderState
    ) -> bool:
        """Check whether a transition is allowed (without executing)."""
        return (from_state, to_state) in _ALLOWED_TRANSITIONS
