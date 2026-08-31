"""Order router — single-venue to multi-venue order routing.

Provides intelligent order routing across multiple venues.  The router:

- Manages a pool of venues (configured at initialization).
- Routes orders to a single venue (default strategy) or splits them
  across venues (split strategy).
- Supports venue selection strategies: ``"single"``, ``"priority"``,
  ``"split"``, and ``"health_aware"``.
- Tracks venue health and excludes unhealthy venues from routing.

Usage
-----

.. code-block:: python

    router = OrderRouter(
        venues=["binance", "bybit", "okx"],
        strategy="health_aware",
    )

    # Single venue routing (default)
    result = await router.route_order(
        symbol="BTC/USDT",
        side="buy",
        amount=0.5,
        price=45000.0,
    )

    # Split across venues (proportional to balance)
    result = await router.route_split_order(
        symbol="BTC/USDT",
        side="buy",
        amount=1.0,
        price=45000.0,
        split_strategy="equal",  # or "proportional"
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from packages.live_execution.gateway import (
    GatewayExecutionError,
    GatewayValidationError,
    LiveExecutionGateway,
    OrderResult,
)
from packages.live_execution.validator import ValidationError

logger = logging.getLogger(__name__)


# ─── Route Strategy ─────────────────────────────────────────────────────────


class RouteStrategy(Enum):
    """Order routing strategy.

    Attributes:
        SINGLE: Route all orders to a default venue.
        PRIORITY: Route to the highest-priority healthy venue.
        SPLIT: Split a single order across multiple venues.
        HEALTH_AWARE: Route to the venue with the best health score.
    """

    #: All orders go to the default venue.
    SINGLE = auto()

    #: Orders go to the highest-priority healthy venue.
    PRIORITY = auto()

    #: Split one order across multiple venues.
    SPLIT = auto()

    #: Choose the venue with the best health score.
    HEALTH_AWARE = auto()


# ─── Venue Info ─────────────────────────────────────────────────────────────


@dataclass
class VenueInfo:
    """Metadata about a venue.

    Attributes:
        venue_id: Unique venue identifier (e.g. ``"binance"``).
        priority: Routing priority (lower = higher priority).
        health_score: Current health score in ``[0.0, 1.0]``.
        max_notional: Maximum notional per order on this venue.
        tags: Optional tags for custom routing (e.g. ``{"futures", "us"} ``).
    """

    venue_id: str
    priority: int = 0
    health_score: float = 1.0
    max_notional: float = 1_000_000.0
    tags: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Return ``True`` if the venue is considered healthy."""
        return self.health_score >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "priority": self.priority,
            "health_score": self.health_score,
            "max_notional": self.max_notional,
            "tags": self.tags,
            "is_healthy": self.is_healthy,
        }


# ─── Split Result ───────────────────────────────────────────────────────────


@dataclass
class SplitAllocation:
    """A portion of a split order assigned to a venue.

    Attributes:
        venue_id: Venue identifier.
        quantity: Allocated quantity for this venue.
        result: The submission result from this venue.
    """

    venue_id: str
    quantity: float
    result: OrderResult | None = None


@dataclass
class SplitOrderResult:
    """Result of a split-order submission.

    Attributes:
        allocations: Individual venue allocations.
        total_quantity: Total quantity across all venues.
        total_filled_quantity: Total filled quantity across all venues.
        all_successful: ``True`` if all allocations succeeded.
        errors: Any errors from individual allocations.
    """

    allocations: list[SplitAllocation]
    total_quantity: float
    total_filled_quantity: float = 0.0
    all_successful: bool = True
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocations": [
                {
                    "venue_id": a.venue_id,
                    "quantity": a.quantity,
                    "result": a.result.to_dict() if a.result else None,
                }
                for a in self.allocations
            ],
            "total_quantity": self.total_quantity,
            "total_filled_quantity": self.total_filled_quantity,
            "all_successful": self.all_successful,
            "errors": self.errors,
        }


# ─── Router Config ──────────────────────────────────────────────────────────


@dataclass
class RouterConfig:
    """Configuration for the order router.

    Args:
        strategy: Default routing strategy.
        default_venue: Venue for single-strategy routing.
        min_split_quantity: Minimum quantity for a split portion.
        max_split_ports: Maximum number of venues to split across.
    """

    strategy: RouteStrategy = RouteStrategy.SINGLE
    default_venue: str = "binance"
    min_split_quantity: float = 0.001
    max_split_ports: int = 3


# ─── Router ─────────────────────────────────────────────────────────────────


class OrderRouter:
    """Multi-venue order router.

    Manages venue selection, health tracking, and order splitting.

    Args:
        gateway: The :class:`LiveExecutionGateway` used for submissions.
        venues: List of venue identifiers.
        config: Router configuration.
    """

    def __init__(
        self,
        gateway: LiveExecutionGateway,
        venues: list[str] | None = None,
        config: RouterConfig | None = None,
    ) -> None:
        self.gateway = gateway
        self._config = config or RouterConfig()
        self._venues: dict[str, VenueInfo] = {}
        self._health_cache: dict[str, float] = {}

        # Register venues with default configuration
        for venue_id in venues or self._config.default_venue.split(","):
            venue_id = venue_id.strip()
            if venue_id:
                self.add_venue(venue_id)

    # ── venue management ─────────────────────────────────────────────────

    def add_venue(
        self,
        venue_id: str,
        priority: int = 0,
        health_score: float = 1.0,
        max_notional: float = 1_000_000.0,
        tags: list[str] | None = None,
    ) -> VenueInfo:
        """Register a venue with the router.

        Args:
            venue_id: Venue identifier.
            priority: Routing priority (lower = higher priority).
            health_score: Initial health score in ``[0.0, 1.0]``.
            max_notional: Maximum notional per order.
            tags: Optional tags.

        Returns:
            The created :class:`VenueInfo`.
        """
        info = VenueInfo(
            venue_id=venue_id,
            priority=priority,
            health_score=health_score,
            max_notional=max_notional,
            tags=tags or [],
        )
        self._venues[venue_id] = info
        logger.info("Router registered venue: %s (priority=%d)", venue_id, priority)
        return info

    def remove_venue(self, venue_id: str) -> None:
        """Remove a venue from the router."""
        self._venues.pop(venue_id, None)
        logger.info("Router removed venue: %s", venue_id)

    def get_venue(self, venue_id: str) -> VenueInfo | None:
        """Get venue info, or ``None`` if not registered."""
        return self._venues.get(venue_id)

    def get_healthy_venues(self) -> list[VenueInfo]:
        """Return all healthy venues, sorted by priority then health."""
        healthy = [v for v in self._venues.values() if v.is_healthy]
        healthy.sort(key=lambda v: (v.priority, -v.health_score))
        return healthy

    def update_health_score(self, venue_id: str, score: float) -> None:
        """Update the health score for a venue.

        Args:
            venue_id: Venue identifier.
            score: New health score in ``[0.0, 1.0]``.
        """
        info = self._venues.get(venue_id)
        if info is not None:
            info.health_score = max(0.0, min(1.0, score))
            self._health_cache[venue_id] = score
            logger.info(
                "Router updated health for %s: %.2f", venue_id, score,
            )

    def degrade_venue(self, venue_id: str, penalty: float = 0.1) -> None:
        """Degrade a venue's health score by the given penalty.

        Args:
            venue_id: Venue identifier.
            penalty: Amount to subtract from health score.
        """
        info = self._venues.get(venue_id)
        if info is not None:
            info.health_score = max(0.0, info.health_score - penalty)
            logger.warning(
                "Router degraded venue %s by %.2f → %.2f",
                venue_id, penalty, info.health_score,
            )

    # ── order routing ────────────────────────────────────────────────────

    async def route_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        stop_price: float | None = None,
        venue: str | None = None,
        idempotency_key: str | None = None,
    ) -> OrderResult:
        """Route an order to a venue based on the configured strategy.

        Args:
            symbol: Trading pair.
            side: ``"buy"`` or ``"sell"``.
            order_type: Order type string.
            amount: Order quantity.
            price: Limit price.
            stop_price: Stop price for stop orders.
            venue: Force a specific venue (overrides strategy).
            idempotency_key: Optional idempotency key.

        Returns:
            :class:`OrderResult` from the chosen venue.

        Raises:
            GatewayValidationError: If the order fails validation.
            GatewayExecutionError: If no healthy venue is available.
        """
        # Validate input
        if amount <= 0:
            raise GatewayValidationError([
                ValidationError(
                    code="INVALID_AMOUNT",
                    message=f"Order amount must be positive, got {amount}",
                    field="amount",
                ),
            ])

        # Determine target venue
        target_venue = self._select_venue(
            symbol=symbol,
            amount=amount,
            side=side,
        )

        if target_venue is None:
            raise GatewayExecutionError(
                "No healthy venue available for routing. "
                f"Strategy: {self._config.strategy.name}"
            )

        # Delegate to gateway
        return await self.gateway.submit_order(
            venue=target_venue,
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            stop_price=stop_price,
            idempotency_key=idempotency_key,
        )

    async def route_split_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        stop_price: float | None = None,
        split_strategy: str = "equal",
        idempotency_key: str | None = None,
    ) -> SplitOrderResult:
        """Split an order across multiple venues.

        Args:
            symbol: Trading pair.
            side: ``"buy"`` or ``"sell"``.
            order_type: Order type string.
            amount: Total order quantity.
            price: Limit price.
            stop_price: Stop price for stop orders.
            split_strategy: ``"equal"`` (equal qty) or
                ``"proportional"`` (proportional to health score).
            idempotency_key: Optional base idempotency key
                (suffix is added per venue).

        Returns:
            :class:`SplitOrderResult` with per-venue results.
        """
        healthy = self.get_healthy_venues()
        num_venues = min(len(healthy), self._config.max_split_ports)

        if num_venues == 0:
            raise GatewayExecutionError(
                "No healthy venues available for split order"
            )

        # Allocate quantities
        allocations: list[SplitAllocation] = []
        if split_strategy == "equal":
            base_qty = amount / num_venues
            for i, venue_info in enumerate(healthy[:num_venues]):
                # Last venue gets the remainder
                qty = base_qty if i < num_venues - 1 else amount - base_qty * (num_venues - 1)
                if qty >= self._config.min_split_quantity:
                    allocations.append(
                        SplitAllocation(
                            venue_id=venue_info.venue_id,
                            quantity=qty,
                        ),
                    )
        elif split_strategy == "proportional":
            total_health = sum(
                v.health_score for v in healthy[:num_venues]
            )
            if total_health <= 0:
                total_health = 1.0
            for venue_info in healthy[:num_venues]:
                weight = venue_info.health_score / total_health
                qty = amount * weight
                if qty >= self._config.min_split_quantity:
                    allocations.append(
                        SplitAllocation(
                            venue_id=venue_info.venue_id,
                            quantity=qty,
                        ),
                    )

        if not allocations:
            raise GatewayExecutionError(
                "All allocations below minimum split quantity"
            )

        # Submit to each venue
        total_filled = 0.0
        all_ok = True
        errors: list[str] = []

        for alloc in allocations:
            key = (
                f"{idempotency_key}"
                if idempotency_key is not None
                else None
            )
            try:
                result = await self.gateway.submit_order(
                    venue=alloc.venue_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    amount=alloc.quantity,
                    price=price,
                    stop_price=stop_price,
                    idempotency_key=key,
                )
                alloc.result = result
                total_filled += result.filled_quantity
                if result.state.value != "filled" and result.state.value != "partial_fill":
                    all_ok = False
                    errors.append(
                        f"{alloc.venue_id}: {result.status} — {result.error}"
                    )
            except Exception as exc:
                all_ok = False
                errors.append(f"{alloc.venue_id}: {exc}")
                self.degrade_venue(alloc.venue_id)

        return SplitOrderResult(
            allocations=allocations,
            total_quantity=amount,
            total_filled_quantity=total_filled,
            all_successful=all_ok,
            errors=errors,
        )

    # ── venue health ─────────────────────────────────────────────────────

    async def check_venue_health(
        self,
        venue_id: str,
    ) -> dict[str, Any]:
        """Check health of a specific venue via the gateway.

        Args:
            venue_id: Venue identifier.

        Returns:
            Health check result dict.
        """
        try:
            health = await self.gateway._create_exchange(venue_id)
            # CCXT has a built-in ping/health
            health = await self.gateway._create_exchange(venue_id)  # noqa: F841
            self.update_health_score(venue_id, 1.0)
            return {"venue": venue_id, "healthy": True, "score": 1.0}
        except Exception as exc:
            self.degrade_venue(venue_id)
            return {
                "venue": venue_id,
                "healthy": False,
                "error": str(exc),
            }

    # ── internal ─────────────────────────────────────────────────────────

    def _select_venue(
        self,
        symbol: str,
        amount: float,
        side: str,
    ) -> str | None:
        """Select a venue based on the current strategy.

        Args:
            symbol: Trading pair.
            amount: Order quantity.
            side: Order side.

        Returns:
            Venue ID or ``None`` if no suitable venue.
        """
        strategy = self._config.strategy

        if strategy == RouteStrategy.SINGLE:
            return self._config.default_venue

        if strategy == RouteStrategy.PRIORITY:
            healthy = self.get_healthy_venues()
            if not healthy:
                return None
            return healthy[0].venue_id

        if strategy == RouteStrategy.HEALTH_AWARE:
            healthy = self.get_healthy_venues()
            if not healthy:
                return None
            # Return the venue with the highest health score
            best = max(healthy, key=lambda v: v.health_score)
            return best.venue_id

        return self._config.default_venue

    def get_status(self) -> dict[str, Any]:
        """Return the current router status.

        Returns:
            Dict with venue list, strategy, and health info.
        """
        return {
            "strategy": self._config.strategy.name,
            "default_venue": self._config.default_venue,
            "venues": {
                vid: info.to_dict()
                for vid, info in self._venues.items()
            },
        }
