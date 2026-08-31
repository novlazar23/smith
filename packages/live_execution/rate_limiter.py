"""Token-bucket rate limiter per venue with adaptive backoff.

Each venue gets its own token bucket with a configurable capacity and refill
rate.  When tokens are exhausted the caller receives an ``awaitable`` that
completes once enough tokens have refilled.

Adaptive backoff
----------------
If the exchange returns a ``429 Too Many Requests`` or a rate-limit error,
the limiter automatically increases the backoff delay (multiplier x 1.5) up
to a configurable maximum.  A slow decay (/ 1.2 every 30 s) brings the
multiplier back to ``1.0`` when the venue is healthy again.

Design
------
- One token bucket per venue.
- Thread-safe via ``asyncio.Lock``.
- Supports ``acquire(venue, tokens=1)`` which blocks until tokens are
  available.

Usage
-----

.. code-block:: python

    limiter = RateLimiter(default_capacity=10, default_refill=5)

    # Default venue (first registered)
    await limiter.acquire("binance")

    # Custom venue-specific limits
    limiter.set_limits("bybit", capacity=5, refill=2)
    await limiter.acquire("bybit")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─── Exceptions ─────────────────────────────────────────────────────────────


class RateLimitExceededError(Exception):
    """Raised when rate limiting cannot be satisfied within the timeout."""

    def __init__(self, venue: str, retry_after: float | None = None) -> None:
        self.venue = venue
        self.retry_after = retry_after
        msg = f"Rate limit exceeded for venue: {venue}"
        if retry_after is not None:
            msg += f". Retry after {retry_after}s"
        super().__init__(msg)


# ─── Token Bucket ───────────────────────────────────────────────────────────


@dataclass
class _TokenBucket:
    """Internal token bucket implementation.

    Attributes:
        capacity: Maximum number of tokens.
        refill_rate: Tokens added per second.
        tokens: Current available tokens.
        last_refill: Unix timestamp of last refill.
    """

    capacity: float
    refill_rate: float
    tokens: float = field(init=False, default=0.0)
    last_refill: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed <= 0:
            return
        added = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + added)
        self.last_refill = now

    def try_consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens without blocking.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            ``True`` if tokens were consumed, ``False`` otherwise.
        """
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_for_tokens(
        self,
        tokens: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        """Block until enough tokens are available.

        Args:
            tokens: Number of tokens needed.
            timeout: Maximum seconds to wait.

        Raises:
            RateLimitExceededError: If timeout is exceeded.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            # Calculate how long until enough tokens refill
            deficit = tokens - self.tokens
            wait_time = deficit / self.refill_rate if self.refill_rate > 0 else 1.0
            remaining = deadline - time.monotonic()
            sleep_time = min(wait_time, remaining, 0.1)
            if sleep_time > 0:
                time.sleep(sleep_time)

        raise RateLimitExceededError(
            venue="<adaptive>",
            retry_after=round(tokens - self.tokens, 2),
        )


# ─── Adaptive Backoff ──────────────────────────────────────────────────────


@dataclass
class _AdaptiveBackoff:
    """Exponential backoff state with slow decay.

    Attributes:
        multiplier: Current backoff multiplier (starts at 1.0).
        last_error_time: When the last rate-limit error occurred.
    """

    multiplier: float = 1.0
    last_error_time: float = field(default=0.0, init=False)

    def on_error(self, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Called when a rate-limit error occurs. Returns the backoff delay.

        Args:
            base_delay: Base delay in seconds before applying multiplier.
            max_delay: Maximum allowed backoff delay.

        Returns:
            The computed backoff delay.
        """
        self.multiplier = min(self.multiplier * 1.5, 8.0)
        self.last_error_time = time.monotonic()
        delay = min(base_delay * self.multiplier, max_delay)
        logger.info(
            "Rate-limit error — backoff multiplier=%.1f, delay=%.2fs",
            self.multiplier,
            delay,
        )
        return delay

    def on_success(self) -> None:
        """Called when a successful request completes.

        Slowly decays the multiplier back to 1.0.
        """
        if time.monotonic() - self.last_error_time > 30:
            self.multiplier = max(self.multiplier / 1.2, 1.0)

    @property
    def current_delay(self) -> float:
        """Current backoff multiplier."""
        return self.multiplier


# ─── Per-Venue State ────────────────────────────────────────────────────────


@dataclass
class _VenueState:
    """State for a single venue's rate limiter.

    Attributes:
        bucket: Token bucket for this venue.
        backoff: Adaptive backoff tracker.
    """

    bucket: _TokenBucket
    backoff: _AdaptiveBackoff = field(default_factory=_AdaptiveBackoff)


# ─── Rate Limiter ───────────────────────────────────────────────────────────


class RateLimiter:
    """Token-bucket rate limiter with adaptive backoff.

    Manages one bucket per venue.  All venues share the same default
    capacity and refill rate unless overridden via :meth:`set_limits`.

    Args:
        default_capacity: Default token bucket capacity.
        default_refill: Default token refill rate (tokens/second).
        default_timeout: Default acquire timeout in seconds.
    """

    def __init__(
        self,
        default_capacity: int = 10,
        default_refill: int = 5,
        default_timeout: float = 30.0,
    ) -> None:
        self._default_capacity = default_capacity
        self._default_refill = default_refill
        self._default_timeout = default_timeout
        self._states: dict[str, _VenueState] = {}
        self._lock = asyncio.Lock()

    # ── venue management ─────────────────────────────────────────────────

    async def register(
        self,
        venue: str,
        capacity: int | None = None,
        refill: int | None = None,
    ) -> None:
        """Register a venue with custom limits.

        Args:
            venue: Venue identifier (e.g. ``"binance"``).
            capacity: Token bucket capacity.  Falls back to
                ``default_capacity`` if ``None``.
            refill: Token refill rate.  Falls back to
                ``default_refill`` if ``None``.
        """
        async with self._lock:
            if venue in self._states:
                return
            cap = capacity if capacity is not None else self._default_capacity
            ref = refill if refill is not None else self._default_refill
            bucket = _TokenBucket(capacity=cap, refill_rate=ref)
            self._states[venue] = _VenueState(bucket=bucket)
            logger.info(
                "RateLimiter registered venue=%s capacity=%d refill=%d",
                venue,
                cap,
                ref,
            )

    def _register_sync(
        self,
        venue: str,
        capacity: int,
        refill: int,
    ) -> None:
        """Synchronous venue registration (no lock needed).

        Used by the gateway constructor which runs synchronously.
        """
        if venue in self._states:
            return
        bucket = _TokenBucket(capacity=capacity, refill_rate=refill)
        self._states[venue] = _VenueState(bucket=bucket)

    def set_limits(
        self,
        venue: str,
        capacity: int,
        refill: int,
    ) -> None:
        """Change limits for an already-registered venue.

        Args:
            venue: Venue identifier.
            capacity: New token bucket capacity.
            refill: New token refill rate.
        """
        # Synchronous convenience — state is already in-memory
        state = self._states.get(venue)
        if state is not None:
            state.bucket = _TokenBucket(
                capacity=capacity,
                refill_rate=refill,
            )
            logger.info(
                "RateLimiter updated venue=%s capacity=%d refill=%d",
                venue,
                capacity,
                refill,
            )

    # ── acquire / release ────────────────────────────────────────────────

    async def acquire(
        self,
        venue: str,
        tokens: int = 1,
        timeout: float | None = None,
    ) -> None:
        """Acquire tokens for a venue, blocking if necessary.

        Args:
            venue: Venue identifier.
            tokens: Number of tokens to acquire.
            timeout: Maximum seconds to wait.  Falls back to
                ``default_timeout`` if ``None``.

        Raises:
            RateLimitExceededError: If tokens cannot be acquired within timeout.
        """
        timeout = timeout if timeout is not None else self._default_timeout

        async with self._lock:
            state = self._states.get(venue)
            if state is None:
                # Auto-register with defaults
                await self.register(venue)
                state = self._states[venue]

        # Check for active backoff before trying
        delay = state.backoff.current_delay
        if delay > 1.0:
            logger.info(
                "Venue %s is under backoff (%.1fx), waiting %.2fs",
                venue,
                delay,
                delay,
            )
            await asyncio.sleep(delay)

        # Try non-blocking first
        if state.bucket.try_consume(tokens):
            state.backoff.on_success()
            return

        # Wait for tokens
        await asyncio.wait_for(
            asyncio.to_thread(
                state.bucket.wait_for_tokens, tokens=tokens, timeout=timeout,
            ),
            timeout=timeout,
        )
        state.backoff.on_success()

    async def record_rate_limit_error(self, venue: str, base_delay: float = 1.0) -> float:
        """Record that a rate-limit error occurred.

        Increases the adaptive backoff multiplier.  Call this when the
        exchange returns ``429`` or a rate-limit HTTP error.

        Args:
            venue: Venue identifier.
            base_delay: Base backoff delay in seconds.

        Returns:
            The computed backoff delay.
        """
        async with self._lock:
            state = self._states.get(venue)
            if state is None:
                await self.register(venue)
                state = self._states[venue]

        delay = state.backoff.on_error(base_delay=base_delay)
        logger.warning(
            "Rate-limit error recorded for venue=%s, backoff=%.2fs",
            venue,
            delay,
        )
        return delay

    async def record_success(self, venue: str) -> None:
        """Record a successful request.

        Slowly decays the backoff multiplier.  Call this for every
        non-error response.

        Args:
            venue: Venue identifier.
        """
        async with self._lock:
            state = self._states.get(venue)
            if state is not None:
                state.backoff.on_success()
