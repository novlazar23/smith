"""Idempotency keys for duplicate order-submit prevention.

Every order submission MUST include an idempotency key.  When a duplicate
request (same key) arrives, the gateway returns the original result instead
of placing a second order on the venue.

Design
------
- In-memory LRU-style store keyed by ``(venue, idempotency_key)``.
- Entries expire after a configurable TTL (default 24 h) to avoid stale
  entries from cancelled orders.
- Thread-safe via a single ``asyncio.Lock``.

Usage
-----

.. code-block:: python

    store = IdempotencyStore(ttl_seconds=3600)

    # First submit — stores the result
    result = await store.record(
        idempotency_key="uuid-1",
        venue="binance",
        result={"order_id": "12345", "status": "pending"},
    )
    # → returns {"order_id": "12345", "status": "pending"}

    # Duplicate submit — returns cached result
    result = await store.record(
        idempotency_key="uuid-1",
        venue="binance",
        result={"order_id": "12345", "status": "pending"},
    )
    # → returns the same cached result
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


# ─── Stored Entry ───────────────────────────────────────────────────────────


@dataclass
class IdempotencyEntry:
    """A cached result keyed by idempotency key.

    Attributes:
        key: Unique idempotency identifier.
        venue: Venue identifier.
        result: The original submission result.
        created_at: When this entry was created.
        expires_at: When this entry should be discarded.
    """

    key: str
    venue: str
    result: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(default=None)  # type: ignore[assignment]

    def is_expired(self) -> bool:
        """Return ``True`` if this entry has passed its TTL."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at


# ─── Idempotency Error ──────────────────────────────────────────────────────


class IdempotencyError(Exception):
    """Raised when a duplicate idempotency key is detected."""

    def __init__(self, key: str, cached_result: dict[str, Any]) -> None:
        super().__init__(
            f"Duplicate idempotency key: {key}. "
            f"Cached result: {cached_result}"
        )
        self.key = key
        self.cached_result = cached_result


# ─── Store ──────────────────────────────────────────────────────────────────


class IdempotencyStore:
    """In-memory idempotency key store with TTL.

    Args:
        ttl_seconds: Time-to-live for each entry in seconds.
    """

    def __init__(self, ttl_seconds: int = 86400) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._store: dict[tuple[str, str], IdempotencyEntry] = {}
        self._lock = asyncio.Lock()

    # ── public API ───────────────────────────────────────────────────────

    async def record(
        self,
        idempotency_key: str,
        venue: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Record a submission result under an idempotency key.

        If a result already exists for ``(venue, idempotency_key)`` and is
        not expired, returns the cached result **without raising**.

        Args:
            idempotency_key: Unique per-submission identifier (UUID recommended).
            venue: Venue identifier.
            result: The original submission result dict.

        Returns:
            The result dict (either cached or newly recorded).
        """
        composite_key = (venue, idempotency_key)

        async with self._lock:
            # Expire stale entries
            self._cleanup_expired()

            if composite_key in self._store:
                entry = self._store[composite_key]
                logger.info(
                    "Idempotency HIT: key=%s venue=%s", idempotency_key, venue,
                )
                return entry.result

            entry = IdempotencyEntry(
                key=idempotency_key,
                venue=venue,
                result=dict(result),
                expires_at=datetime.now(UTC) + self._ttl,
            )
            self._store[composite_key] = entry
            logger.info(
                "Idempotency RECORD: key=%s venue=%s", idempotency_key, venue,
            )
            return entry.result

    async def get(
        self, idempotency_key: str, venue: str
    ) -> dict[str, Any] | None:
        """Retrieve a cached result, or ``None`` if not found / expired.

        Args:
            idempotency_key: The idempotency key to look up.
            venue: Venue identifier.

        Returns:
            The cached result or ``None``.
        """
        composite_key = (venue, idempotency_key)

        async with self._lock:
            entry = self._store.get(composite_key)
            if entry is not None and not entry.is_expired():
                return entry.result
            return None

    async def exists(
        self, idempotency_key: str, venue: str
    ) -> bool:
        """Check if an idempotency key exists and is not expired.

        Args:
            idempotency_key: The idempotency key to check.
            venue: Venue identifier.

        Returns:
            ``True`` if a valid entry exists.
        """
        return await self.get(idempotency_key, venue) is not None

    async def generate_key(self, venue: str) -> str:
        """Generate a fresh UUID idempotency key.

        Args:
            venue: Venue identifier (for logging).

        Returns:
            A new UUID v4 string.
        """
        key = str(uuid.uuid4())
        logger.info("Generated idempotency key: %s for venue=%s", key, venue)
        return key

    async def clear(self) -> None:
        """Remove all stored entries."""
        async with self._lock:
            self._store.clear()
            logger.info("Idempotency store cleared")

    # ── internal ─────────────────────────────────────────────────────────

    def _cleanup_expired(self) -> None:
        """Remove expired entries (must be called with lock held)."""
        now = datetime.now(UTC)
        expired_keys = [
            k for k, v in self._store.items()
            if v.expires_at is not None and now > v.expires_at
        ]
        for k in expired_keys:
            del self._store[k]
        if expired_keys:
            logger.debug("Cleaned up %d expired idempotency entries", len(expired_keys))
