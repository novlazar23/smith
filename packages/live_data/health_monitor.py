"""Per-venue health monitoring with latency tracking and connection pool.

This module provides:

- ``ConnectionType`` — enum for WebSocket, REST, and REST backup connections.
- ``VenueHealthState`` — tracks health state, latency samples, and errors per
  venue per connection type.
- ``HealthMonitor`` — orchestrates per-venue health checks across all connection
  types, tracks latency percentiles, detects timeouts, and manages a connection
  pool.
- ``ConnectionPool`` — manages a pool of connections per venue with lazy
  creation, max-size limits, and health-aware eviction.

Usage
-----

.. code-block:: python

    monitor = HealthMonitor(
        venues=["binance", "okx"],
        ws_timeout=10.0,
        rest_timeout=5.0,
        pool_max_size=5,
    )

    # Periodically check health
    await monitor.check_all()

    # Inspect health state
    state = monitor.get_state("binance")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Connection type enum
# ──────────────────────────────────────────────────────────────


class ConnectionType(StrEnum):
    """Supported connection types per venue."""

    WEBSOCKET = "websocket"
    REST = "rest"
    REST_BACKUP = "rest_backup"


# ──────────────────────────────────────────────────────────────
# Latency tracker
# ──────────────────────────────────────────────────────────────


class _LatencyTracker:
    """Sliding-window latency tracker with percentile computation.

    Stores the last *window_size* samples and computes p50, p95, p99.
    """

    def __init__(self, window_size: int = 128) -> None:
        self._samples: deque[float] = deque(maxlen=window_size)

    def record(self, latency_ms: float) -> None:
        """Record a latency sample in milliseconds."""
        self._samples.append(latency_ms)

    def sample_count(self) -> int:
        return len(self._samples)

    def percentile(self, pct: float) -> float:
        """Return the given percentile (0-100), or 0.0 if no samples."""
        if not self._samples:
            return 0.0
        sorted_vals = sorted(self._samples)
        idx = int(len(sorted_vals) * pct / 100.0)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def latest(self) -> float:
        """Return the most recent sample, or 0.0 if empty."""
        if not self._samples:
            return 0.0
        return self._samples[-1]

    def average(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def clear(self) -> None:
        self._samples.clear()


# ──────────────────────────────────────────────────────────────
# Per-venue health state
# ──────────────────────────────────────────────────────────────


@dataclass
class VenueHealthState:
    """Aggregate health state for one venue across all connection types.

    Attributes:
        venue: Venue identifier (e.g. ``"binance"``).
        connection_states: Per-connection-type health info.
        overall_healthy: ``True`` only when **all** connection types are healthy.
        last_check_time: UTC timestamp of the most recent check.
        total_checks: Number of checks performed since creation.
        total_failures: Number of failed checks since creation.
    """

    venue: str
    connection_states: dict[ConnectionType, dict[str, Any]] = field(
        default_factory=dict,
    )
    overall_healthy: bool = True
    last_check_time: float = field(default_factory=time.monotonic)
    total_checks: int = 0
    total_failures: int = 0

    def mark_check(self, conn_type: ConnectionType, healthy: bool,  # noqa: FBT001
                   details: dict[str, Any] | None = None) -> None:
        """Record the result of a single health check."""
        self.total_checks += 1
        if not healthy:
            self.total_failures += 1

        entry: dict[str, Any] = {
            "healthy": healthy,
            "checked_at": time.monotonic(),
        }
        if details:
            entry["details"] = details
        self.connection_states[conn_type] = entry

        # overall_healthy is True only if every type is healthy
        self.overall_healthy = all(
            cs.get("healthy", False) for cs in self.connection_states.values()
        )
        self.last_check_time = time.monotonic()

    @property
    def failure_rate(self) -> float:
        """Ratio of failed checks (0.0-1.0). Returns 0.0 if no checks."""
        if self.total_checks == 0:
            return 0.0
        return self.total_failures / self.total_checks


# ──────────────────────────────────────────────────────────────
# Connection pool
# ──────────────────────────────────────────────────────────────


@dataclass
class _PoolEntry:
    """Single entry in the connection pool."""

    conn_id: str
    created_at: float
    last_healthy: float
    is_active: bool = True


class ConnectionPool:
    """Connection pool per venue with lazy creation and eviction.

    Creates connections on demand up to ``max_size``.  Entries that have not
    been healthy for ``stale_threshold_seconds`` are marked inactive and
    eventually evicted.

    Attributes:
        venue: Venue the pool serves.
        max_size: Maximum concurrent connections.
        stale_threshold_seconds: Seconds without a healthy ping before
            eviction is considered.
    """

    def __init__(
        self,
        venue: str,
        max_size: int = 5,
        stale_threshold_seconds: float = 300.0,
    ) -> None:
        self.venue = venue
        self.max_size = max_size
        self.stale_threshold_seconds = stale_threshold_seconds
        self._entries: dict[str, _PoolEntry] = {}
        self._lock = asyncio.Lock()

    async def create_connection(self, conn_type: ConnectionType,
                                conn_id: str | None = None) -> str:
        """Create a new connection in the pool, evicting stale entries first.

        Returns the (new or existing) connection id.
        """
        async with self._lock:
            await self._evict_stale()

            if len(self._entries) >= self.max_size:
                # Evict the least-recently-active entry
                oldest_id = min(
                    self._entries,
                    key=lambda k: self._entries[k].last_healthy,
                )
                del self._entries[oldest_id]
                logger.info(
                    "Pool for %s evicted stale connection %s to make room",
                    self.venue, oldest_id,
                )

            if conn_id is None:
                conn_id = f"{conn_type}-{time.monotonic()}"
            self._entries[conn_id] = _PoolEntry(
                conn_id=conn_id,
                created_at=time.monotonic(),
                last_healthy=time.monotonic(),
                is_active=True,
            )
            logger.debug(
                "Pool for %s created connection %s (type=%s, size=%d)",
                self.venue, conn_id, conn_type, len(self._entries),
            )
            return conn_id

    async def mark_healthy(self, conn_id: str) -> None:
        """Mark a connection as healthy (update last_healthy timestamp)."""
        async with self._lock:
            if conn_id in self._entries:
                self._entries[conn_id].last_healthy = time.monotonic()

    async def mark_inactive(self, conn_id: str) -> None:
        """Mark a connection inactive (e.g. after disconnect)."""
        async with self._lock:
            if conn_id in self._entries:
                self._entries[conn_id].is_active = False

    async def evict(self, conn_id: str) -> None:
        """Remove a connection from the pool entirely."""
        async with self._lock:
            self._entries.pop(conn_id, None)

    async def _evict_stale(self) -> None:
        """Remove inactive entries that have been stale too long."""
        now = time.monotonic()
        stale = [
            cid for cid, entry in self._entries.items()
            if not entry.is_active
            and now - entry.last_healthy > self.stale_threshold_seconds
        ]
        for cid in stale:
            del self._entries[cid]
        if stale:
            logger.info(
                "Pool for %s evicted %d stale connections",
                self.venue, len(stale),
            )

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def active_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.is_active)


# ──────────────────────────────────────────────────────────────
# HealthMonitor
# ──────────────────────────────────────────────────────────────


class HealthMonitor:
    """Per-venue health monitor with latency tracking and connection pool.

    Tracks per-connection-type health, latency percentiles, and timeout
    detection.  Integrates with :class:`ConnectionPool` for connection
    lifecycle management.

    Args:
        venues: List of venue identifiers to monitor.
        ws_timeout: WebSocket ping/pong timeout in seconds.
        rest_timeout: REST request timeout in seconds.
        pool_max_size: Maximum connections per venue in the pool.
        latency_window: Number of latency samples to keep for percentiles.
    """

    def __init__(
        self,
        venues: list[str] | None = None,
        ws_timeout: float = 10.0,
        rest_timeout: float = 5.0,
        pool_max_size: int = 5,
        latency_window: int = 128,
    ) -> None:
        self._venues: list[str] = venues or []
        self.ws_timeout = ws_timeout
        self.rest_timeout = rest_timeout
        self.pool_max_size = pool_max_size
        self.latency_window = latency_window

        self._health_states: dict[str, VenueHealthState] = {}
        self._latency_trackers: dict[str, dict[ConnectionType, _LatencyTracker]] = {}
        self._pools: dict[str, ConnectionPool] = {}

        self._check_callbacks: dict[
            ConnectionType, list[Any]
        ] = {ct: [] for ct in ConnectionType}

        for venue in self._venues:
            self._health_states[venue] = VenueHealthState(venue=venue)
            self._latency_trackers[venue] = {
                ct: _LatencyTracker(latency_window) for ct in ConnectionType
            }
            self._pools[venue] = ConnectionPool(
                venue=venue, max_size=pool_max_size,
            )

    @property
    def venues(self) -> list[str]:
        """List of monitored venues."""
        return list(self._venues)

    def register_check_callback(
        self,
        conn_type: ConnectionType,
        callback: Callable[[str, ConnectionType, bool, float], None],
    ) -> None:
        """Register a callback to be called for each health-check result.

        The callback is invoked as ``callback(venue, conn_type, healthy, latency_ms)``.
        """
        self._check_callbacks[conn_type].append(callback)

    async def check_venue(
        self,
        venue: str,
        conn_type: ConnectionType,
        latency_ms: float | None = None,
    ) -> VenueHealthState:
        """Perform a health check for one venue and one connection type.

        Args:
            venue: Venue identifier.
            conn_type: Connection type to check.
            latency_ms: Measured latency in milliseconds. If provided,
                the tracker records it.

        Returns:
            Updated :class:`VenueHealthState` for the venue.

        Raises:
            ValueError: If the venue is not registered.
        """
        if venue not in self._health_states:
            raise ValueError(f"Unknown venue: {venue!r}")

        healthy = latency_ms is not None and latency_ms < self._threshold(conn_type)
        if healthy:
            assert latency_ms is not None
            self._latency_trackers[venue][conn_type].record(latency_ms)

        self._health_states[venue].mark_check(conn_type, healthy)

        # Notify callbacks
        for cb in self._check_callbacks[conn_type]:
            try:
                cb(venue, conn_type, healthy, latency_ms or 0.0)
            except Exception:
                logger.exception("Health-check callback raised")

        return self._health_states[venue]

    async def check_all(self) -> dict[str, VenueHealthState]:
        """Run health checks on all venues (placeholder — call ``check_venue`` per type).

        Returns:
            Dict mapping venue name to its :class:`VenueHealthState`.
        """
        # ponytail: check_all liefert den Ist-Zustand ohne Ping; echte
        # Connection-Checks kommen mit den Venue-Adaptern.
        return dict(self._health_states)

    def get_state(self, venue: str) -> VenueHealthState:
        """Return the health state for a venue."""
        if venue not in self._health_states:
            raise ValueError(f"Unknown venue: {venue!r}")
        return self._health_states[venue]

    def get_pool(self, venue: str) -> ConnectionPool:
        """Return the connection pool for a venue."""
        if venue not in self._pools:
            raise ValueError(f"Unknown venue: {venue!r}")
        return self._pools[venue]

    def get_latency(self, venue: str,
                    conn_type: ConnectionType) -> _LatencyTracker:
        """Return the latency tracker for a venue + connection type."""
        if venue not in self._latency_trackers:
            raise ValueError(f"Unknown venue: {venue!r}")
        return self._latency_trackers[venue][conn_type]

    def _threshold(self, conn_type: ConnectionType) -> float:
        """Return the latency threshold in ms above which a check is unhealthy."""
        if conn_type == ConnectionType.WEBSOCKET:
            return self.ws_timeout * 1000.0
        if conn_type == ConnectionType.REST:
            return self.rest_timeout * 1000.0
        return max(self.ws_timeout, self.rest_timeout) * 1000.0

    @property
    def overall_health(self) -> dict[str, bool]:
        """Return ``{venue: is_healthy}`` for all monitored venues."""
        return {
            v: s.overall_healthy for v, s in self._health_states.items()
        }
