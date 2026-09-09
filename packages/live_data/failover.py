"""Primary/backup venue failover with automatic switching and consistency checks.

This module provides:

- ``FailoverState`` — enum of the failover lifecycle states.
- ``VenueConfig`` — configuration for a single venue (primary or backup).
- ``FailoverDecision`` — outcome of a failover evaluation.
- ``FailoverManager`` — orchestrates primary/backup failover based on
  health state, with automatic switching and data consistency checks.

Usage
-----

.. code-block:: python

    manager = FailoverManager(
        venues=[
            VenueConfig(
                name="binance",
                role="primary",
                priority=1,
            ),
            VenueConfig(
                name="okx",
                role="backup",
                priority=2,
            ),
        ],
        health_threshold=0.5,
        failover_cooldown=60.0,
    )

    decision = await manager.evaluate()
    if decision.new_venue:
        print(f"Failover to {decision.new_venue}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Failover state enum
# ──────────────────────────────────────────────────────────────


class FailoverState(StrEnum):
    """Failover lifecycle states."""

    NORMAL = "normal"             # primary is healthy
    DEGRADED = "degraded"         # primary unhealthy, trying backup
    FAILED_OVER = "failed_over"   # currently using backup
    SWITTING_BACK = "switcing_back"  # switching back to primary
    ERROR = "error"               # unrecoverable error


# ──────────────────────────────────────────────────────────────
# Venue config
# ──────────────────────────────────────────────────────────────


@dataclass
class VenueConfig:
    """Configuration for a single venue in the failover topology.

    Args:
        name: Venue identifier (e.g. ``"binance"``, ``"okx"``).
        role: Either ``"primary"`` or ``"backup"``.
        priority: Lower numbers = higher priority (for ranking backups).
        max_notional: Maximum notional to route through this venue.
        tags: Optional tags for custom routing decisions.
    """

    name: str
    role: str = "primary"
    priority: int = 0
    max_notional: float = 1_000_000.0
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.role not in ("primary", "backup"):
            raise ValueError(
                f"Invalid role {self.role!r}: must be 'primary' or 'backup'"
            )

    @property
    def is_primary(self) -> bool:
        return self.role == "primary"

    @property
    def is_backup(self) -> bool:
        return self.role == "backup"


# ──────────────────────────────────────────────────────────────
# Failover decision
# ──────────────────────────────────────────────────────────────


@dataclass
class FailoverDecision:
    """Outcome of a single failover evaluation.

    Attributes:
        state: Current failover state.
        active_venue: Venue currently being used for data.
        new_venue: Venue to switch to (``None`` if no switch needed).
        reason: Human-readable reason for the decision.
        consistency_ok: Whether a data-consistency check passed
            (relevant when switching to backup).
    """

    state: FailoverState
    active_venue: str
    new_venue: str | None = None
    reason: str = ""
    consistency_ok: bool = True


# ──────────────────────────────────────────────────────────────
# FailoverManager
# ──────────────────────────────────────────────────────────────


class FailoverManager:
    """Orchestrates primary/backup failover based on health state.

    Monitors venue health and automatically switches to a backup venue
    when the primary becomes unhealthy.  Supports switching back to the
    primary once it recovers.

    Args:
        venues: List of :class:`VenueConfig` for all venues.
        health_threshold: Minimum health score for a venue to be considered
            healthy.  Defaults to 0.5.
        failover_cooldown: Minimum seconds between failover decisions.
            Defaults to 60.0.
        max_consistency_failures: Maximum allowed consistency failures
            before declaring the backup also unhealthy.
    """

    def __init__(
        self,
        venues: list[VenueConfig],
        health_threshold: float = 0.5,
        failover_cooldown: float = 60.0,
        max_consistency_failures: int = 3,
    ) -> None:
        self.venues: list[VenueConfig] = venues
        self.health_threshold = health_threshold
        self.failover_cooldown = failover_cooldown
        self.max_consistency_failures = max_consistency_failures

        self._venue_map: dict[str, VenueConfig] = {
            v.name: v for v in self.venues
        }
        self._health_scores: dict[str, float] = {
            v.name: 1.0 for v in self.venues
        }
        self._state = FailoverState.NORMAL
        primary_config = self._find_primary()
        self._active_venue: str = primary_config.name if primary_config else (self.venues[0].name if self.venues else "")
        self._last_failover_time: float = time.monotonic()
        self._consistency_failures: int = 0

    # -- properties --

    @property
    def state(self) -> FailoverState:
        return self._state

    @property
    def active_venue(self) -> str:
        return self._active_venue

    @property
    def health_scores(self) -> dict[str, float]:
        return dict(self._health_scores)

    @property
    def consistency_failures(self) -> int:
        return self._consistency_failures

    # -- public API --

    def update_health(
        self, venue: str, health_score: float
    ) -> None:
        """Update the health score for a venue.

        Args:
            venue: Venue identifier.
            health_score: Health score in ``[0.0, 1.0]``.

        Raises:
            ValueError: If the venue is not configured.
        """
        if venue not in self._venue_map:
            raise ValueError(f"Unknown venue: {venue!r}")
        self._health_scores[venue] = max(0.0, min(1.0, health_score))

    def reset_consistency_failures(self) -> None:
        """Reset the consistency failure counter (called after a successful
        data-consistency check)."""
        self._consistency_failures = 0

    async def evaluate(self) -> FailoverDecision:
        """Evaluate current health and decide whether to failover.

        Returns:
            :class:`FailoverDecision` with the current state and any
            recommended switch.
        """
        primary = self._find_primary()
        if primary is None:
            self._state = FailoverState.ERROR
            return FailoverDecision(
                state=FailoverState.ERROR,
                active_venue=self._active_venue,
                reason="No primary venue configured",
            )

        primary_score = self._health_scores.get(primary.name, 0.0)
        now = time.monotonic()

        # -- primary healthy: switch back or stay normal --
        if primary_score >= self.health_threshold:
            if (
                self._state in (FailoverState.FAILED_OVER, FailoverState.DEGRADED)
                and self._active_venue != primary.name
            ):
                if now - self._last_failover_time >= self.failover_cooldown:
                    self._state = FailoverState.SWITTING_BACK
                    return FailoverDecision(
                        state=FailoverState.NORMAL,
                        active_venue=self._active_venue,
                        new_venue=primary.name,
                        reason="Primary recovered, switch back approved",
                    )
                self._state = FailoverState.FAILED_OVER
                return FailoverDecision(
                    state=FailoverState.FAILED_OVER,
                    active_venue=self._active_venue,
                    reason="Primary recovered, switch-back cooldown active",
                )
            self._state = FailoverState.NORMAL
            return FailoverDecision(
                state=FailoverState.NORMAL,
                active_venue=self._active_venue,
                reason="Primary healthy",
            )

        # -- primary unhealthy — find best backup --
        backups = sorted(
            [v for v in self.venues if v.is_backup],
            key=lambda v: v.priority,
        )
        if not backups:
            self._state = FailoverState.DEGRADED
            return FailoverDecision(
                state=FailoverState.DEGRADED,
                active_venue=self._active_venue,
                reason=f"Primary {primary.name} unhealthy, no backup available",
            )

        # Find the first healthy backup
        best_backup: VenueConfig | None = None
        for backup in backups:
            score = self._health_scores.get(backup.name, 0.0)
            if score >= self.health_threshold:
                best_backup = backup
                break

        if best_backup is None:
            self._consistency_failures += 1
            if self._consistency_failures >= self.max_consistency_failures:
                self._state = FailoverState.ERROR
                return FailoverDecision(
                    state=FailoverState.ERROR,
                    active_venue=self._active_venue,
                    reason=(
                        "All venues unhealthy; consistency failures "
                        f"{self._consistency_failures}/{self.max_consistency_failures}"
                    ),
                )
            self._state = FailoverState.DEGRADED
            return FailoverDecision(
                state=FailoverState.DEGRADED,
                active_venue=self._active_venue,
                reason=(
                    f"Primary {primary.name} unhealthy, "
                    "no backup available yet"
                ),
            )

        # Check if already using this backup
        if self._active_venue == best_backup.name:
            self._state = FailoverState.FAILED_OVER
            return FailoverDecision(
                state=FailoverState.FAILED_OVER,
                active_venue=best_backup.name,
                reason=f"Already using backup {best_backup.name}",
                consistency_ok=True,
            )

        # -- switch to backup --
        self._state = FailoverState.DEGRADED
        self._last_failover_time = now
        return FailoverDecision(
            state=FailoverState.FAILED_OVER,
            active_venue=self._active_venue,
            new_venue=best_backup.name,
            reason=(
                f"Primary {primary.name} unhealthy "
                f"(score={primary_score:.2f}), switching to "
                f"backup {best_backup.name}"
            ),
        )

    def complete_switch(self, new_venue: str) -> None:
        """Complete a venue switch after a failover decision.

        Args:
            new_venue: The venue that has been switched to.

        Raises:
            ValueError: If the venue is not in the configured venues.
        """
        if new_venue not in self._venue_map:
            raise ValueError(f"Unknown venue: {new_venue!r}")
        self._active_venue = new_venue
        venue = self._venue_map[new_venue]
        self._state = (
            FailoverState.NORMAL if venue.is_primary else FailoverState.FAILED_OVER
        )
        self._last_failover_time = time.monotonic()

    def _find_primary(self) -> VenueConfig | None:
        """Return the configured primary venue, or ``None``."""
        primaries = [v for v in self.venues if v.is_primary]
        if not primaries:
            return None
        return min(primaries, key=lambda v: v.priority)
