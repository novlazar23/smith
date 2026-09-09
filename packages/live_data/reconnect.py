"""Auto-reconnect with exponential backoff + jitter and state recovery.

This module provides:

- ``ReconnectConfig`` — configuration for backoff behaviour (max attempts,
  base delay, max delay, jitter factor).
- ``ReconnectEvent`` — records of individual reconnect attempts.
- ``ReconnectState`` — enum of the reconnector lifecycle state.
- ``AutoReconnector`` — manages reconnection with exponential backoff + jitter,
  tracks attempt history, and supports state recovery callbacks.

Usage
-----

.. code-block:: python

    config = ReconnectConfig(
        max_attempts=20,
        base_delay=1.0,
        max_delay=60.0,
        jitter_factor=0.5,
    )

    reconnector = AutoReconnector(
        venue="binance",
        config=config,
        on_reconnect=restore_state,
        on_state_change=notify_ops,
    )

    await reconnector.run()
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Reconnect state enum
# ──────────────────────────────────────────────────────────────


class ReconnectState(StrEnum):
    """Lifecycle states of the reconnector."""

    IDLE = "idle"              # not running
    CONNECTING = "connecting"  # current attempt in progress
    RECONNECTING = "reconnecting"  # between attempts, backoff active
    CONNECTED = "connected"    # successfully reconnected
    FAILED = "failed"          # max attempts reached


# ──────────────────────────────────────────────────────────────
# Reconnect event
# ──────────────────────────────────────────────────────────────


@dataclass
class ReconnectEvent:
    """Record of a single reconnect attempt."""

    timestamp: float
    attempt: int
    delay_before: float
    success: bool
    error: str | None = None

    @property
    def iso_timestamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))


# ──────────────────────────────────────────────────────────────
# Reconnect config
# ──────────────────────────────────────────────────────────────


@dataclass
class ReconnectConfig:
    """Configuration for auto-reconnect behaviour.

    Args:
        max_attempts: Maximum reconnect attempts (0 = unlimited).
        base_delay: Base delay in seconds before the first attempt.
        max_delay: Cap on the delay between attempts.
        jitter_factor: Randomisation factor — the delay is multiplied by
            ``[1 - jitter_factor, 1 + jitter_factor]``.  Use ``0.0`` for
            no jitter, ``0.5`` for ±50 %.
        on_reconnect: Optional callback invoked just before each connect.
            Signature: ``(venue: str, attempt: int, delay: float) -> None``.
        on_state_change: Optional callback invoked on every state transition.
            Signature: ``(state: ReconnectState, venue: str) -> None``.
    """

    max_attempts: int = 20
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.5
    on_reconnect: Callable[[str, int, float], Any] | None = None
    on_state_change: Callable[[ReconnectState, str], Any] | None = None


# ──────────────────────────────────────────────────────────────
# AutoReconnector
# ──────────────────────────────────────────────────────────────


class AutoReconnector:
    """Auto-reconnect with exponential backoff + jitter and state recovery.

    After each failed connection attempt the delay grows exponentially
    (``base_delay * 2^(attempt-1)``) capped at ``max_delay``.  Each delay
    is further multiplied by a jitter factor drawn from
    ``[1 - jitter_factor, 1 + jitter_factor]``.

    The reconnector tracks attempt history in :attr:`events` and supports
    an optional ``on_reconnect`` callback that can be used to restore state
    (e.g. re-subscribe to streams) before each new connection attempt.

    Args:
        venue: Venue identifier.
        config: Reconnect configuration.
        connect_hook: Async callable that performs the actual reconnection.
            Must set ``self._connected`` to ``True`` on success.
    """

    def __init__(
        self,
        venue: str,
        config: ReconnectConfig,
        connect_hook: Callable[[], Any] | None = None,
    ) -> None:
        self.venue = venue
        self.config = config
        self.connect_hook = connect_hook or self._default_connect

        self._state = ReconnectState.IDLE
        self._connected = False
        self._running = False
        self._stop_event: asyncio.Event | None = None
        self.events: list[ReconnectEvent] = []
        self._attempt_counter: int = 0

    # -- properties --

    @property
    def state(self) -> ReconnectState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def success_count(self) -> int:
        return sum(1 for e in self.events if e.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for e in self.events if not e.success)

    @property
    def last_delay(self) -> float:
        if self.events:
            return self.events[-1].delay_before
        return 0.0

    # -- public API --

    async def run(self) -> bool:
        """Start the reconnection loop.

        Returns:
            ``True`` if eventually connected, ``False`` if all attempts failed.
        """
        self._running = True
        self._stop_event = asyncio.Event()
        self._connected = False
        self.events.clear()
        self._attempt_counter = 0

        await self._change_state(ReconnectState.CONNECTING)

        while self._running:
            attempt = self._attempt_counter + 1

            # Check max attempts
            if self.config.max_attempts > 0 and attempt > self.config.max_attempts:
                logger.error(
                    "Venue %s: max reconnect attempts (%d) reached",
                    self.venue, self.config.max_attempts,
                )
                await self._change_state(ReconnectState.FAILED)
                return False

            # Compute delay with exponential backoff + jitter
            raw_delay = self.config.base_delay * (2 ** (attempt - 1))
            raw_delay = min(raw_delay, self.config.max_delay)
            jitter = 1.0 + random.uniform(
                -self.config.jitter_factor,
                self.config.jitter_factor,
            )
            delay = raw_delay * jitter

            event = ReconnectEvent(
                timestamp=time.monotonic(),
                attempt=attempt,
                delay_before=delay,
                success=False,
            )

            logger.info(
                "Venue %s: reconnect attempt %d in %.2fs (raw=%.2f)",
                self.venue, attempt, delay, raw_delay,
            )

            # Notify state change
            await self._change_state(ReconnectState.RECONNECTING)
            await asyncio.sleep(delay)

            if not self._running:
                break

            # Invoke reconnect hook
            if self.config.on_reconnect:
                try:
                    self.config.on_reconnect(self.venue, attempt, delay)
                except Exception:
                    logger.exception("on_reconnect callback raised")

            # Execute connect hook
            try:
                result = self.connect_hook()
                if asyncio.iscoroutine(result):
                    result = await result

                if result is True:
                    self._connected = True
                    event.success = True
                    logger.info("Venue %s: reconnected after %d attempts",
                                self.venue, attempt)
                    await self._change_state(ReconnectState.CONNECTED)
                    self._attempt_counter = attempt
                    return True
            except Exception as exc:
                event.error = str(exc)
                logger.warning("Venue %s: reconnect attempt %d failed: %s",
                               self.venue, attempt, exc)
            finally:
                self.events.append(event)
                self._attempt_counter = attempt

        return False

    async def stop(self) -> None:
        """Stop the reconnection loop."""
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        if self._state != ReconnectState.IDLE:
            await self._change_state(ReconnectState.IDLE)

    async def reset(self) -> None:
        """Reset state for a fresh reconnection cycle."""
        self._connected = False
        self.events.clear()
        self._attempt_counter = 0
        await self._change_state(ReconnectState.IDLE)

    # -- internal --

    async def _change_state(self, new_state: ReconnectState) -> None:
        self._state = new_state
        if self.config.on_state_change:
            try:
                self.config.on_state_change(new_state, self.venue)
            except Exception:
                logger.exception("on_state_change callback raised")

    @staticmethod
    async def _default_connect() -> bool:
        raise NotImplementedError(
            "connect_hook must be provided or subclassed"
        )

    @property
    def config(self) -> ReconnectConfig:
        return self._config

    @config.setter
    def config(self, value: ReconnectConfig) -> None:
        self._config = value
