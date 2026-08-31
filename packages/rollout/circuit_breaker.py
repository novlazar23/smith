"""Circuit breaker for exchange connectivity and error-rate monitoring.

Implements a sliding-window error-rate counter that opens a circuit
breaker when the configured error rate exceeds ``max_exchange_error_rate``.
While the circuit is open all exchange calls are suppressed and retries
use exponential backoff.  The circuit requires manual reset after opening.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState:
    """Possible states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"


@dataclass
class CircuitBreaker:
    """Circuit breaker protecting against exchange errors.

    Parameters
    ----------
    error_rate_threshold :
        Fraction of calls that may fail before the circuit opens.
    window_seconds :
        Sliding window over which the error rate is computed.
    min_backoff_seconds :
        Minimum wait time before a retry attempt.
    max_backoff_seconds :
        Maximum wait time between retry attempts.
    max_retries :
        Number of retry attempts before marking the circuit as persistently open.
    """

    error_rate_threshold: float = 0.10
    window_seconds: float = 300.0
    min_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 300.0
    max_retries: int = 10

    _calls: list[float] = field(default_factory=list, init=False)
    _errors: list[float] = field(default_factory=list, init=False)
    _state: str = field(default=CircuitState.CLOSED, init=False)
    _last_error_time: float = field(default=0.0, init=False)
    _attempt_count: int = field(default=0, init=False)
    _open_at: float = field(default=0.0, init=False)

    # ── Public state ──

    @property
    def state(self) -> str:
        """Current circuit breaker state ('closed' or 'open')."""
        # Auto-transition from open → closed after max_backoff
        if self._state == CircuitState.OPEN and self._has_retried():
            self._state = CircuitState.CLOSED
            self._attempt_count = 0
            self._clear_window()
            logger.info("circuit_breaker: circuit closed after backoff")
        return self._state

    # ── Core API ──

    def record_call(self, success: bool) -> bool:
        """Record a single exchange call result.

        Returns *True* if the call is allowed to proceed (circuit closed
        or attempt within backoff budget), *False* if the call is
        suppressed by the open circuit.
        """
        now = time.monotonic()

        self._calls.append(now)
        self._prune_window(now)

        # ── While open: suppress unless we are in a retry window ──
        if self._state == CircuitState.OPEN:
            if not self._can_retry(now):
                logger.warning(
                    "circuit_breaker: call suppressed (state=open, backoff active)"
                )
                return False
            # Allow one probe call per retry window
            self._attempt_count += 1

        if success:
            return True

        # ── Failure: track and potentially open ──
        self._errors.append(now)
        self._last_error_time = now

        error_rate = self._current_error_rate()
        if error_rate >= self.error_rate_threshold:
            self._open(now)
            return False

        return True

    def force_open(self, reason: str = "manual") -> None:
        """Immediately open the circuit breaker (manual or automatic trigger)."""
        self._open(time.monotonic())
        logger.warning("circuit_breaker: circuit forced open — reason=%s", reason)

    def reset(self) -> None:
        """Manually reset the circuit breaker to a closed state."""
        self._state = CircuitState.CLOSED
        self._attempt_count = 0
        self._clear_window()
        logger.info("circuit_breaker: manual reset performed")

    # ── Diagnostics ──

    def stats(self) -> dict:
        """Return diagnostic statistics about the circuit breaker."""
        return {
            "state": self._state,
            "error_rate": round(self._current_error_rate(), 4),
            "total_calls_in_window": len(self._calls),
            "total_errors_in_window": len(self._errors),
            "attempt_count": self._attempt_count,
            "last_error_at": self._last_error_time,
        }

    # ── Internal helpers ──

    def _open(self, now: float) -> None:
        self._state = CircuitState.OPEN
        self._open_at = now
        self._attempt_count = 0
        logger.error(
            "circuit_breaker: circuit opened (error_rate=%.2f > threshold=%.2f)",
            self._current_error_rate(),
            self.error_rate_threshold,
        )

    def _prune_window(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._calls = [t for t in self._calls if t > cutoff]
        self._errors = [t for t in self._errors if t > cutoff]

    def _clear_window(self) -> None:
        self._calls.clear()
        self._errors.clear()

    def _current_error_rate(self) -> float:
        total = len(self._calls)
        if total == 0:
            return 0.0
        return len(self._errors) / total

    def _backoff_seconds(self) -> float:
        """Calculate exponential backoff duration in seconds."""
        exponent = min(self._attempt_count, 20)  # cap to avoid overflow
        value = self.min_backoff_seconds * (2 ** exponent)
        return min(value, self.max_backoff_seconds)

    def _can_retry(self, now: float) -> bool:
        """Return True if enough time has passed for the next retry attempt."""
        elapsed = now - self._open_at
        required = self._backoff_seconds()
        return elapsed >= required

    def _has_retried(self) -> bool:
        """Return True if we have attempted a retry (attempt count > 0)."""
        return self._attempt_count > 0
