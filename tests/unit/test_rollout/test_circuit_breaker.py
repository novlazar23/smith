"""Tests for packages.rollout.circuit_breaker — open/suppress/reset/backoff behavior."""

from __future__ import annotations

import time

from packages.rollout import CircuitBreaker, CircuitState


def record_mixed(cb: CircuitBreaker, successes: int, failures: int) -> None:
    for _ in range(successes):
        cb.record_call(success=True)
    for _ in range(failures):
        cb.record_call(success=False)


class TestOpenOnErrorRate:
    def test_single_failure_opens_circuit(self) -> None:
        cb = CircuitBreaker(error_rate_threshold=0.10)
        assert cb.record_call(success=False) is False
        assert cb.state == CircuitState.OPEN

    def test_stays_closed_below_threshold(self) -> None:
        cb = CircuitBreaker(error_rate_threshold=0.10)
        record_mixed(cb, successes=19, failures=1)
        assert cb.state == CircuitState.CLOSED

    def test_error_rate_at_threshold_opens(self) -> None:
        cb = CircuitBreaker(error_rate_threshold=0.10)
        record_mixed(cb, successes=9, failures=1)
        assert cb.state == CircuitState.OPEN


class TestSuppressionWhileOpen:
    def test_calls_suppressed_during_backoff(self) -> None:
        cb = CircuitBreaker(min_backoff_seconds=3600.0)
        cb.force_open("test")
        assert cb.record_call(success=True) is False
        assert cb.state == CircuitState.OPEN
        assert cb.stats()["attempt_count"] == 0

    def test_probe_allowed_after_backoff_elapses(self) -> None:
        cb = CircuitBreaker(min_backoff_seconds=3600.0)
        cb.force_open("test")
        cb._open_at = time.monotonic() - 7200.0
        assert cb.record_call(success=True) is True
        assert cb.stats()["attempt_count"] == 1

    def test_circuit_auto_closes_after_retry(self) -> None:
        cb = CircuitBreaker(min_backoff_seconds=3600.0)
        cb.force_open("test")
        cb._open_at = time.monotonic() - 7200.0
        cb.record_call(success=True)
        assert cb.state == CircuitState.CLOSED


class TestResetAndForceOpen:
    def test_force_open(self) -> None:
        cb = CircuitBreaker()
        cb.force_open("manual")
        assert cb.state == CircuitState.OPEN
        assert cb.stats()["state"] == CircuitState.OPEN

    def test_reset_closes_and_clears_window(self) -> None:
        cb = CircuitBreaker()
        cb.record_call(success=False)
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.stats()["total_calls_in_window"] == 0
        assert cb.stats()["total_errors_in_window"] == 0
        assert cb.stats()["attempt_count"] == 0


class TestBackoff:
    def test_backoff_doubles_per_attempt(self) -> None:
        cb = CircuitBreaker(min_backoff_seconds=1.0, max_backoff_seconds=300.0)
        assert cb._backoff_seconds() == 1.0
        cb._attempt_count = 1
        assert cb._backoff_seconds() == 2.0
        cb._attempt_count = 2
        assert cb._backoff_seconds() == 4.0
        cb._attempt_count = 5
        assert cb._backoff_seconds() == 32.0

    def test_backoff_capped_at_max(self) -> None:
        cb = CircuitBreaker(min_backoff_seconds=1.0, max_backoff_seconds=300.0)
        cb._attempt_count = 30
        assert cb._backoff_seconds() == 300.0

    def test_can_retry_false_before_backoff(self) -> None:
        cb = CircuitBreaker(min_backoff_seconds=3600.0)
        cb.force_open("test")
        assert cb._can_retry(time.monotonic()) is False

    def test_can_retry_true_after_backoff(self) -> None:
        cb = CircuitBreaker(min_backoff_seconds=3600.0)
        cb.force_open("test")
        cb._open_at = time.monotonic() - 3601.0
        assert cb._can_retry(time.monotonic()) is True


class TestStats:
    def test_stats_report_window_counts(self) -> None:
        cb = CircuitBreaker(error_rate_threshold=0.10)
        record_mixed(cb, successes=19, failures=1)
        s = cb.stats()
        assert s["state"] == CircuitState.CLOSED
        assert s["error_rate"] == 0.05
        assert s["total_calls_in_window"] == 20
        assert s["total_errors_in_window"] == 1
        assert s["last_error_at"] > 0

    def test_failed_probe_reopens(self) -> None:
        cb = CircuitBreaker(min_backoff_seconds=3600.0)
        cb.record_call(success=False)
        assert cb.state == CircuitState.OPEN
        cb._calls.clear()
        cb._errors.clear()
        cb._open_at = time.monotonic() - 7200.0
        assert cb.record_call(success=False) is False
        assert cb.state == CircuitState.OPEN
        assert cb.stats()["attempt_count"] == 0
