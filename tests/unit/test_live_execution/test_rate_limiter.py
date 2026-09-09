"""Unit tests for packages.live_execution.rate_limiter.

Covers venue registration, token bucket consumption/refill, the acquire
success path (no sleeping), and the adaptive backoff delay using private
state instead of real time.
"""

from __future__ import annotations

import time

import pytest
from packages.live_execution.rate_limiter import (
    RateLimiter,
    RateLimitExceededError,
)


class TestVenueRegistration:
    async def test_register_creates_bucket_with_limits(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance", capacity=5, refill=2)
        state = limiter._states["binance"]
        assert state.bucket.capacity == 5
        assert state.bucket.refill_rate == 2
        assert state.bucket.tokens == 5

    async def test_register_falls_back_to_defaults(self) -> None:
        limiter = RateLimiter(default_capacity=7, default_refill=3)
        await limiter.register("bybit")
        state = limiter._states["bybit"]
        assert state.bucket.capacity == 7
        assert state.bucket.refill_rate == 3

    async def test_register_is_idempotent(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance", capacity=5)
        await limiter.register("binance", capacity=99)
        assert limiter._states["binance"].bucket.capacity == 5

    def test_set_limits_replaces_bucket(self) -> None:
        limiter = RateLimiter()
        limiter._register_sync("binance", capacity=10, refill=5)
        limiter.set_limits("binance", capacity=3, refill=1)
        state = limiter._states["binance"]
        assert state.bucket.capacity == 3
        assert state.bucket.refill_rate == 1
        assert state.bucket.tokens == 3


class TestTokenBucket:
    async def test_consume_until_empty(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance", capacity=3, refill=1)
        bucket = limiter._states["binance"].bucket
        assert bucket.try_consume() is True
        assert bucket.try_consume() is True
        assert bucket.try_consume() is True
        assert bucket.try_consume() is False

    async def test_refill_after_elapsed_time(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance", capacity=10, refill=5)
        bucket = limiter._states["binance"].bucket
        bucket.tokens = 0.0
        bucket.last_refill = time.monotonic() - 1.0
        assert bucket.try_consume() is True
        assert bucket.tokens == pytest.approx(4.0, abs=1e-2)

    async def test_refill_capped_at_capacity(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance", capacity=10, refill=5)
        bucket = limiter._states["binance"].bucket
        bucket.tokens = 0.0
        bucket.last_refill = time.monotonic() - 3600.0
        bucket._refill()
        assert bucket.tokens == 10.0


class TestAcquire:
    async def test_acquire_consumes_one_token_without_sleeping(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance", capacity=10, refill=5)
        await limiter.acquire("binance", tokens=1)
        bucket = limiter._states["binance"].bucket
        assert bucket.tokens == pytest.approx(9.0, abs=1e-6)

    async def test_acquire_auto_registers_unknown_venue(self) -> None:
        limiter = RateLimiter()
        await limiter.acquire("bybit")
        assert "bybit" in limiter._states
        assert limiter._states["bybit"].bucket.tokens == pytest.approx(9.0, abs=1e-6)


class TestAdaptiveBackoff:
    async def test_initial_delay_is_one_second(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance")
        assert limiter._states["binance"].backoff.current_delay == 1.0

    async def test_error_increases_delay(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance")
        first = await limiter.record_rate_limit_error("binance", base_delay=1.0)
        backoff = limiter._states["binance"].backoff
        assert first == pytest.approx(1.5)
        assert backoff.current_delay == pytest.approx(1.5)
        second = await limiter.record_rate_limit_error("binance", base_delay=1.0)
        assert second == pytest.approx(2.25)
        assert backoff.current_delay == pytest.approx(2.25)

    async def test_multiplier_capped_at_eight(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance")
        for _ in range(20):
            await limiter.record_rate_limit_error("binance", base_delay=1.0)
        backoff = limiter._states["binance"].backoff
        assert backoff.multiplier == 8.0
        assert backoff.current_delay == pytest.approx(8.0)

    async def test_delay_capped_at_max_delay(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance")
        delay = 1.0
        for _ in range(20):
            delay = await limiter.record_rate_limit_error("binance", base_delay=10.0)
        assert delay == 60.0
        assert limiter._states["binance"].backoff.current_delay == 60.0

    async def test_success_after_30s_decays_delay(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance")
        await limiter.record_rate_limit_error("binance", base_delay=1.0)
        backoff = limiter._states["binance"].backoff
        assert backoff.current_delay == pytest.approx(1.5)
        backoff.last_error_time = time.monotonic() - 31.0
        await limiter.record_success("binance")
        assert backoff.current_delay == pytest.approx(1.5 / 1.2)
        assert backoff.multiplier == pytest.approx(1.5 / 1.2)

    async def test_success_after_recent_error_does_not_decay(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance")
        await limiter.record_rate_limit_error("binance", base_delay=1.0)
        backoff = limiter._states["binance"].backoff
        await limiter.record_success("binance")
        assert backoff.current_delay == pytest.approx(1.5)
        assert backoff.multiplier == 1.5

    async def test_decay_floor_is_one_second(self) -> None:
        limiter = RateLimiter()
        await limiter.register("binance")
        backoff = limiter._states["binance"].backoff
        backoff.delay = 1.0
        backoff.multiplier = 1.0
        backoff.last_error_time = time.monotonic() - 31.0
        await limiter.record_success("binance")
        assert backoff.current_delay == 1.0


class TestRateLimitExceededError:
    def test_error_carries_venue(self) -> None:
        err = RateLimitExceededError("binance", retry_after=2.5)
        assert err.venue == "binance"
        assert err.retry_after == 2.5
        assert "binance" in str(err)
