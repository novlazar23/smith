"""Tests für RateLimiter."""

from __future__ import annotations

import threading

import pytest

from trading_harness.services.rate_limiter import RateLimiter


class TestRateLimiterBasic:
    """Grundlegende RateLimiter-Tests."""

    def test_allow_when_under_limit(self):
        """Erlaubt Order wenn unter Limit."""
        rl = RateLimiter(global_limit=10, symbol_limit=5)
        assert rl.allow("BTCUSDT") is True

    def test_refuses_when_global_exceeded(self):
        """Verweigert Order wenn globales Limit erreicht."""
        rl = RateLimiter(global_limit=3, symbol_limit=10)
        assert rl.allow("BTCUSDT") is True
        assert rl.allow("ETHUSDT") is True
        assert rl.allow("SOLUSDT") is True
        assert rl.allow("XRPUSDT") is False  # global limit erreicht

    def test_refuses_when_symbol_exceeded(self):
        """Verweigert Order wenn pro-Symbol Limit erreicht."""
        rl = RateLimiter(global_limit=100, symbol_limit=2)
        assert rl.allow("BTCUSDT") is True
        assert rl.allow("BTCUSDT") is True
        assert rl.allow("BTCUSDT") is False  # symbol limit erreicht
        assert rl.allow("ETHUSDT") is True  # anderes Symbol OK


class TestRateLimiterConcurrency:
    """Thread-Safety-Tests."""

    def test_concurrent_requests(self):
        """Parallele Requests sind thread-sicher."""
        rl = RateLimiter(global_limit=10, symbol_limit=10)
        results: list[bool] = []
        lock = threading.Lock()

        def make_request() -> None:
            result = rl.allow("BTCUSDT")
            with lock:
                results.append(result)

        threads = [threading.Thread(target=make_request) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Maximal global_limit viele sollten True sein
        true_count = sum(1 for r in results if r)
        assert true_count <= 10
        assert len(results) == 20


class TestRateLimiterReset:
    """Reset-Tests."""

    def test_reset_all(self):
        """Reset setzt alle Limits zurück."""
        rl = RateLimiter(global_limit=2, symbol_limit=2)
        rl.allow("BTCUSDT")
        rl.allow("ETHUSDT")
        # Jetzt sollte global limit erreicht sein
        assert rl.allow("SOLUSDT") is False
        # Reset
        rl.reset()
        # Jetzt sollte wieder erlaubt sein
        assert rl.allow("SOLUSDT") is True

    def test_reset_symbol(self):
        """Reset nur für ein Symbol."""
        rl = RateLimiter(global_limit=100, symbol_limit=2)
        rl.allow("BTCUSDT")
        rl.allow("BTCUSDT")
        assert rl.allow("BTCUSDT") is False  # BTC limit erreicht
        # Reset nur BTC
        rl.reset("BTCUSDT")
        assert rl.allow("BTCUSDT") is True
        assert rl.allow("BTCUSDT") is True
        assert rl.allow("BTCUSDT") is False  # wieder erreicht


class TestRateLimiterProperties:
    """Property-Tests."""

    def test_global_limit(self):
        assert RateLimiter(global_limit=5).global_limit == 5

    def test_symbol_limit(self):
        assert RateLimiter(symbol_limit=3).symbol_limit == 3