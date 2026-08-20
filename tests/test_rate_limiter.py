"""Tests für RateLimiter."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from trading_harness.services.rate_limiter import RateLimiter


class _FakeClock:
    """Deterministische Zeitquelle für time.monotonic-Tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _patch_monotonic(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Ersetzt time.monotonic im rate_limiter-Modul durch eine Fake-Clock."""
    clock = _FakeClock()
    monkeypatch.setattr(
        "trading_harness.services.rate_limiter.time",
        SimpleNamespace(monotonic=clock),
    )
    return clock


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

    def test_symbol_limit_concurrent(self):
        """Parallele Requests auf ein Symbol respektieren das Pro-Symbol-Limit."""
        rl = RateLimiter(global_limit=100, symbol_limit=2)
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

        # Maximal symbol_limit viele sollten True sein
        true_count = sum(1 for r in results if r)
        assert true_count <= 2
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


class TestRateLimiterTimeSemantics:
    """Dokumentierte N/min-Semantik (Spec R5.10, Epic WI-P5-2).

    Die Refill-Rate muss mit dem Limit skalieren: global_limit/60 bzw.
    symbol_limit/60 Tokens pro Sekunde, Kapazität = Limit (Burst).
    Review-9 B2: Refill=1 Token/s unabhängig vom Limit ließ 70/min statt
    10/min durch und verstieß gegen die dokumentierte Semantik.
    """

    def test_refill_scales_with_limit(self, monkeypatch):
        """10/min => Refill 10/60 Tokens/s: nach 6 s exakt 1 Token."""
        clock = _patch_monotonic(monkeypatch)
        rl = RateLimiter(global_limit=10, symbol_limit=100)
        for _ in range(10):
            assert rl.allow("BTCUSDT") is True
        clock.advance(6.0)
        rl._refill()
        # 6 s * 10/60 Tokens/s = 1 Token (Refill=1 Token/s ergäbe 6)
        assert rl._global_tokens == pytest.approx(1.0, abs=1e-9)
        # 6.0 * (10/60) liegt 1 ulp unter 1.0; kleiner Schubs zur Freigabe
        clock.advance(0.01)
        assert rl.allow("BTCUSDT") is True
        assert rl.allow("BTCUSDT") is False

    def test_symbol_refill_scales_with_limit(self, monkeypatch):
        """2/min pro Symbol => Refill 2/60 Tokens/s: nach 30 s exakt 1 Token."""
        clock = _patch_monotonic(monkeypatch)
        rl = RateLimiter(global_limit=100, symbol_limit=2)
        assert rl.allow("BTCUSDT") is True
        assert rl.allow("BTCUSDT") is True
        assert rl.allow("BTCUSDT") is False
        clock.advance(30.0)
        rl._refill()
        # 30 s * 2/60 Tokens/s = 1 Token (Refill=1 Token/s ergäbe 2, die Kapazität)
        assert rl._symbol_tokens["BTCUSDT"] == pytest.approx(1.0, abs=1e-9)
        clock.advance(0.01)
        assert rl.allow("BTCUSDT") is True
        assert rl.allow("BTCUSDT") is False

    def test_refill_does_not_exceed_capacity(self, monkeypatch):
        """Refill bleibt bei Kapazität = Limit (Burst-Semantik unverändert)."""
        clock = _patch_monotonic(monkeypatch)
        rl = RateLimiter(global_limit=10, symbol_limit=100)
        clock.advance(120.0)
        rl._refill()
        # 2 min * 10/min = 20 Tokens, aber Kapazität = 10
        assert rl._global_tokens == pytest.approx(10.0, abs=1e-9)
        for _ in range(10):
            assert rl.allow("BTCUSDT") is True
        assert rl.allow("BTCUSDT") is False

    def test_sustained_throughput_per_minute(self, monkeypatch):
        """Sustained-Rate = 10/min: 60 s => burst(10) + ~10 weitere Orders.

        Alte Semantik (Refill 1 Token/s) hätte hier 70 Orders/60 s
        durchgelassen; die dokumentierte N/min-Semantik genau ~20.
        """
        clock = _patch_monotonic(monkeypatch)
        rl = RateLimiter(global_limit=10, symbol_limit=100)
        allowed = 0
        for _ in range(10):
            if rl.allow("BTCUSDT"):
                allowed += 1
        for _ in range(60):
            clock.advance(1.0)
            if rl.allow("BTCUSDT"):
                allowed += 1
        # Ideal 20 (10 Burst + 10 Refill); Float-Grenzen am Token-Schwellwert
        # erlauben ±1
        assert 19 <= allowed <= 21

    def test_symbol_sustained_rate_per_minute(self, monkeypatch):
        """Pro-Symbol Sustained-Rate = 2/min: 60 s => burst(2) + ~2 weitere."""
        clock = _patch_monotonic(monkeypatch)
        rl = RateLimiter(global_limit=100, symbol_limit=2)
        allowed = 0
        for _ in range(2):
            if rl.allow("BTCUSDT"):
                allowed += 1
        for _ in range(60):
            clock.advance(1.0)
            if rl.allow("BTCUSDT"):
                allowed += 1
        # Ideal 4 (2 Burst + 2 Refill); Float-Grenzen erlauben ±1
        assert 3 <= allowed <= 5