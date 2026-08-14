"""Tests für LiveExecutionService."""

from __future__ import annotations

import threading

import pytest

from trading_harness.services.exchange_adapter import StubExchangeAdapter
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.live_execution_service import (
    ExecutionConfig,
    LiveExecutionService,
)
from trading_harness.services.order_deduplicator import OrderDeduplicator
from trading_harness.services.rate_limiter import RateLimiter


class TestLiveExecutionServiceBasic:
    """Grundlegende LiveExecutionService-Tests."""

    def test_default_disabled(self):
        """Standardmäßig deaktiviert."""
        svc = LiveExecutionService()
        assert svc.is_live_enabled is False

    def test_submit_when_disabled(self):
        """Submit gibt REJECTED wenn deaktiviert."""
        svc = LiveExecutionService()
        result = svc.submit_order(
            decision_id="dec-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "LIVE_EXECUTION_DISABLED"

    def test_activate(self):
        """Activate setzt live_enabled auf True."""
        svc = LiveExecutionService()
        svc.activate_live()
        assert svc.is_live_enabled is True

    def test_deactivate(self):
        """Deactivate setzt live_enabled auf False."""
        svc = LiveExecutionService()
        svc.activate_live()
        svc.deactivate_live()
        assert svc.is_live_enabled is False


class TestLiveExecutionServiceKillSwitch:
    """KillSwitch-Integration."""

    def test_kill_switch_blocks_order(self):
        """Kill Switch blockiert Orders auch wenn live_enabled."""
        ks = KillSwitch(enabled=False)
        svc = LiveExecutionService(
            kill_switch=ks,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        ks.activate()  # Kill Switch aktivieren
        result = svc.submit_order(
            decision_id="dec-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "KILL_SWITCH_ACTIVE"


class TestLiveExecutionServiceRateLimit:
    """RateLimit-Integration."""

    def test_rate_limit_blocks_order(self):
        """Rate Limit blockiert Orders."""
        rl = RateLimiter(global_limit=1, symbol_limit=10)
        svc = LiveExecutionService(
            rate_limiter=rl,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        svc.submit_order(
            decision_id="dec-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        # Zweite Order sollte blockiert werden
        result = svc.submit_order(
            decision_id="dec-2",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "RATE_LIMIT_EXCEEDED"


class TestLiveExecutionServiceDedup:
    """Dedup-Integration."""

    def test_duplicate_rejected(self):
        """Duplikate werden blockiert."""
        dd = OrderDeduplicator()
        svc = LiveExecutionService(
            deduplicator=dd,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        svc.submit_order(
            decision_id="dec-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        # Gleiche Order sollte blockiert werden
        result = svc.submit_order(
            decision_id="dec-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "DUPLICATE_DECISION_ID"


class TestLiveExecutionServiceMinCapital:
    """Min Capital-Tests."""

    def test_min_capital_enforced(self):
        """Min Capital wird enforced."""
        svc = LiveExecutionService(
            config=ExecutionConfig(
                live_execution_enabled=True,
                min_capital=1.0,
            ),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="dec-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,  # unter min_capital
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "MIN_CAPITAL_NOT_MET"

    def test_min_capital_passes(self):
        """Min Capital wird erreicht — geht an Exchange (stub gibt ERROR)."""
        svc = LiveExecutionService(
            config=ExecutionConfig(
                live_execution_enabled=True,
                min_capital=0.5,
            ),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="dec-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,  # über min_capital
            price=50000.0,
        )
        # Stub adapter gibt NOT_IMPLEMENTED zurück -> wird zu ERROR
        assert result["status"] == "ERROR"
        assert "NO_EXCHANGE_ADAPTER_IMPLEMENTED" in (result.get("error") or "")


class TestLiveExecutionServiceConcurrency:
    """Thread-Safety-Tests."""

    def test_concurrent_submit(self):
        """Parallele Calls sind thread-sicher."""
        svc = LiveExecutionService(
            config=ExecutionConfig(live_execution_enabled=False),
        )
        results: list[dict] = []
        lock = threading.Lock()

        def submit() -> None:
            result = svc.submit_order(
                decision_id="dec-1",
                run_id="run-1",
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                price=50000.0,
            )
            with lock:
                results.append(result)

        threads = [threading.Thread(target=submit) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Alle sollten REJECTED sein (deaktiviert)
        assert all(r["status"] == "REJECTED" for r in results)
        assert len(results) == 50


class TestLiveExecutionServiceLogs:
    """Log-Tests."""

    def test_logs_recorded(self):
        """Logs werden aufgezeichnet."""
        svc = LiveExecutionService(
            config=ExecutionConfig(live_execution_enabled=False),
        )
        svc.submit_order(
            decision_id="dec-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        logs = svc.get_logs()
        assert len(logs) == 1
        assert logs[0]["decision_id"] == "dec-1"

    def test_logs_filter_by_decision_id(self):
        """Logs können nach decision_id gefiltert werden."""
        svc = LiveExecutionService(
            config=ExecutionConfig(live_execution_enabled=False),
        )
        svc.submit_order(
            decision_id="dec-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        svc.submit_order(
            decision_id="dec-2",
            run_id="run-1",
            symbol="ETHUSDT",
            side="SHORT",
            quantity=1.0,
            price=3000.0,
        )
        logs = svc.get_logs(decision_id="dec-1")
        assert len(logs) == 1
        assert logs[0]["symbol"] == "BTCUSDT"