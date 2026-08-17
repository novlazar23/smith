"""Tests für LiveExecutionService."""

from __future__ import annotations

import threading

from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.live_execution_service import (
    ExecutionConfig,
    LiveExecutionService,
)
from trading_harness.services.order_deduplicator import OrderDeduplicator
from trading_harness.services.paper_exchange import PaperExchange
from trading_harness.services.paper_exchange_adapter import PaperExchangeAdapter
from trading_harness.services.paper_trade_store import InMemoryPaperTradeStore
from trading_harness.services.rate_limiter import RateLimiter


def _make_store() -> InMemoryPaperTradeStore:
    """Erstellt einen InMemoryPaperTradeStore für PaperExchange."""
    return InMemoryPaperTradeStore()


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


class TestLiveExecutionServicePaperPipeline:
    """Integration tests for PaperExchangeAdapter in the execution pipeline."""

    def _make_adapter_with_fill_rate(self, fill_rate: float = 1.0) -> PaperExchangeAdapter:
        pe = PaperExchange(fill_rate=fill_rate, fee_rate=0.0, stores=_make_store())
        return PaperExchangeAdapter(paper_exchange=pe)

    def test_paper_adapter_full_pipeline_submit(self):
        """PaperExchangeAdapter durchläuft die komplette Pipeline erfolgreich."""
        adapter = self._make_adapter_with_fill_rate()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="paper-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "FILLED"
        assert result["order_id"] is not None
        assert result["error"] is None
        assert result["symbol"] == "BTCUSDT"
        assert result["side"] == "LONG"

    def test_paper_adapter_short_order(self):
        """Short-Order durchläuft die Pipeline."""
        adapter = self._make_adapter_with_fill_rate()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="paper-2",
            run_id="run-1",
            symbol="ETHUSDT",
            side="SHORT",
            quantity=10.0,
            price=3000.0,
        )
        assert result["status"] == "FILLED"
        assert result["order_id"] is not None

    def test_paper_adapter_rejected_with_kill_switch(self):
        """Kill Switch blockiert Paper-Orders."""
        adapter = self._make_adapter_with_fill_rate()
        ks = KillSwitch(enabled=False)
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            kill_switch=ks,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        ks.activate()
        result = svc.submit_order(
            decision_id="paper-3",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "KILL_SWITCH_ACTIVE"

    def test_paper_adapter_rejected_with_rate_limit(self):
        """Rate Limit blockiert zweite Paper-Order."""
        rl = RateLimiter(global_limit=1, symbol_limit=10)
        adapter = self._make_adapter_with_fill_rate()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            rate_limiter=rl,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        svc.submit_order(
            decision_id="paper-4a",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        result = svc.submit_order(
            decision_id="paper-4b",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "RATE_LIMIT_EXCEEDED"

    def test_paper_adapter_duplicate_decision_rejected(self):
        """Dopplte decision_id wird vom Deduplicator blockiert."""
        dd = OrderDeduplicator()
        adapter = self._make_adapter_with_fill_rate()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            deduplicator=dd,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        svc.submit_order(
            decision_id="paper-5",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        result = svc.submit_order(
            decision_id="paper-5",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "DUPLICATE_DECISION_ID"

    def test_paper_adapter_min_capital_enforced(self):
        """Min Capital blockiert Paper-Orders unter dem Limit."""
        adapter = self._make_adapter_with_fill_rate()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            config=ExecutionConfig(
                live_execution_enabled=True,
                min_capital=1.0,
            ),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="paper-6",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "MIN_CAPITAL_NOT_MET"

    def test_paper_adapter_log_recorded(self):
        """Successful Paper-Order wird geloggt."""
        adapter = self._make_adapter_with_fill_rate()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        svc.submit_order(
            decision_id="paper-7",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        logs = svc.get_logs(decision_id="paper-7")
        assert len(logs) == 1
        assert logs[0]["status"] == "FILLED"
        assert logs[0]["symbol"] == "BTCUSDT"

    def test_paper_adapter_disabled_rejected(self):
        """Deaktivierte Pipeline lehnt Paper-Orders ab."""
        adapter = self._make_adapter_with_fill_rate()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            config=ExecutionConfig(live_execution_enabled=False),
        )
        result = svc.submit_order(
            decision_id="paper-8",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "LIVE_EXECUTION_DISABLED"

    def test_paper_adapter_multiple_independent_symbols(self):
        """Unabhängige Symbole durchlaufen die Pipeline."""
        rl = RateLimiter(global_limit=100, symbol_limit=10)
        adapter = self._make_adapter_with_fill_rate()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            rate_limiter=rl,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        result_btc = svc.submit_order(
            decision_id="paper-9a",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        result_eth = svc.submit_order(
            decision_id="paper-9b",
            run_id="run-1",
            symbol="ETHUSDT",
            side="SHORT",
            quantity=10.0,
            price=3000.0,
        )
        assert result_btc["status"] == "FILLED"
        assert result_eth["status"] == "FILLED"

    def test_paper_adapter_side_normalization_in_log(self):
        """Side wird im Log als LONG/SHORT gespeichert."""
        adapter = self._make_adapter_with_fill_rate()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        svc.submit_order(
            decision_id="paper-10",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        logs = svc.get_logs(decision_id="paper-10")
        assert logs[0]["side"] == "LONG"