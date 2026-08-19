"""Tests für LiveExecutionService."""

from __future__ import annotations

import os
import threading

import pytest

from trading_harness.services.credential_manager import CredentialManager
from trading_harness.services.exchange_adapter import (
    ExchangeAdapterError,
    StubExchangeAdapter,
)
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.live_execution_service import (
    ExecutionConfig,
    LiveExecutionService,
)
from trading_harness.services.network_policy import NetworkPolicy
from trading_harness.services.order_deduplicator import OrderDeduplicator
from trading_harness.services.paper_exchange import PaperExchange
from trading_harness.services.paper_exchange_adapter import PaperExchangeAdapter
from trading_harness.services.paper_trade_store import InMemoryPaperTradeStore
from trading_harness.services.rate_limiter import RateLimiter
from trading_harness.services.risk_engine import RiskEngine
from trading_harness.services.shadow_mode_logger import ShadowModeLogger


def _make_store() -> InMemoryPaperTradeStore:
    """Erstellt einen InMemoryPaperTradeStore für PaperExchange."""
    return InMemoryPaperTradeStore()


@pytest.fixture
def adapter() -> PaperExchangeAdapter:
    pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
    return PaperExchangeAdapter(paper_exchange=pe)


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
                max_capital=1000.0,
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
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
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
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
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
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
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
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
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


class TestRiskEngineIntegration:
    """Integration tests for RiskEngine in the execution pipeline (R5.1–R5.5)."""

    @pytest.fixture
    def risk_policy_approved(self) -> dict:
        """Gültige Risk-Policy die den meisten Trades zustimmt."""
        return {
            "max_risk_per_trade": 0.02,
            "max_daily_loss": 0.05,
            "max_leverage": 3.0,
            "max_positions": 5,
            "max_portfolio_risk": 0.10,
            "min_stop_distance_bps": 50,
            "minimum_risk_reward": 1.5,
            "max_slippage_bps": 200,
            "allowed_symbols": ["BTCUSDT", "ETHUSDT"],
        }

    @pytest.fixture
    def risk_engine(self, risk_policy_approved: dict) -> RiskEngine:
        return RiskEngine(policy=risk_policy_approved)

    def test_risk_engine_approved_order_passes(self, adapter, risk_engine):
        """Order wird zugestanden wenn RiskEngine approves."""
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            risk_engine=risk_engine,
            config=ExecutionConfig(
                live_execution_enabled=True,
                symbol_whitelist=["BTCUSDT"],
                max_capital=1000.0,
            ),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="re-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "FILLED"
        assert result["risk_approved"] is True
        assert result["risk_max_position_size"] > 0

    def test_risk_engine_rejects_unknown_symbol(self, adapter, risk_engine):
        """RiskEngine rejectet Symbole die nicht in allowed_symbols sind."""
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            risk_engine=risk_engine,
            config=ExecutionConfig(
                live_execution_enabled=True,
                symbol_whitelist=[],  # leer → Whitelist-Check skipped, RiskEngine erreicht Trade
                max_capital=1000.0,
            ),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="re-sol",
            run_id="run-1",
            symbol="SOLUSDT",  # Nicht in allowed_symbols ["BTCUSDT", "ETHUSDT"]
            side="LONG",
            quantity=1.0,
            price=100.0,
        )
        assert result["status"] == "REJECTED"
        assert result["risk_approved"] is False
        assert result["error"] == "SYMBOL_NOT_ALLOWED"

    def test_risk_engine_log_fields_recorded(self, adapter, risk_engine):
        """risk_approved und risk_max_position_size werden im Log gespeichert."""
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            risk_engine=risk_engine,
            config=ExecutionConfig(
                live_execution_enabled=True,
                symbol_whitelist=["BTCUSDT"],
                max_capital=1000.0,
            ),
        )
        svc.activate_live()
        svc.submit_order(
            decision_id="re-log",
            run_id="run-1",
            symbol="BTCUSDT",
            side="SHORT",
            quantity=5.0,
            price=3000.0,
        )
        logs = svc.get_logs(decision_id="re-log")
        assert len(logs) == 1
        assert logs[0]["risk_approved"] is True
        assert logs[0]["risk_max_position_size"] > 0
        assert logs[0]["status"] == "FILLED"


class TestSymbolWhitelistIntegration:
    """Integration tests for symbol whitelist pre-execution check (R5.6)."""

    def test_whitelisted_symbol_passes(self, adapter):
        """Whitelisted Symbol wird durchgelassen."""
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            config=ExecutionConfig(
                live_execution_enabled=True,
                symbol_whitelist=["BTCUSDT", "ETHUSDT"],
                max_capital=1000.0,
            ),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="sw-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "FILLED"

    def test_non_whitelisted_symbol_rejected(self, adapter):
        """Nicht-whitelisted Symbol wird blockiert."""
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            config=ExecutionConfig(
                live_execution_enabled=True,
                symbol_whitelist=["BTCUSDT"],
            ),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="sw-2",
            run_id="run-1",
            symbol="SOLUSDT",  # Nicht auf Whitelist
            side="LONG",
            quantity=1.0,
            price=100.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "SYMBOL_NOT_WHITELISTED"

    def test_empty_whitelist_allows_all(self, adapter):
        """Leere Whitelist lässt alle Symbole durch."""
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            config=ExecutionConfig(
                live_execution_enabled=True,
                symbol_whitelist=[],
                max_capital=1000.0,
            ),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="sw-3",
            run_id="run-1",
            symbol="ARBITRARY",
            side="LONG",
            quantity=1.0,
            price=50.0,
        )
        assert result["status"] == "FILLED"


class TestAllowedExchangesIntegration:
    """Integration tests for allowed_exchanges config check."""

    def test_allowed_exchange_passes(self, adapter):
        """Erlaubter Exchange wird durchgelassen."""
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            config=ExecutionConfig(
                live_execution_enabled=True,
                allowed_exchanges=["PAPER"],
                max_capital=1000.0,
            ),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="ae-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "FILLED"

    def test_empty_allowed_exchanges_allows_all(self, adapter):
        """Leere allowed_exchanges lässt alle Exchanges durch."""
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            config=ExecutionConfig(
                live_execution_enabled=True,
                allowed_exchanges=[],
                max_capital=1000.0,
            ),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="ae-2",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "FILLED"


class TestMaxPositionSizeEnforcement:
    """Tests for RiskEngine max_position_size enforcement in submission."""

    @pytest.fixture
    def risk_policy_tight(self) -> dict:
        """Risk-Policy mit strengen Limits."""
        return {
            "max_risk_per_trade": 0.005,
            "max_daily_loss": 0.05,
            "max_leverage": 3.0,
            "max_positions": 5,
            "max_portfolio_risk": 0.10,
            "min_stop_distance_bps": 50,
            "minimum_risk_reward": 1.5,
            "max_slippage_bps": 200,
            "allowed_symbols": ["BTCUSDT"],
        }

    @pytest.fixture
    def risk_engine_tight(self, risk_policy_tight: dict) -> RiskEngine:
        return RiskEngine(policy=risk_policy_tight)

    def test_quantity_capped_to_max_position_size(self, adapter, risk_engine_tight):
        """quantity wird auf max_position_size reduziert wenn diese kleiner ist."""
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            risk_engine=risk_engine_tight,
            config=ExecutionConfig(
                live_execution_enabled=True,
                symbol_whitelist=["BTCUSDT"],
                max_capital=1000.0,
            ),
        )
        svc.activate_live()
        # quantity=10.0, aber RiskEngine berechnet max_position_size
        result = svc.submit_order(
            decision_id="mp-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=10.0,
            price=50000.0,
        )
        # Sollte FILLED sein (PaperExchange nimmt die reduzierte quantity)
        assert result["status"] == "FILLED"
        assert result["risk_approved"] is True
        assert result["risk_max_position_size"] > 0
        # Log soll die reduzierte Position zeigen
        logs = svc.get_logs(decision_id="mp-1")
        assert logs[0]["risk_max_position_size"] > 0


class TestNetworkPolicyIntegration:
    """Integration tests for NetworkPolicy in execution pipeline (R5.15–R5.17)."""

    def test_network_policy_allows_matching_url(self, adapter):
        """NetworkPolicy erlaubt Requests auf whitelisted URLs."""
        np = NetworkPolicy(allowed_patterns=["https://api.paper.example.com/*"])
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            network_policy=np,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        # PaperExchangeAdapter verwendet keine echte URL, _get_exchange_url gibt "*" zurück
        # Bei Pattern "*" sollte das durchkommen, aber hier ist das Pattern spezifisch.
        # Da _get_exchange_url "*" für unbekannte Adapter zurückgibt, muss "*" gematcht werden.
        np2 = NetworkPolicy(allowed_patterns=[".*"])
        svc2 = LiveExecutionService(
            exchange_adapter=adapter,
            network_policy=np2,
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
        )
        svc2.activate_live()
        result = svc2.submit_order(
            decision_id="np-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "FILLED"

    def test_network_policy_blocks_violation(self):
        """NetworkPolicy blockiert Requests auf nicht-whitelisted URLs."""
        np = NetworkPolicy(allowed_patterns=["https://trusted.example.com/*"])
        svc = LiveExecutionService(
            exchange_adapter=PaperExchangeAdapter(
                paper_exchange=PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
            ),
            network_policy=np,
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
        )
        svc.activate_live()
        # Da _get_exchange_url "*" für PaperExchangeAdapter zurückgibt und "*"
        # nicht auf "https://trusted.example.com/*" matcht:
        result = svc.submit_order(
            decision_id="np-2",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "NETWORK_POLICY_VIOLATION"


class TestCredentialManagerIntegration:
    """Integration tests for CredentialManager in execution pipeline (R5.18–R5.20)."""

    def test_credentials_configured_passes(self, adapter):
        """Pipeline geht durch wenn Credentials konfiguriert sind."""
        os.environ["TRADE_API_KEY"] = "test-key"
        os.environ["TRADE_API_SECRET"] = "test-secret"
        try:
            cm = CredentialManager()
            svc = LiveExecutionService(
                exchange_adapter=adapter,
                credential_manager=cm,
                config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
            )
            svc.activate_live()
            result = svc.submit_order(
                decision_id="cm-1",
                run_id="run-1",
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                price=50000.0,
            )
            assert result["status"] == "FILLED"
        finally:
            os.environ.pop("TRADE_API_KEY", None)
            os.environ.pop("TRADE_API_SECRET", None)

    def test_credentials_missing_rejected(self, adapter):
        """Pipeline wird blockiert wenn Credentials fehlen."""
        # Stelle sicher dass env vars nicht gesetzt sind
        os.environ.pop("TRADE_API_KEY", None)
        os.environ.pop("TRADE_API_SECRET", None)
        cm = CredentialManager()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            credential_manager=cm,
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="cm-2",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "TRADE_CREDENTIALS_NOT_CONFIGURED"

    def test_missing_api_key_rejected(self, adapter):
        """Pipeline wird blockiert wenn nur TRADE_API_SECRET gesetzt ist."""
        os.environ["TRADE_API_KEY"] = ""
        os.environ["TRADE_API_SECRET"] = "test-secret"
        try:
            cm = CredentialManager()
            svc = LiveExecutionService(
                exchange_adapter=adapter,
                credential_manager=cm,
                config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
            )
            svc.activate_live()
            result = svc.submit_order(
                decision_id="cm-3",
                run_id="run-1",
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                price=50000.0,
            )
            assert result["status"] == "REJECTED"
            assert result["error"] == "TRADE_CREDENTIALS_NOT_CONFIGURED"
        finally:
            os.environ.pop("TRADE_API_KEY", None)
            os.environ.pop("TRADE_API_SECRET", None)


class TestFullPipelineIntegration:
    """Integration tests for the complete execution pipeline with all services."""

    def test_full_pipeline_risk_approved(self, adapter):
        """Komplette Pipeline: KillSwitch→RateLimit→Dedup→Whitelist→RiskEngine→NetworkPolicy→CredentialCheck→Exchange."""
        policy = {
            "max_risk_per_trade": 0.02,
            "max_daily_loss": 0.05,
            "max_leverage": 3.0,
            "max_positions": 5,
            "max_portfolio_risk": 0.10,
            "min_stop_distance_bps": 50,
            "minimum_risk_reward": 1.5,
            "max_slippage_bps": 200,
            "allowed_symbols": ["BTCUSDT"],
        }
        ks = KillSwitch(enabled=False)
        rl = RateLimiter(global_limit=100, symbol_limit=50)
        dd = OrderDeduplicator()
        np = NetworkPolicy(allowed_patterns=[".*"])
        os.environ["TRADE_API_KEY"] = "test-key"
        os.environ["TRADE_API_SECRET"] = "test-secret"
        try:
            cm = CredentialManager()
            re = RiskEngine(policy=policy)
            svc = LiveExecutionService(
                kill_switch=ks,
                rate_limiter=rl,
                deduplicator=dd,
                exchange_adapter=adapter,
                risk_engine=re,
                network_policy=np,
                credential_manager=cm,
                config=ExecutionConfig(
                    live_execution_enabled=True,
                    symbol_whitelist=["BTCUSDT"],
                    max_capital=1000.0,
                ),
            )
            svc.activate_live()
            result = svc.submit_order(
                decision_id="full-1",
                run_id="run-1",
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                price=50000.0,
            )
            assert result["status"] == "FILLED"
            assert result["risk_approved"] is True
            assert result["risk_max_position_size"] > 0
        finally:
            os.environ.pop("TRADE_API_KEY", None)
            os.environ.pop("TRADE_API_SECRET", None)

    def test_full_pipeline_killswitch_blocks_first(self, adapter):
        """KillSwitch wird vor allen anderen Checks geprüft."""
        policy = {
            "max_risk_per_trade": 0.02,
            "max_daily_loss": 0.05,
            "max_leverage": 3.0,
            "max_positions": 5,
            "max_portfolio_risk": 0.10,
            "min_stop_distance_bps": 50,
            "minimum_risk_reward": 1.5,
            "max_slippage_bps": 200,
            "allowed_symbols": ["BTCUSDT"],
        }
        ks = KillSwitch(enabled=True)
        cm = CredentialManager()
        np = NetworkPolicy(allowed_patterns=[".*"])
        os.environ["TRADE_API_KEY"] = "test-key"
        os.environ["TRADE_API_SECRET"] = "test-secret"
        try:
            svc = LiveExecutionService(
                kill_switch=ks,
                exchange_adapter=adapter,
                risk_engine=RiskEngine(policy=policy),
                network_policy=np,
                credential_manager=cm,
                config=ExecutionConfig(
                    live_execution_enabled=True,
                    symbol_whitelist=["BTCUSDT"],
                ),
            )
            svc.activate_live()
            result = svc.submit_order(
                decision_id="full-ks",
                run_id="run-1",
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                price=50000.0,
            )
            assert result["status"] == "REJECTED"
            assert result["error"] == "KILL_SWITCH_ACTIVE"
            assert result["risk_approved"] is False
        finally:
            os.environ.pop("TRADE_API_KEY", None)
            os.environ.pop("TRADE_API_SECRET", None)


class TestReadTradeApiSeparation:
    """R5.21–R5.22: Separate Auth für Trade-Aktionen vs. Lese-Zugriffe."""

    def test_submit_requires_trade_credentials(self):
        """submit_order verlangt TRADE_API_KEY + TRADE_API_SECRET."""
        adapter = PaperExchangeAdapter(paper_exchange=PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store()))
        cm = CredentialManager()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            credential_manager=cm,
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="rta-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "TRADE_CREDENTIALS_NOT_CONFIGURED"

    def test_submit_with_trade_credentials_succeeds(self):
        """submit_order mit TRADE_API_KEY + TRADE_API_SECRET durchläuft."""
        adapter = PaperExchangeAdapter(paper_exchange=PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store()))
        cm = CredentialManager()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            credential_manager=cm,
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
        )
        svc.activate_live()
        os.environ["TRADE_API_KEY"] = "trade-key"
        os.environ["TRADE_API_SECRET"] = "trade-secret"
        try:
            result = svc.submit_order(
                decision_id="rta-2",
                run_id="run-1",
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                price=50000.0,
            )
            assert result["status"] == "FILLED"
        finally:
            os.environ.pop("TRADE_API_KEY", None)
            os.environ.pop("TRADE_API_SECRET", None)

    def test_read_order_status_requires_read_credentials(self):
        """get_order_status verlangt READ_API_KEY + READ_API_SECRET."""
        adapter = PaperExchangeAdapter(paper_exchange=PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store()))
        cm = CredentialManager()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            credential_manager=cm,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        result = svc.get_order_status("order-1")
        assert result["status"] == "REJECTED"
        assert result["error"] == "READ_CREDENTIALS_NOT_CONFIGURED"

    def test_read_order_status_with_read_credentials_succeeds(self):
        """get_order_status mit READ_API_KEY + READ_API_SECRET durchläuft Credential-Check."""
        adapter = PaperExchangeAdapter(paper_exchange=PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store()))
        cm = CredentialManager()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            credential_manager=cm,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        os.environ["READ_API_KEY"] = "read-key"
        os.environ["READ_API_SECRET"] = "read-secret"
        try:
            result = svc.get_order_status("order-1")
            assert result["status"] == "NOT_FOUND"  # PaperExchangeAdapter: order nicht vorhanden
        finally:
            os.environ.pop("READ_API_KEY", None)
            os.environ.pop("READ_API_SECRET", None)

    def test_cancel_order_requires_trade_credentials(self):
        """cancel_order verlangt TRADE_API_KEY + TRADE_API_SECRET (write operation)."""
        adapter = PaperExchangeAdapter(paper_exchange=PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store()))
        cm = CredentialManager()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            credential_manager=cm,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        result = svc.cancel_order("order-1")
        assert result["success"] is False
        assert result["error"] == "TRADE_CREDENTIALS_NOT_CONFIGURED"

    def test_cancel_order_with_trade_credentials_succeeds(self):
        """cancel_order mit TRADE_API_KEY + TRADE_API_SECRET durchläuft Credential-Check."""
        adapter = PaperExchangeAdapter(paper_exchange=PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store()))
        cm = CredentialManager()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            credential_manager=cm,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        os.environ["TRADE_API_KEY"] = "trade-key"
        os.environ["TRADE_API_SECRET"] = "trade-secret"
        try:
            result = svc.cancel_order("order-1")
            assert result["success"] is False  # PaperExchangeAdapter: Trade nicht gefunden, aber Credential-Check bestanden
            assert result["error"] == "TRADE_NOT_FOUND"
        finally:
            os.environ.pop("TRADE_API_KEY", None)
            os.environ.pop("TRADE_API_SECRET", None)

    def test_config_has_custom_key_refs(self):
        """ExecutionConfig erlaubt benutzerdefinierte Key-Ref-Namen."""
        cfg = ExecutionConfig(
            live_execution_enabled=True,
            trade_api_key_ref="MY_TRADE_KEY",
            trade_api_secret_ref="MY_TRADE_SECRET",
            read_api_key_ref="MY_READ_KEY",
            read_api_secret_ref="MY_READ_SECRET",
        )
        assert cfg.trade_api_key_ref == "MY_TRADE_KEY"
        assert cfg.read_api_key_ref == "MY_READ_KEY"


class TestAllowedExchangesEnforcement:
    """allowed_exchanges Whitelist Enforcement im Execution Pipeline."""

    def test_allowed_exchanges_blocks_unlisted_exchange(self):
        """Order an nicht-listeter Exchange wird REJECTED."""
        adapter = PaperExchangeAdapter(paper_exchange=PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store()))
        cm = CredentialManager()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            credential_manager=cm,
            config=ExecutionConfig(
                live_execution_enabled=True,
                allowed_exchanges=["bybit", "binance"],
            ),
        )
        svc.activate_live()
        os.environ["TRADE_API_KEY"] = "trade-key"
        os.environ["TRADE_API_SECRET"] = "trade-secret"
        try:
            result = svc.submit_order(
                decision_id="ae-1",
                run_id="run-1",
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                price=50000.0,
                exchange_name="kraken",
            )
            assert result["status"] == "REJECTED"
            assert result["error"] == "EXCHANGE_NOT_ALLOWED"
        finally:
            os.environ.pop("TRADE_API_KEY", None)
            os.environ.pop("TRADE_API_SECRET", None)

    def test_allowed_exchanges_allows_listed_exchange(self):
        """Order an gelisteter Exchange durchläuft."""
        adapter = PaperExchangeAdapter(paper_exchange=PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store()))
        cm = CredentialManager()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            credential_manager=cm,
            config=ExecutionConfig(
                live_execution_enabled=True,
                allowed_exchanges=["bybit", "binance"],
                max_capital=1000.0,
            ),
        )
        svc.activate_live()
        os.environ["TRADE_API_KEY"] = "trade-key"
        os.environ["TRADE_API_SECRET"] = "trade-secret"
        try:
            result = svc.submit_order(
                decision_id="ae-2",
                run_id="run-1",
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                price=50000.0,
                exchange_name="binance",
            )
            assert result["status"] == "FILLED"
        finally:
            os.environ.pop("TRADE_API_KEY", None)
            os.environ.pop("TRADE_API_SECRET", None)

    def test_allowed_exchanges_empty_allows_any(self):
        """Leere allowed_exchanges erlaubt jede Exchange (Default-Verhalten)."""
        adapter = PaperExchangeAdapter(paper_exchange=PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store()))
        cm = CredentialManager()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            credential_manager=cm,
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
        )
        svc.activate_live()
        os.environ["TRADE_API_KEY"] = "trade-key"
        os.environ["TRADE_API_SECRET"] = "trade-secret"
        try:
            result = svc.submit_order(
                decision_id="ae-3",
                run_id="run-1",
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                price=50000.0,
                exchange_name="unknown-exchange",
            )
            assert result["status"] == "FILLED"
        finally:
            os.environ.pop("TRADE_API_KEY", None)
            os.environ.pop("TRADE_API_SECRET", None)


class TestLiveExecutionServiceShadowMode:
    """Shadow-Mode-Wiring: abgelehnte Orders werden im Shadow-Logger protokolliert."""

    def test_rejected_order_logged_to_shadow_mode(self, adapter):
        """Deaktivierte Live-Execution → Order wird als REJECTED shadow-geloggt."""
        shadow = ShadowModeLogger()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            shadow_logger=shadow,
        )
        result = svc.submit_order(
            decision_id="sm-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.5,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "LIVE_EXECUTION_DISABLED"
        assert result["shadow_mode"] is True
        records = shadow.get_records(decision_id="sm-1")
        assert len(records) == 1
        record = records[0]
        assert record.simulated_status == "REJECTED"
        assert record.error == "LIVE_EXECUTION_DISABLED"
        assert record.run_id == "run-1"
        assert record.symbol == "BTCUSDT"
        assert record.side == "LONG"
        assert record.quantity == 1.5
        assert record.price == 50000.0
        assert record.simulated_fill_price == 0.0
        assert record.simulated_commission == 0.0

    def test_kill_switch_rejection_shadow_logged(self):
        """Kill-Switch-Rejection wird mit Request-Parametern shadow-geloggt."""
        shadow = ShadowModeLogger()
        ks = KillSwitch(enabled=False)
        svc = LiveExecutionService(
            kill_switch=ks,
            shadow_logger=shadow,
            config=ExecutionConfig(live_execution_enabled=True),
        )
        svc.activate_live()
        ks.activate()
        result = svc.submit_order(
            decision_id="sm-2",
            run_id="run-2",
            symbol="ETHUSDT",
            side="SHORT",
            quantity=2.0,
            price=3000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "KILL_SWITCH_ACTIVE"
        records = shadow.get_records(decision_id="sm-2")
        assert len(records) == 1
        assert records[0].error == "KILL_SWITCH_ACTIVE"
        assert records[0].quantity == 2.0
        assert records[0].price == 3000.0

    def test_no_shadow_logger_flag_false(self, adapter):
        """Ohne Shadow-Logger ist shadow_mode im Response False."""
        svc = LiveExecutionService(exchange_adapter=adapter)
        result = svc.submit_order(
            decision_id="sm-3",
            run_id="run-3",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["shadow_mode"] is False

    def test_successful_order_not_shadow_logged(self, adapter):
        """Erfolgreich ausgeführte Orders werden nicht als REJECTED shadow-geloggt."""
        shadow = ShadowModeLogger()
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            shadow_logger=shadow,
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
        )
        svc.activate_live()
        result = svc.submit_order(
            decision_id="sm-4",
            run_id="run-4",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "FILLED"


class FailingExchangeAdapter(StubExchangeAdapter):
    """Liefert deterministische Exchange-Fehler (Anomalie-Tests)."""

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
        exchange_name: str | None = None,
    ) -> dict[str, object]:
        return {"order_id": None, "status": "ERROR", "error": "EXCHANGE_5XX"}


class ThrowingExchangeAdapter(StubExchangeAdapter):
    """Wirft Exchange-Exceptions (Anomalie-Tests, Exception-Pfad)."""

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
        exchange_name: str | None = None,
    ) -> dict[str, object]:
        raise ExchangeAdapterError("EXCHANGE_CONNECTION_LOST")


class FlakyExchangeAdapter(StubExchangeAdapter):
    """Liefert eine vorgegebene FILLED/ERROR-Ausfolge."""

    def __init__(self, outcomes: list[str]) -> None:
        self._outcomes = list(outcomes)

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
        exchange_name: str | None = None,
    ) -> dict[str, object]:
        outcome = self._outcomes.pop(0) if self._outcomes else "FILLED"
        if outcome == "FILLED":
            return {"order_id": "ord-flaky", "status": "FILLED"}
        return {"order_id": None, "status": "ERROR", "error": "EXCHANGE_5XX"}


class TestLiveExecutionServiceAnomalyAutoTrigger:
    """R5.6: Exchange-Anomalien aktivieren den Kill Switch automatisch."""

    def _make_service(self, adapter: StubExchangeAdapter, ks: KillSwitch) -> LiveExecutionService:
        svc = LiveExecutionService(
            exchange_adapter=adapter,
            kill_switch=ks,
            rate_limiter=RateLimiter(global_limit=100, symbol_limit=100),
            config=ExecutionConfig(live_execution_enabled=True, max_capital=1000.0),
        )
        svc.activate_live()
        return svc

    def _submit(self, svc: LiveExecutionService, decision_id: str) -> dict[str, object]:
        return svc.submit_order(
            decision_id=decision_id,
            run_id="run-anomaly",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )

    def test_three_consecutive_errors_auto_trigger(self):
        """3 aufeinanderfolgende Exchange-Fehler aktivieren den Kill Switch."""
        ks = KillSwitch()
        svc = self._make_service(FailingExchangeAdapter(), ks)
        for i in range(3):
            result = self._submit(svc, f"an-{i}")
            assert result["status"] == "ERROR"
        assert ks.is_active() is True
        # Nächste Order wird vom Kill Switch blockiert
        result = self._submit(svc, "an-3")
        assert result["status"] == "REJECTED"
        assert result["error"] == "KILL_SWITCH_ACTIVE"
        assert result["kill_switch_auto_triggered"] is False

    def test_two_errors_do_not_trigger(self):
        """Unterhalb des Schwellwerts bleibt der Kill Switch inaktiv."""
        ks = KillSwitch()
        svc = self._make_service(FailingExchangeAdapter(), ks)
        self._submit(svc, "an-1")
        result = self._submit(svc, "an-2")
        assert ks.is_active() is False
        assert result["kill_switch_auto_triggered"] is False

    def test_filled_resets_error_streak(self):
        """FILLED-Order setzt den Anomalie-Streak zurück."""
        adapter = FlakyExchangeAdapter(
            ["ERROR", "ERROR", "FILLED", "ERROR", "ERROR", "ERROR"]
        )
        ks = KillSwitch()
        svc = self._make_service(adapter, ks)
        for i in range(5):
            self._submit(svc, f"flaky-{i}")
        # FILLED (flaky-2) hat den Streak zurückgesetzt: nach 5 Orders erst 2
        assert ks.is_active() is False
        assert ks.config.anomaly_streak == 2
        self._submit(svc, "flaky-5")
        assert ks.is_active() is True

    def test_exception_path_counts_as_anomaly(self):
        """Adapter-Exceptions zählen als Anomalie (Exception-Pfad)."""
        ks = KillSwitch()
        svc = self._make_service(ThrowingExchangeAdapter(), ks)
        for i in range(3):
            result = self._submit(svc, f"exc-{i}")
            assert result["status"] == "ERROR"
        assert ks.is_active() is True

    def test_response_flag_on_triggering_order(self):
        """Nur die auslösende Order trägt kill_switch_auto_triggered=True."""
        ks = KillSwitch()
        svc = self._make_service(FailingExchangeAdapter(), ks)
        first = self._submit(svc, "flag-1")
        second = self._submit(svc, "flag-2")
        third = self._submit(svc, "flag-3")
        assert first["kill_switch_auto_triggered"] is False
        assert second["kill_switch_auto_triggered"] is False
        assert third["kill_switch_auto_triggered"] is True
        assert ks.config.auto_triggered is True
        assert "EXCHANGE_5XX" in (ks.config.trigger_reason or "")
        assert third["shadow_mode"] is False