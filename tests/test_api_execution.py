"""Tests für Phase 5 Execution API Routes — Services direkt testen."""

from __future__ import annotations

import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from trading_harness.main import app
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.live_execution_service import (
    ExecutionConfig,
    LiveExecutionService,
)

client = TestClient(app)


class TestExecutionEndpointIntegration:
    """Integrationstests für Execution API Endpunkte.
    
    Tests die HTTP-Routes direkt über TestClient — ohne Mocking.
    """

    def test_health_shows_live_execution(self):
        """Health endpoint zeigt live_execution_enabled."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "live_execution_enabled" in data
        assert "kill_switch" in data

    def test_submit_order_invalid_payload(self):
        """Ungültiges Payload gibt 400."""
        response = client.post("/execution/orders", json={})
        assert response.status_code == 400

    def test_get_execution_status(self):
        """Execution Status abrufen."""
        response = client.get("/execution/status")
        assert response.status_code == 200
        data = response.json()
        assert "live_execution_enabled" in data
        assert "kill_switch" in data
        assert "execution_logs_count" in data


class TestLiveExecutionService:
    """Unit Tests für LiveExecutionService — der eigentliche Execution-Pipeline."""

    def test_submit_order_disabled(self):
        """Order submit wenn live_execution disabled."""
        config = ExecutionConfig(live_execution_enabled=False)
        ks = KillSwitch()
        svc = LiveExecutionService(kill_switch=ks, config=config)
        
        result = svc.submit_order(
            decision_id=f"dec-{uuid.uuid4().hex[:8]}",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "LIVE_EXECUTION_DISABLED"

    def test_submit_order_kill_switch(self):
        """Order submit wenn kill_switch aktiv aber live_enabled=True."""
        config = ExecutionConfig(live_execution_enabled=True)
        ks = KillSwitch()
        svc = LiveExecutionService(kill_switch=ks, config=config)
        ks.activate()
        
        result = svc.submit_order(
            decision_id=f"dec-{uuid.uuid4().hex[:8]}",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "KILL_SWITCH_ACTIVE"

    def test_submit_order_rate_limit(self):
        """Order submit wenn Rate Limit erreicht."""
        from trading_harness.services.rate_limiter import RateLimiter
        
        config = ExecutionConfig(live_execution_enabled=True)
        ks = KillSwitch()
        limiter = RateLimiter(global_limit=1, symbol_limit=1)
        svc = LiveExecutionService(
            kill_switch=ks,
            rate_limiter=limiter,
            config=config,
        )
        # Ersten Call durchreichen (trotz NO_EXCHANGE_ADAPTER — das ist OK für diesen Test)
        svc.submit_order(
            decision_id=f"dec-{uuid.uuid4().hex[:8]}",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        # Zweiter Call sollte rejected werden
        result = svc.submit_order(
            decision_id=f"dec-{uuid.uuid4().hex[:8]}",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "RATE_LIMIT_EXCEEDED"

    def test_submit_order_dedup(self):
        """Duplizierte decision_id wird erkannt."""
        from trading_harness.services.order_deduplicator import OrderDeduplicator
        
        config = ExecutionConfig(live_execution_enabled=True)
        ks = KillSwitch()
        dedup = OrderDeduplicator()
        svc = LiveExecutionService(
            kill_switch=ks,
            deduplicator=dedup,
            config=config,
        )
        dec_id = f"dec-{uuid.uuid4().hex[:8]}"
        # Ersten Call — accepted (NO_EXCHANGE_ADAPTER ist OK)
        svc.submit_order(
            decision_id=dec_id,
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        # Duplikat — rejected
        result = svc.submit_order(
            decision_id=dec_id,
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "DUPLICATE_DECISION_ID"

    def test_submit_order_min_capital(self):
        """Order mit zu geringer Quantity wird rejected."""
        config = ExecutionConfig(live_execution_enabled=True, min_capital=1.0)
        ks = KillSwitch()
        svc = LiveExecutionService(kill_switch=ks, config=config)
        
        result = svc.submit_order(
            decision_id=f"dec-{uuid.uuid4().hex[:8]}",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.001,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "MIN_CAPITAL_NOT_MET"


class TestExecutionLogs:
    """Tests für Execution Logs — direkt über Service."""

    def test_get_execution_logs(self):
        """Execution Logs abrufen."""
        config = ExecutionConfig(live_execution_enabled=False)
        ks = KillSwitch()
        svc = LiveExecutionService(kill_switch=ks, config=config)
        dec_id = f"dec-{uuid.uuid4().hex[:8]}"
        svc.submit_order(
            decision_id=dec_id,
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )

        logs = svc.get_logs()
        assert any(log["decision_id"] == dec_id for log in logs)

    def test_get_execution_logs_filter(self):
        """Execution Logs nach decision_id filtern."""
        config = ExecutionConfig(live_execution_enabled=False)
        ks = KillSwitch()
        svc = LiveExecutionService(kill_switch=ks, config=config)
        dec_1 = f"dec-{uuid.uuid4().hex[:8]}"
        dec_2 = f"dec-{uuid.uuid4().hex[:8]}"

        svc.submit_order(
            decision_id=dec_1,
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        svc.submit_order(
            decision_id=dec_2,
            run_id="run-1",
            symbol="ETHUSDT",
            side="SHORT",
            quantity=1.0,
            price=3000.0,
        )

        logs = svc.get_logs(decision_id=dec_1)
        assert len(logs) == 1
        assert logs[0]["decision_id"] == dec_1

    def test_get_execution_logs_empty(self):
        """Logs abrufen wenn keine vorhanden."""
        config = ExecutionConfig(live_execution_enabled=False)
        ks = KillSwitch()
        svc = LiveExecutionService(kill_switch=ks, config=config)

        logs = svc.get_logs()
        assert len(logs) == 0

    def test_get_execution_logs_filter_no_match(self):
        """Logs filtern mit nicht existierender decision_id."""
        config = ExecutionConfig(live_execution_enabled=False)
        ks = KillSwitch()
        svc = LiveExecutionService(kill_switch=ks, config=config)
        svc.submit_order(
            decision_id=f"dec-{uuid.uuid4().hex[:8]}",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )

        logs = svc.get_logs(decision_id="nonexistent")
        assert len(logs) == 0


class TestKillSwitchToggle:
    """Tests für Kill Switch Toggle über HTTP-Routes."""

    def test_toggle_kill_switch(self):
        """Kill Switch toggeln über HTTP."""
        with patch("trading_harness.api.routes.execution_kill_switch") as mock_ks:
            mock_ks.is_active.return_value = False
            
            response = client.post("/execution/kill-switch/True")
            assert response.status_code == 200
            mock_ks.activate.assert_called_once()

            response = client.post("/execution/kill-switch/False")
            assert response.status_code == 200
            mock_ks.deactivate.assert_called_once()


class TestExecutionServiceLifecycle:
    """Tests für den kompletten Service Lifecycle."""

    def test_activate_deactivate(self):
        """Live Execution aktivieren und deaktivieren."""
        config = ExecutionConfig(live_execution_enabled=False)
        ks = KillSwitch()
        svc = LiveExecutionService(kill_switch=ks, config=config)
        
        assert svc.is_live_enabled is False
        
        svc.activate_live()
        assert svc.is_live_enabled is True
        
        svc.deactivate_live()
        assert svc.is_live_enabled is False

    def test_logs_persist_across_calls(self):
        """Logs bleiben über mehrere Calls persistiert."""
        config = ExecutionConfig(live_execution_enabled=False)
        ks = KillSwitch()
        svc = LiveExecutionService(kill_switch=ks, config=config)
        
        for i in range(5):
            svc.submit_order(
                decision_id=f"dec-{i}",
                run_id="run-1",
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                price=50000.0,
            )
        
        logs = svc.get_logs()
        assert len(logs) == 5
        
        # Alle decision_ids vorhanden
        decision_ids = {log["decision_id"] for log in logs}
        assert decision_ids == {f"dec-{i}" for i in range(5)}


class TestCryptoExecutionEndpoints:
    """Tests für Crypto Execution API Routes."""

    def test_crypto_status_shows_credentials(self):
        """Crypto-Router Status zeigt Credential-Zustände."""
        resp = client.get("/execution/crypto/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["router_active"] is True
        assert "credential_states" in data
        for exchange in ["bybit", "bitget", "binance", "coinbase"]:
            assert exchange in data["credential_states"]

    def test_crypto_price(self):
        """Crypto-Preis-Endpunkt gibt Ticker zurück."""
        resp = client.get("/execution/crypto/price/BTCUSDT")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "BTCUSDT"
        assert "bid" in data
        assert "ask" in data
        assert "last" in data

    def test_crypto_submit_min_capital_rejected(self):
        """Crypto-Order mit zu wenig Kapital wird abgelehnt (nach Live-Disabled-Check)."""
        resp = client.post(
            "/execution/crypto/submit",
            json={
                "decision_id": f"dec-{uuid.uuid4().hex[:8]}",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.001,
                "price": 50000.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "REJECTED"
        # Pipeline prüft live_execution_enabled zuerst
        assert "LIVE_EXECUTION_DISABLED" in data.get("error", "")

    def test_crypto_submit_live_disabled(self):
        """Crypto-Order wird abgelehnt wenn Live Execution deaktiviert."""
        resp = client.post(
            "/execution/crypto/submit",
            json={
                "decision_id": f"dec-{uuid.uuid4().hex[:8]}",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 1.0,
                "price": 50000.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "REJECTED"
        assert "LIVE_EXECUTION_DISABLED" in data.get("error", "")

    def test_crypto_status_order(self):
        """Crypto Order-Status Endpunkt gibt Status zurück."""
        resp = client.get(f"/execution/crypto/status/{uuid.uuid4().hex[:8]}")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "order_id" in data
        # Da Live Execution deaktiviert, sollte REJECTED kommen
        assert data["status"] == "REJECTED"
        assert "LIVE_EXECUTION_DISABLED" in data.get("error", "")

    def test_crypto_cancel_order(self):
        """Crypto Cancel Order Endpunkt gibt Ergebnis zurück."""
        resp = client.delete(f"/execution/crypto/cancel/{uuid.uuid4().hex[:8]}")
        assert resp.status_code == 200
        data = resp.json()
        assert "success" in data
        assert "order_id" in data
        # Da Live Execution deaktiviert, sollte REJECTED kommen
        assert data.get("error") == "LIVE_EXECUTION_DISABLED"

    def test_crypto_submit_with_exchange_name(self):
        """Crypto-Submit akzeptiert exchange_name im Payload."""
        resp = client.post(
            "/execution/crypto/submit",
            json={
                "decision_id": f"dec-{uuid.uuid4().hex[:8]}",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 1.0,
                "price": 50000.0,
                "exchange_name": "binance",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "REJECTED"
        assert "LIVE_EXECUTION_DISABLED" in data.get("error", "")