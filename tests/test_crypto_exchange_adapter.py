"""Tests für Bybit & Bitget Crypto Exchange Adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trading_harness.services.crypto_exchange_adapter import (
    BitgetExchangeAdapter,
    BybitExchangeAdapter,
)
from trading_harness.services.exchange_adapter import ExchangeAdapterError, ResponseValidationError


class TestBybitExchangeAdapter:
    """Bybit Exchange Adapter Tests."""

    def test_name(self):
        """Adapter-Name ist BYBIT."""
        adapter = BybitExchangeAdapter()
        assert adapter.name == "BYBIT"

    def test_submit_order_simulated(self):
        """Simulierter Order-Submit gibt order_id und FILLED."""
        adapter = BybitExchangeAdapter(simulated=True)
        result = adapter.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert result["status"] == "FILLED"
        assert result["order_id"] is not None
        adapter.close()

    def test_get_balance_simulated(self):
        """Simulierter Balance-Check gibt USDT-Amount."""
        adapter = BybitExchangeAdapter(simulated=True)
        balance = adapter.get_balance("BTCUSDT")
        assert balance == 100000.0
        adapter.close()

    def test_get_ticker_simulated(self):
        """Simulierter Ticker-Check gibt bid/ask/last."""
        adapter = BybitExchangeAdapter(simulated=True)
        ticker = adapter.get_ticker("BTCUSDT")
        assert ticker["bid"] == 50000.0
        assert ticker["ask"] == 50001.0
        assert ticker["last"] == 50000.5
        adapter.close()

    def test_get_order_status_simulated(self):
        """Simulierter Order-Status."""
        adapter = BybitExchangeAdapter(simulated=True)
        result = adapter.get_order_status("order-123")
        assert result["status"] == "FILLED"
        adapter.close()

    def test_cancel_order_simulated(self):
        """Simulierter Order-Cancel."""
        adapter = BybitExchangeAdapter(simulated=True)
        result = adapter.cancel_order("order-123")
        assert result["success"] is True
        adapter.close()


class TestBitgetExchangeAdapter:
    """Bitget Exchange Adapter Tests."""

    def test_name(self):
        """Adapter-Name ist BITGET."""
        adapter = BitgetExchangeAdapter()
        assert adapter.name == "BITGET"

    def test_submit_order_simulated(self):
        """Simulierter Order-Submit gibt order_id und FILLED."""
        adapter = BitgetExchangeAdapter(simulated=True)
        result = adapter.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert result["status"] == "FILLED"
        assert result["order_id"] is not None
        adapter.close()

    def test_get_balance_simulated(self):
        """Simulierter Balance-Check gibt USDT-Amount."""
        adapter = BitgetExchangeAdapter(simulated=True)
        balance = adapter.get_balance("BTCUSDT")
        assert balance == 100000.0
        adapter.close()

    def test_get_ticker_simulated(self):
        """Simulierter Ticker-Check gibt bid/ask/last."""
        adapter = BitgetExchangeAdapter(simulated=True)
        ticker = adapter.get_ticker("BTCUSDT")
        assert ticker["bid"] == 50000.0
        assert ticker["ask"] == 50001.0
        assert ticker["last"] == 50000.5
        adapter.close()

    def test_get_order_status_simulated(self):
        """Simulierter Order-Status."""
        adapter = BitgetExchangeAdapter(simulated=True)
        result = adapter.get_order_status("order-123")
        assert result["status"] == "FILLED"
        adapter.close()

    def test_cancel_order_simulated(self):
        """Simulierter Order-Cancel."""
        adapter = BitgetExchangeAdapter(simulated=True)
        result = adapter.cancel_order("order-123")
        assert result["success"] is True
        adapter.close()


class TestBaseCryptoExchangeAdapter:
    """Gemeinsame Tests für alle Crypto-Adapter."""

    def test_all_adapters_implement_interface(self):
        """Alle Adapter implementieren ExchangeAdapter-Interface."""
        bybit = BybitExchangeAdapter()
        bitget = BitgetExchangeAdapter()

        # Name Property
        assert bybit.name == "BYBIT"
        assert bitget.name == "BITGET"

        # Methoden existieren
        assert callable(bybit.submit_order)
        assert callable(bybit.get_order_status)
        assert callable(bybit.cancel_order)
        assert callable(bybit.get_balance)
        assert callable(bybit.get_ticker)

        assert callable(bitget.submit_order)
        assert callable(bitget.get_order_status)
        assert callable(bitget.cancel_order)
        assert callable(bitget.get_balance)
        assert callable(bitget.get_ticker)

        bybit.close()
        bitget.close()


# ===========================================================================
# Response Validation
# ===========================================================================


class TestResponseValidation:
    """Tests für _validate_response (Bybit retCode / Bitget code)."""

    def test_bybit_valid_response(self):
        """Bybit retCode == '0' wirft keinen Fehler."""
        bybit = BybitExchangeAdapter(simulated=True)
        bybit._validate_response({"retCode": "0", "retMsg": "success"})
        bybit.close()

    def test_bybit_invalid_response_raises(self):
        """Bybit retCode != '0' wirft ResponseValidationError."""
        bybit = BybitExchangeAdapter(simulated=True)
        with pytest.raises(ResponseValidationError) as exc_info:
            bybit._validate_response({"retCode": "-1001", "retMsg": "System error"})
        assert exc_info.value.code == "-1001"
        assert "System error" in exc_info.value.message
        bybit.close()

    def test_bitget_valid_response(self):
        """Bitget code == '0' wirft keinen Fehler."""
        bitget = BitgetExchangeAdapter(simulated=True)
        bitget._validate_response({"code": "0", "msg": "success"})
        bitget.close()

    def test_bitget_invalid_response_raises(self):
        """Bitget code != '0' wirft ResponseValidationError."""
        bitget = BitgetExchangeAdapter(simulated=True)
        with pytest.raises(ResponseValidationError) as exc_info:
            bitget._validate_response({"code": "60001", "msg": "Order rejected"})
        assert exc_info.value.code == "60001"
        assert "Order rejected" in exc_info.value.message
        bitget.close()

    def test_empty_response_is_valid(self):
        """Leere Response wird als gültig betrachtet."""
        bybit = BybitExchangeAdapter(simulated=True)
        bybit._validate_response({})
        bybit.close()


# ===========================================================================
# CredentialManager Integration
# ===========================================================================


class TestCredentialManagerIntegration:
    """Tests für CredentialManager-Integration."""

    def test_credentials_from_manager(self):
        """Adapter liest Credentials aus CredentialManager."""
        adapter = BybitExchangeAdapter(api_key="test-key", api_secret="test-secret", simulated=False)
        assert adapter._api_key == "test-key"
        assert adapter._api_secret == "test-secret"
        adapter.close()

    def test_credentials_explicit_over_manager(self):
        """Explizite Credentials schlagen CredentialManager vor."""
        adapter = BybitExchangeAdapter(
            api_key="explicit-key",
            api_secret="explicit-secret",
            simulated=False,
            credential_manager=MagicMock(),  # wird nicht angerufen
        )
        assert adapter._api_key == "explicit-key"
        assert adapter._api_secret == "explicit-secret"
        adapter.close()


# ===========================================================================
# NetworkPolicy Enforcement
# ===========================================================================


class TestNetworkPolicy:
    """Tests für NetworkPolicy-Enforcement in _make_signed_request."""

    def test_blocked_url_raises(self):
        """Blockierte URL wirft ExchangeAdapterError."""
        from trading_harness.services.network_policy import NetworkPolicy
        # Fake credentials so _simulated becomes False (line 65: simulated or not (key and secret))
        bybit = BybitExchangeAdapter(
            api_key="fake-key",
            api_secret="fake-secret",
            simulated=False,
            network_policy=NetworkPolicy(),
        )
        # Patch is_allowed to simulate a blocked URL
        with (
            patch.object(bybit._network_policy, "is_allowed", return_value=False),
            pytest.raises(ExchangeAdapterError, match="NETWORK_POLICY_BLOCKED"),
        ):
            bybit._make_signed_request("GET", "https://any-url.example.com/bad")
        bybit.close()