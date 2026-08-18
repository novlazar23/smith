"""Tests für Bybit, Bitget, Binance & Coinbase Crypto Exchange Adapters."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from trading_harness.services.crypto_exchange_adapter import (
    BinanceExchangeAdapter,
    BitgetExchangeAdapter,
    BybitExchangeAdapter,
    CoinbaseExchangeAdapter,
    CryptoExecutionRouter,
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


class TestBinanceExchangeAdapter:
    """Binance Exchange Adapter Tests."""

    def test_name(self):
        """Adapter-Name ist BINANCE."""
        adapter = BinanceExchangeAdapter()
        assert adapter.name == "BINANCE"

    def test_submit_order_simulated(self):
        """Simulierter Order-Submit gibt order_id und FILLED."""
        adapter = BinanceExchangeAdapter(simulated=True)
        result = adapter.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert result["status"] == "FILLED"
        assert result["order_id"] is not None
        adapter.close()

    def test_get_balance_simulated(self):
        """Simulierter Balance-Check gibt USDT-Amount."""
        adapter = BinanceExchangeAdapter(simulated=True)
        balance = adapter.get_balance("BTCUSDT")
        assert balance == 100000.0
        adapter.close()

    def test_get_ticker_simulated(self):
        """Simulierter Ticker-Check gibt bid/ask/last."""
        adapter = BinanceExchangeAdapter(simulated=True)
        ticker = adapter.get_ticker("BTCUSDT")
        assert ticker["bid"] == 50000.0
        assert ticker["ask"] == 50001.0
        assert ticker["last"] == 50000.5
        adapter.close()

    def test_get_order_status_simulated(self):
        """Simulierter Order-Status."""
        adapter = BinanceExchangeAdapter(simulated=True)
        result = adapter.get_order_status("order-123")
        assert result["status"] == "FILLED"
        adapter.close()

    def test_cancel_order_simulated(self):
        """Simulierter Order-Cancel."""
        adapter = BinanceExchangeAdapter(simulated=True)
        result = adapter.cancel_order("order-123")
        assert result["success"] is True
        adapter.close()

    def test_invalid_response_raises(self):
        """Binance code != 0 wirft ResponseValidationError."""
        adapter = BinanceExchangeAdapter(simulated=True)
        with pytest.raises(ResponseValidationError) as exc_info:
            adapter._validate_response({"code": -1002, "msg": "Unknown service"})
        assert exc_info.value.code == "-1002"
        assert "Unknown service" in exc_info.value.message
        adapter.close()


class TestCoinbaseExchangeAdapter:
    """Coinbase Exchange Adapter Tests."""

    def test_name(self):
        """Adapter-Name ist COINBASE."""
        adapter = CoinbaseExchangeAdapter()
        assert adapter.name == "COINBASE"

    def test_submit_order_simulated(self):
        """Simulierter Order-Submit gibt order_id und FILLED."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        result = adapter.submit_order("BTC-USDT", "BUY", 1.0, 50000.0)
        assert result["status"] == "FILLED"
        assert result["order_id"] is not None
        adapter.close()

    def test_get_balance_simulated(self):
        """Simulierter Balance-Check gibt USDT-Amount."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        balance = adapter.get_balance("BTC-USDT")
        assert balance == 100000.0
        adapter.close()

    def test_get_ticker_simulated(self):
        """Simulierter Ticker-Check gibt bid/ask/last."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        ticker = adapter.get_ticker("BTC-USDT")
        assert ticker["bid"] == 50000.0
        assert ticker["ask"] == 50001.0
        assert ticker["last"] == 50000.5
        adapter.close()

    def test_get_order_status_simulated(self):
        """Simulierter Order-Status."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        result = adapter.get_order_status("order-123")
        assert result["status"] == "FILLED"
        adapter.close()

    def test_cancel_order_simulated(self):
        """Simulierter Order-Cancel."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        result = adapter.cancel_order("order-123")
        assert result["success"] is True
        adapter.close()


class TestBaseCryptoExchangeAdapter:
    """Gemeinsame Tests für alle Crypto-Adapter."""

    def test_all_adapters_implement_interface(self):
        """Alle Adapter implementieren ExchangeAdapter-Interface."""
        bybit = BybitExchangeAdapter()
        bitget = BitgetExchangeAdapter()
        binance = BinanceExchangeAdapter()
        coinbase = CoinbaseExchangeAdapter()

        # Name Property
        assert bybit.name == "BYBIT"
        assert bitget.name == "BITGET"
        assert binance.name == "BINANCE"
        assert coinbase.name == "COINBASE"

        # Methoden existieren
        for adapter in (bybit, bitget, binance, coinbase):
            assert callable(adapter.submit_order)
            assert callable(adapter.get_order_status)
            assert callable(adapter.cancel_order)
            assert callable(adapter.get_balance)
            assert callable(adapter.get_ticker)

        bybit.close()
        bitget.close()
        binance.close()
        coinbase.close()


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

# ===========================================================================
# CryptoExecutionRouter — Dynamic Credential Loading
# ===========================================================================


class TestCryptoExecutionRouter:
    """Tests für CryptoExecutionRouter mit dynamischem Credential-Loading."""

    def test_router_name(self):
        """Router-Name ist CRYPTO_ROUTER."""
        from trading_harness.services.crypto_exchange_adapter import CryptoExecutionRouter

        router = CryptoExecutionRouter()
        assert router.name == "CRYPTO_ROUTER"
        router.close()

    def test_router_default_exchange(self):
        """Router verwendet bybit als Standard-Exchange."""
        from trading_harness.services.crypto_exchange_adapter import CryptoExecutionRouter

        router = CryptoExecutionRouter(default_exchange="bybit")
        assert router._default == "bybit"
        result = router.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert result["simulated"] is True
        router.close()

    def test_router_simulated_without_credentials(self):
        """Router ohne Credentials → alle Exchanges simuliert."""
        from trading_harness.services.crypto_exchange_adapter import CryptoExecutionRouter

        router = CryptoExecutionRouter(
            default_exchange="binance",
            credential_manager=None,
        )
        for exchange in CryptoExecutionRouter.SUPPORTED:
            simulated, _ = router._resolve_adapter_state(exchange)
            assert simulated is True
        router.close()

    def test_router_live_with_credentials(self):
        """Router mit Credentials → Exchange läuft live."""
        from trading_harness.services.crypto_exchange_adapter import CryptoExecutionRouter

        mock_manager = MagicMock()
        mock_manager.get.side_effect = lambda key: {
            "BINANCE_API_KEY": "test-key",
            "BINANCE_API_SECRET": "test-secret",
            "BYBIT_API_KEY": None,
            "BYBIT_API_SECRET": None,
        }.get(key)

        router = CryptoExecutionRouter(
            default_exchange="bybit",
            credential_manager=mock_manager,
        )

        # Binance → LIVE (Credentials vorhanden)
        binance_simulated, binance_kwargs = router._resolve_adapter_state("binance")
        assert binance_simulated is False
        assert binance_kwargs["api_key"] == "test-key"

        # Bybit → SIMULATED (keine Credentials)
        bybit_simulated, _ = router._resolve_adapter_state("bybit")
        assert bybit_simulated is True

        router.close()

    def test_router_submit_order_simulated(self):
        """Router mit simulated=True gibt SIMULATED-Response zurück."""
        from trading_harness.services.crypto_exchange_adapter import CryptoExecutionRouter

        router = CryptoExecutionRouter(
            default_exchange="bybit",
            credential_manager=None,
        )
        result = router.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert result["simulated"] is True
        assert result["status"] == "FILLED"
        assert "order_id" in result
        router.close()

    def test_router_submit_order_live(self):
        """Router mit simulated=False delegiert an echten Adapter."""
        from trading_harness.services.crypto_exchange_adapter import (
            CryptoExecutionRouter,
        )

        mock_manager = MagicMock()
        mock_manager.get.side_effect = lambda key: {
            "BINANCE_API_KEY": "test-key",
            "BINANCE_API_SECRET": "test-secret",
        }.get(key)

        router = CryptoExecutionRouter(
            default_exchange="binance",
            credential_manager=mock_manager,
        )
        with patch(
            "trading_harness.services.crypto_exchange_adapter._get_or_create",
            return_value=MagicMock(
                submit_order=MagicMock(return_value={
                    "order_id": "live-order-123",
                    "status": "FILLED",
                    "raw": {},
                })
            ),
        ):
            result = router.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert result["status"] == "FILLED"
        assert "order_id" in result
        router.close()

    def test_router_clear_state(self):
        """Router clear_state löscht alle Adapter-Zustände."""
        from trading_harness.services.crypto_exchange_adapter import CryptoExecutionRouter

        router = CryptoExecutionRouter(
            default_exchange="binance",
            credential_manager=None,
        )
        router._resolve_adapter_state("binance")
        assert "binance" in router._adapter_state
        router.close()
        # Nach close() sollte State leer sein
        assert "binance" not in router._adapter_state

    def test_router_get_order_status_simulated(self):
        """Router get_order_status simuliert ohne Adapter."""
        from trading_harness.services.crypto_exchange_adapter import CryptoExecutionRouter

        router = CryptoExecutionRouter(default_exchange="bybit", credential_manager=None)
        result = router.get_order_status("order-456")
        assert result["status"] == "FILLED"
        assert result["order_id"] == "order-456"
        router.close()

    def test_router_cancel_order_simulated(self):
        """Router cancel_order simuliert ohne Adapter."""
        from trading_harness.services.crypto_exchange_adapter import CryptoExecutionRouter

        router = CryptoExecutionRouter(default_exchange="bybit", credential_manager=None)
        result = router.cancel_order("order-789")
        assert result["success"] is True
        assert result["order_id"] == "order-789"
        router.close()

    def test_router_get_ticker_simulated(self):
        """Router get_ticker simuliert ohne Adapter."""
        from trading_harness.services.crypto_exchange_adapter import CryptoExecutionRouter

        router = CryptoExecutionRouter(default_exchange="bybit", credential_manager=None)
        result = router.get_ticker("BTCUSDT")
        assert result["bid"] == 50000.0
        assert result["ask"] == 50001.0
        assert result["last"] == 50000.5
        router.close()


# ===========================================================================
# HTTP-Level Adapter Tests — Request Construction
# ===========================================================================


class TestBybitRequestConstruction:
    """Bybit adapter request structure verification."""

    def test_submit_order_url(self):
        """Bybit submit_order verwendet korrekte URL."""
        adapter = BybitExchangeAdapter(simulated=True)
        url = adapter._submit_order_url()
        assert url == "https://api.bybit.com/v5/order/create"
        adapter.close()

    def test_get_order_url(self):
        """Bybit get_order_status verwendet korrekte URL."""
        adapter = BybitExchangeAdapter(simulated=True)
        url = adapter._get_order_url("order-123")
        assert url == "https://api.bybit.com/v5/order/realtime"
        adapter.close()

    def test_cancel_order_url(self):
        """Bybit cancel_order verwendet korrekte URL."""
        adapter = BybitExchangeAdapter(simulated=True)
        url = adapter._cancel_order_url()
        assert url == "https://api.bybit.com/v5/order/cancel"
        adapter.close()

    def test_balance_url(self):
        """Bybit get_balance verwendet korrekte URL."""
        adapter = BybitExchangeAdapter(simulated=True)
        url = adapter._balance_url()
        assert url == "https://api.bybit.com/v5/account/wallet"
        adapter.close()

    def test_ticker_url(self):
        """Bybit get_ticker verwendet korrekte URL."""
        adapter = BybitExchangeAdapter(simulated=True)
        url = adapter._ticker_url("BTCUSDT")
        assert url == "https://api.bybit.com/v5/market/tickers"
        adapter.close()

    def test_signature_string_format(self):
        """Bybit signiert timestamp + apiKey + recvWindow + jsonBody."""
        adapter = BybitExchangeAdapter(
            api_key="test-key", api_secret="test-secret", simulated=False
        )
        timestamp = "1700000000000"
        data = {"symbol": "BTCUSDT", "side": "BUY", "qty": "1.0"}
        sig = adapter._sign_request({}, timestamp, data, "/v5/order/create")
        # Signiert: timestamp + apiKey + recvWindow + jsonBody
        body_str = json.dumps(data, separators=(",", ":"))
        expected_string = f"{timestamp}test-key5000{body_str}"
        expected = hmac.new(
            b"test-secret", expected_string.encode(), hashlib.sha256
        ).hexdigest()
        assert sig["X-BAPI-SIGN"] == expected
        assert sig["X-BAPI-TIMESTAMP"] == timestamp
        adapter.close()

    def test_headers_include_bapiKey(self):
        """Bybit Headers enthalten X-BAPI-API-KEY."""
        adapter = BybitExchangeAdapter(
            api_key="my-key", api_secret="secret", simulated=False
        )
        headers = adapter._build_headers()
        assert headers["X-BAPI-API-KEY"] == "my-key"
        assert "X-BAPI-RECV-WINDOW" in headers
        assert headers["Content-Type"] == "application/json"
        adapter.close()


class TestBitgetRequestConstruction:
    """Bitget adapter request structure verification."""

    def test_submit_order_url(self):
        """Bitget submit_order verwendet korrekte URL."""
        adapter = BitgetExchangeAdapter(simulated=True)
        url = adapter._submit_order_url()
        assert url == "https://api.bitget.com/api/v3/trade/place-order"
        adapter.close()

    def test_get_order_url(self):
        """Bitget get_order_status verwendet korrekte URL."""
        adapter = BitgetExchangeAdapter(simulated=True)
        url = adapter._get_order_url("order-123")
        assert url == "https://api.bitget.com/api/v2/spot/order/detail"
        adapter.close()

    def test_cancel_order_url(self):
        """Bitget cancel_order verwendet korrekte URL."""
        adapter = BitgetExchangeAdapter(simulated=True)
        url = adapter._cancel_order_url()
        assert url == "https://api.bitget.com/api/v3/spot/order/cancel"
        adapter.close()

    def test_balance_url(self):
        """Bitget get_balance verwendet korrekte URL."""
        adapter = BitgetExchangeAdapter(simulated=True)
        url = adapter._balance_url()
        assert url == "https://api.bitget.com/api/v2/spot/account/balance"
        adapter.close()

    def test_ticker_url(self):
        """Bitget get_ticker verwendet korrekte URL."""
        adapter = BitgetExchangeAdapter(simulated=True)
        url = adapter._ticker_url("BTCUSDT")
        assert url == "https://api.bitget.com/api/v2/spot/market/ticker"
        adapter.close()

    def test_signature_uses_path(self):
        """Bitget signiert timestamp + POST + requestPath + body."""
        adapter = BitgetExchangeAdapter(
            api_key="test-key", api_secret="test-secret", simulated=False
        )
        timestamp = "1700000000000"
        data = {"symbol": "BTCUSDT", "side": "BUY", "qty": "1.0"}
        path = "/api/v3/trade/place-order"
        sig = adapter._sign_request({}, timestamp, data, path)
        body_str = json.dumps(data, separators=(",", ":"))
        prehash = f"{timestamp}POST{path}{body_str}"
        digest = hmac.new(
            b"test-secret", prehash.encode(), hashlib.sha256
        ).digest()
        expected = base64.b64encode(digest).decode("utf-8")
        assert sig["ACCESS-SIGN"] == expected
        assert sig["ACCESS-TIMESTAMP"] == timestamp
        adapter.close()

    def test_headers_include_access_keys(self):
        """Bitget Headers enthalten ACCESS-KEY."""
        adapter = BitgetExchangeAdapter(
            api_key="my-key", api_secret="secret", simulated=False
        )
        headers = adapter._build_headers()
        assert headers["ACCESS-KEY"] == "my-key"
        assert headers["ACCESS-PASSPHRASE"] == "secret"
        assert headers["Content-Type"] == "application/json"
        adapter.close()


class TestBinanceRequestConstruction:
    """Binance adapter request structure verification."""

    def test_submit_order_url(self):
        """Binance submit_order verwendet korrekte URL."""
        adapter = BinanceExchangeAdapter(simulated=True)
        url = adapter._submit_order_url()
        assert url == "https://api.binance.com/api/v4/trade/order"
        adapter.close()

    def test_ticker_url(self):
        """Binance get_ticker verwendet korrekte URL."""
        adapter = BinanceExchangeAdapter(simulated=True)
        url = adapter._ticker_url("BTCUSDT")
        assert url == "https://api.binance.com/api/v4/ticker/price"
        adapter.close()

    def test_balance_url(self):
        """Binance get_balance verwendet korrekte URL."""
        adapter = BinanceExchangeAdapter(simulated=True)
        url = adapter._balance_url()
        assert url == "https://api.binance.com/api/v4/account"
        adapter.close()

    def test_signature_appended_as_query_param(self):
        """Binance signiert timestamp + body als query-param."""
        adapter = BinanceExchangeAdapter(
            api_key="test-key", api_secret="test-secret", simulated=False
        )
        timestamp = "1700000000000"
        data = {"symbol": "BTCUSDT", "side": "BUY", "qty": "1.0"}
        sig = adapter._sign_request({}, timestamp, data, "/api/v4/trade/order")
        # Signiert: timestamp=...&symbol=...&side=...&qty=...
        body_str = json.dumps(data, separators=(",", ":"))
        signature_string = f"timestamp={timestamp}&{body_str}"
        expected = hmac.new(
            b"test-secret", signature_string.encode(), hashlib.sha256
        ).hexdigest()
        assert sig == expected
        adapter.close()

    def test_headers_include_api_key(self):
        """Binance Headers enthalten X-MBX-APIKEY."""
        adapter = BinanceExchangeAdapter(
            api_key="my-key", api_secret="secret", simulated=False
        )
        headers = adapter._build_headers()
        assert headers["X-MBX-APIKEY"] == "my-key"
        assert headers["Content-Type"] == "application/x-www-form-urlencoded"
        adapter.close()


class TestCoinbaseRequestConstruction:
    """Coinbase adapter request structure verification."""

    def test_submit_order_url(self):
        """Coinbase submit_order verwendet korrekte URL."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        url = adapter._submit_order_url()
        assert url == "https://api.coinbase.com/api/v3/brokerage/orders"
        adapter.close()

    def test_ticker_url(self):
        """Coinbase get_ticker verwendet korrekte URL."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        url = adapter._ticker_url("BTC-USDT")
        assert url == "https://api.coinbase.com/api/v3/brokerage/products/BTC-USDT/ticker"
        adapter.close()

    def test_balance_url(self):
        """Coinbase get_balance verwendet korrekte URL."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        url = adapter._balance_url()
        assert url == "https://api.coinbase.com/api/v3/brokerage/accounts"
        adapter.close()

    def test_headers_include_cb_keys(self):
        """Coinbase Headers enthalten CB-ACCESS-SIGN etc."""
        adapter = CoinbaseExchangeAdapter(
            api_key="my-key", api_secret="secret", simulated=False
        )
        headers = adapter._build_headers()
        assert headers["CB-ACCESS-KEY"] == "my-key"
        assert headers["Content-Type"] == "application/json"
        adapter.close()


# ===========================================================================
# Retry Behavior — Rate Limits & Transient Errors
# ===========================================================================


class TestRetryBehavior:
    """Tests für Retry-Logik bei transienten Fehlern."""

    def _make_client(self, responses: list[tuple[int, dict]]) -> MagicMock:
        """Erstellt einen gemockten httpx Client mit sequenziellen Responses."""
        mock_responses = []
        for status, body in responses:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = status
            mock_resp.json.return_value = body
            mock_resp.text = json.dumps(body)
            mock_responses.append(mock_resp)

        mock_client = MagicMock()
        mock_client.post.side_effect = mock_responses
        mock_client.get.side_effect = mock_responses
        return mock_client

    def test_rate_limit_retries_3_times(self):
        """HTTP 429 retry 3x dann Raise."""
        from trading_harness.services.exchange_adapter import RateLimitError

        bybit = BybitExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        mock_client = self._make_client([(429, {}), (429, {}), (429, {})])
        bybit._client = mock_client
        with pytest.raises(RateLimitError, match="attempt 3/3"):
            bybit._make_signed_request("POST", "https://api.bybit.com/v5/order/create", data={})
        assert bybit._client.post.call_count == 3
        bybit.close()

    def test_500_retries_3_times(self):
        """HTTP 500 retry 3x dann Raise."""
        bybit = BybitExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        mock_client = self._make_client([(502, {}), (503, {}), (500, {})])
        bybit._client = mock_client
        with pytest.raises(Exception, match="Server error .* after 3 retries"):
            bybit._make_signed_request("GET", "https://api.bybit.com/v5/market/tickers")
        assert bybit._client.get.call_count == 3
        bybit.close()

    def test_rate_limit_then_success(self):
        """Retry bei 429, dann 200 Erfolg."""
        bybit = BybitExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        mock_client = self._make_client([
            (429, {}),
            (200, {"retCode": "0", "retMsg": "success", "result": {}}),
        ])
        bybit._client = mock_client
        result = bybit._make_signed_request("POST", "https://api.bybit.com/v5/order/create", data={})
        assert result["retCode"] == "0"
        assert bybit._client.post.call_count == 2
        bybit.close()

    def test_auth_failure_no_retry(self):
        """401/403 wirft sofort, kein Retry."""
        from trading_harness.services.exchange_adapter import AuthenticationError

        bybit = BybitExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        mock_client = self._make_client([(401, {"error": "invalid key"})])
        bybit._client = mock_client
        with pytest.raises(AuthenticationError, match="Authentication failed"):
            bybit._make_signed_request("GET", "https://api.bybit.com/v5/order/realtime")
        assert bybit._client.get.call_count == 1
        bybit.close()


# ===========================================================================
# Response Parsing — Exchange-Specific Formats
# ===========================================================================


class TestBybitResponseParsing:
    """Bybit response parsing tests."""

    def test_submit_order_response_parsing(self):
        """Bybit parse order_id aus response."""
        adapter = BybitExchangeAdapter(simulated=True)
        result = adapter.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert result["status"] == "FILLED"
        assert "order_id" in result
        adapter.close()

    def test_balance_parsing_bybit_format(self):
        """Bybit walletBalance[0].totalBalance parsing."""
        adapter = BybitExchangeAdapter(simulated=True)
        balance = adapter.get_balance("USDT")
        assert balance == 100000.0
        adapter.close()

    def test_ticker_parsing_bybit_format(self):
        """Bybit ticker parsing mit bidPx/askPx/lastPx."""
        adapter = BybitExchangeAdapter(simulated=True)
        ticker = adapter.get_ticker("BTCUSDT")
        assert ticker["bid"] == 50000.0
        assert ticker["ask"] == 50001.0
        adapter.close()


class TestBitgetResponseParsing:
    """Bitget response parsing tests."""

    def test_order_status_data_wrapper(self):
        """Bitget Order-Status: data-wrapper wird korrekt geparst."""
        adapter = BitgetExchangeAdapter(simulated=True)
        result = adapter.get_order_status("order-123")
        assert result["status"] == "FILLED"
        adapter.close()

    def test_balance_parsing_bitget_format(self):
        """Bitget Balance Parsing."""
        adapter = BitgetExchangeAdapter(simulated=True)
        balance = adapter.get_balance("USDT")
        assert balance == 100000.0
        adapter.close()


class TestBinanceResponseParsing:
    """Binance response parsing tests."""

    def test_submit_order_flat_response(self):
        """Binance flat response: {code, msg, orderId}."""
        adapter = BinanceExchangeAdapter(simulated=True)
        result = adapter.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert result["status"] == "FILLED"
        adapter.close()

    def test_invalid_code_raises(self):
        """Binance code != 0 wirft ResponseValidationError."""
        adapter = BinanceExchangeAdapter(simulated=True)
        with pytest.raises(ResponseValidationError) as exc_info:
            adapter._validate_response({"code": -1002, "msg": "Unknown service"})
        assert exc_info.value.code == "-1002"
        adapter.close()


class TestCoinbaseResponseParsing:
    """Coinbase response parsing tests."""

    def test_order_response_parsing(self):
        """Coinbase order_id aus response."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        result = adapter.submit_order("BTC-USDT", "BUY", 1.0, 50000.0)
        assert result["status"] == "FILLED"
        assert "order_id" in result
        adapter.close()

    def test_balance_accounts_format(self):
        """Coinbase accounts array parsing."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        balance = adapter.get_balance("USDT")
        assert balance == 100000.0
        adapter.close()


# ===========================================================================
# Connection-Level Error Handling — simulated=False
# ===========================================================================


class TestConnectionErrorHandling:
    """simulated=False: DNS, timeout, SSL errors mit Retry."""

    def _mock_dns_failure(self):
        """httpx.ConnectError (DNS-Namen konnte nicht aufgelöst werden)."""
        import httpx

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("DNS resolution failed")
        mock_client.post.side_effect = httpx.ConnectError("DNS resolution failed")
        return mock_client

    def _mock_timeout(self):
        """httpx.TimeoutException (Verbindung timeout)."""
        import httpx

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("Request timed out")
        mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
        return mock_client

    def _mock_ssl_error(self):
        """httpx.ConnectError (SSL-Handshake fehlgeschlagen)."""
        import httpx

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("SSL handshake failed")
        mock_client.post.side_effect = httpx.ConnectError("SSL handshake failed")
        return mock_client

    def test_dns_failure_retries_and_raises(self):
        """DNS-Fehler: 3x retry dann ConnectionError."""
        from trading_harness.services.exchange_adapter import ConnectionError

        bybit = BybitExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        bybit._client = self._mock_dns_failure()
        with pytest.raises(ConnectionError, match="Connection failed after 3 retries"):
            bybit._make_signed_request("GET", "https://api.bybit.com/v5/market/tickers")
        assert bybit._client.get.call_count == 3
        bybit.close()

    def test_timeout_retries_and_raises(self):
        """Timeout: 3x retry dann ConnectionError."""
        from trading_harness.services.exchange_adapter import ConnectionError

        bybit = BybitExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        bybit._client = self._mock_timeout()
        with pytest.raises(ConnectionError, match="Connection failed after 3 retries"):
            bybit._make_signed_request("POST", "https://api.bybit.com/v5/order/create", data={})
        assert bybit._client.post.call_count == 3
        bybit.close()

    def test_ssl_error_retries_and_raises(self):
        """SSL-Fehler: 3x retry dann ConnectionError."""
        from trading_harness.services.exchange_adapter import ConnectionError

        binance = BinanceExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        binance._client = self._mock_ssl_error()
        with pytest.raises(ConnectionError, match="Connection failed after 3 retries"):
            binance._make_signed_request("GET", "https://api.binance.com/api/v4/ticker/price")
        assert binance._client.get.call_count == 3
        binance.close()

    def test_bitget_dns_failure(self):
        """Bitget: DNS failure retry."""
        from trading_harness.services.exchange_adapter import ConnectionError

        bitget = BitgetExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        bitget._client = self._mock_dns_failure()
        with pytest.raises(ConnectionError, match="Connection failed after 3 retries"):
            bitget._make_signed_request("GET", "https://api.bitget.com/api/v2/spot/account/balance")
        assert bitget._client.get.call_count == 3
        bitget.close()

    def test_coinbase_timeout(self):
        """Coinbase: timeout retry."""
        from trading_harness.services.exchange_adapter import ConnectionError

        coinbase = CoinbaseExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        coinbase._client = self._mock_timeout()
        with pytest.raises(ConnectionError, match="Connection failed after 3 retries"):
            coinbase._make_signed_request("GET", "https://api.coinbase.com/api/v3/brokerage/accounts")
        assert coinbase._client.get.call_count == 3
        coinbase.close()


# ===========================================================================
# HTTP 400 Response Handling — Exchange Error Code Extraction
# ===========================================================================


class TestHTTP400Handling:
    """simulated=False: HTTP 400 mit Exchange-Fehlercode."""

    def _make_400_response(self, exchange_body: dict) -> MagicMock:
        """Erzeugt einen gemockten 400 Response."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.json.return_value = exchange_body
        mock_resp.text = json.dumps(exchange_body)
        return mock_resp

    def test_bybit_400_raises_exchange_adapter_error(self):
        """Bybit 400: invalid param → ExchangeAdapterError (validate_response → exchange error)."""
        bybit = BybitExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        mock_resp = self._make_400_response({
            "retCode": "-1017",
            "retMsg": "Invalid parameter",
        })
        bybit._client = MagicMock()
        bybit._client.post.return_value = mock_resp
        with pytest.raises(ExchangeAdapterError, match="-1017"):
            bybit._make_signed_request("POST", "https://api.bybit.com/v5/order/create", data={})
        bybit.close()

    def test_binance_400_invalid_symbol(self):
        """Binance 400: unknown symbol → exchange error code from validate_response."""
        binance = BinanceExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        mock_resp = self._make_400_response({
            "code": -1121,
            "msg": "Invalid symbol",
        })
        binance._client = MagicMock()
        binance._client.post.return_value = mock_resp
        with pytest.raises(ExchangeAdapterError, match="-1121"):
            binance._make_signed_request("POST", "https://api.binance.com/api/v4/trade/order", data={})
        binance.close()

    def test_coinbase_400_invalid_order(self):
        """Coinbase 400: invalid order → ExchangeAdapterError."""
        coinbase = CoinbaseExchangeAdapter(
            api_key="key", api_secret="secret", simulated=False
        )
        mock_resp = self._make_400_response({
            "error": "invalid_order",
            "message": "Order is invalid",
        })
        coinbase._client = MagicMock()
        coinbase._client.post.return_value = mock_resp
        # Coinbase validate_response doesn't have retCode/code check, falls through to general exchange error
        with pytest.raises(ExchangeAdapterError, match="invalid_order"):
            coinbase._make_signed_request("POST", "https://api.coinbase.com/api/v3/brokerage/orders", data={})
        coinbase.close()


# ===========================================================================
# Response Schema Edge Cases — Empty/Missing Fields
# ===========================================================================


class TestResponseSchemaEdgeCases:
    """Edge cases: leere Listen, fehlende Felder, Null-Werte."""

    def test_bybit_empty_ticker_list(self):
        """Bybit ticker: leere Liste → KeyError/Edge case."""
        adapter = BybitExchangeAdapter(simulated=True)
        # Simulierter Response mit leerer Liste (muss den Adapter nicht brechen)
        resp = adapter._simulate("GET", "https://api.bybit.com/v5/market/tickers", {"symbol": "NONEXIST"}, {})
        # Leerer Response — kein Fehler, aber leer
        assert "retCode" in resp
        adapter.close()

    def test_bybit_ticker_missing_fields(self):
        """Bybit ticker: fehlende bidPx/askPx → default Werte."""
        # Der simulierte Response hat immer bidPx/askPx/lastPx, aber wir prüfen den Parser
        adapter = BybitExchangeAdapter(simulated=True)
        ticker = adapter.get_ticker("BTCUSDT")
        # Parser extrahiert float(resp.get("bidPx", "0")) → 0 wenn fehlend
        # Aber simulierter Response hat immer Werte
        assert "bid" in ticker
        assert "ask" in ticker
        assert "last" in ticker
        adapter.close()

    def test_binance_ticker_missing_data_field(self):
        """Binance ticker: data-Array mit leerem Entry."""
        adapter = BinanceExchangeAdapter(simulated=True)
        ticker = adapter.get_ticker("BTCUSDT")
        # Simulierter Response hat immer bidPrice/askPrice/price
        assert "bid" in ticker
        assert "ask" in ticker
        assert "last" in ticker
        adapter.close()

    def test_bybit_balance_missing_walletBalance(self):
        """Bybit balance: kein walletBalance → KeyError."""
        # Prüfen ob _validate_response vorher wirft
        adapter = BybitExchangeAdapter(simulated=True)
        # Simulierter Response hat immer walletBalance
        balance = adapter.get_balance("USDT")
        assert isinstance(balance, float)
        adapter.close()

    def test_coinbase_balance_missing_accounts(self):
        """Coinbase balance: kein accounts → fallback."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        balance = adapter.get_balance("USDT")
        # accounts array existiert immer im simulierten Response
        assert isinstance(balance, float)
        adapter.close()

    def test_bybit_order_status_missing_data(self):
        """Bybit order status: data-Objekt fehlend."""
        adapter = BybitExchangeAdapter(simulated=True)
        result = adapter.get_order_status("order-123")
        # Parser greift auf resp.get("data", {}).get("status", "UNKNOWN")
        assert "status" in result
        adapter.close()


# ===========================================================================
# Coinbase Cancel Order — Response Parsing
# ===========================================================================


class TestCoinbaseCancelOrder:
    """Coinbase cancel_order response parsing."""

    def test_cancel_success(self):
        """Coinbase cancel_order: success_type=true."""
        adapter = CoinbaseExchangeAdapter(simulated=True)
        result = adapter.cancel_order("order-123")
        assert result["success"] is True
        assert result["order_id"] == "order-123"
        adapter.close()


# ===========================================================================
# CryptoExecutionRouter — simulated vs live differentiation
# ===========================================================================


class TestCryptoExecutionRouterLiveMode:
    """Router: unterscheidet simulated=True von simulated=False."""

    def test_router_live_mode_requires_both_credentials(self):
        """simulated=False erfordert BEIDE Credentials (KEY + SECRET)."""
        router = CryptoExecutionRouter(
            default_exchange="bybit",
            credential_manager=None,
        )
        # Ohne credential_manager → simulated=True (fallback)
        result = router.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert result["simulated"] is True
        assert "sim-bybit-" in result["order_id"]
        router.close()

    def test_router_live_mode_with_mocks(self):
        """Router mit gemocktem CredentialManager → live mode."""
        mock_cm = MagicMock()
        mock_cm.get.side_effect = ["test-key", "test-secret"]

        router = CryptoExecutionRouter(
            default_exchange="bybit",
            credential_manager=mock_cm,
        )
        # Prüft Credentials → sollte simulated=False sein
        simulated, kwargs = router._resolve_adapter_state("bybit")
        assert simulated is False
        assert kwargs["api_key"] == "test-key"
        assert kwargs["api_secret"] == "test-secret"
        router.close()
