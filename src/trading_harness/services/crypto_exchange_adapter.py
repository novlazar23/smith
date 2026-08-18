"""Crypto Exchange Adapters — Bybit V5 & Bitget V3 via shared base class.

Real HTTP integration (httpx) with HMAC-SHA256 signing.
Simulation mode enabled when credentials are not configured.
Network policy enforced; credentials sourced via CredentialManager.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from trading_harness.services.credential_manager import CredentialManager
from trading_harness.services.exchange_adapter import (
    ExchangeAdapter,
    ExchangeAdapterError,
)
from trading_harness.services.network_policy import NetworkPolicy


class BaseCryptoExchangeAdapter(ExchangeAdapter, ABC):
    """Basisklasse für Crypto-Exchange-Adapter mit HTTP- und Signatur-Logik.

    NetworkPolicy wird vor jeder HTTP-Anfrage geprüft (R5.15–R5.16).
    Credentials werden optional über CredentialManager gelesen (R5.18–R5.19).
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        simulated: bool = True,
        network_policy: NetworkPolicy | None = None,
        credential_manager: CredentialManager | None = None,
        credential_key_prefix: str = "",
        passphrase: str = "",
    ) -> None:
        self._network_policy = network_policy
        self._credential_manager = credential_manager
        self._credential_key_prefix = credential_key_prefix
        self._passphrase = passphrase

        # Credentials: explizit übergeben ODER über CredentialManager
        if not api_key and credential_manager and credential_key_prefix:
            api_key = credential_manager.get(f"{credential_key_prefix}_API_KEY") or ""
        if not api_secret and credential_manager and credential_key_prefix:
            api_secret = credential_manager.get(f"{credential_key_prefix}_API_SECRET") or ""
        if not passphrase and credential_manager and credential_key_prefix:
            passphrase = credential_manager.get(f"{credential_key_prefix}_API_PASSPHRASE") or ""

        self._api_key = api_key
        self._api_secret = api_secret
        self._simulated = simulated or not (api_key and api_secret)
        self._client = httpx.Client(timeout=10.0)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _sign_request(
        self, params: dict[str, Any], timestamp: str, data: dict[str, Any] | None = None
    ) -> dict[str, str]:
        """Signiert eine Anfrage mit HMAC-SHA256."""
        ...

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """Baut Auth-Header."""
        ...

    @abstractmethod
    def _submit_order_url(self) -> str:
        ...

    @abstractmethod
    def _get_order_url(self, order_id: str) -> str:
        ...

    @abstractmethod
    def _cancel_order_url(self) -> str:
        ...

    @abstractmethod
    def _balance_url(self) -> str:
        ...

    @abstractmethod
    def _ticker_url(self, symbol: str) -> str:
        ...

    def _make_signed_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Führt eine signierte API-Anfrage aus.

        NetworkPolicy wird vor jeder nicht-simulierten Anfrage geprüft (R5.15–R5.16).
        """
        if self._simulated:
            return self._simulate(method, url, params, data)

        # R5.15–R5.16: Network Policy Check
        if self._network_policy and not self._network_policy.is_allowed(method, url):
            raise ExchangeAdapterError(f"NETWORK_POLICY_BLOCKED: {method} {url}")

        timestamp = str(int(time.time() * 1000))
        signed = self._sign_request(params or {}, timestamp, data)

        headers = self._build_headers()
        # Merge signature fields into headers
        for k, v in signed.items():
            headers[k] = v

        try:
            if method == "GET":
                resp = self._client.get(url, params=params, headers=headers)
            else:
                resp = self._client.post(url, params=params, json=data or {}, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise ExchangeAdapterError(f"API request failed: {e}") from e

    def _simulate(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Simulierte Antwort (keine echte Netzwerk-Aktion)."""
        params = params or {}
        data = data or {}
        if "order/create" in url or "createOrder" in url or "/trade/place-order" in url:
            return {
                "success": True,
                "order_id": f"sim-{int(time.time() * 1000)}",
                "status": "FILLED",
                "filled_quantity": params.get("qty", 1.0) or data.get("qty", 1.0),
            }
        if "order/realtime" in url or "order/detail" in url:
            return {"orderId": params.get("orderId", data.get("orderId", "")), "status": "FILLED"}
        if "cancel" in url:
            return {"success": True, "order_id": params.get("orderId", data.get("orderId", ""))}
        if "balance" in url:
            return {"USDT": 100000.0, "BTC": 1.0}
        if "ticker" in url or "tickers" in url:
            return {"bid": 50000.0, "ask": 50001.0, "last": 50000.5}
        return {"success": True}

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": "BUY" if side.upper() in ("BUY", "LONG") else "SELL",
            "qty": str(quantity),
            "price": str(price) if price > 0 else "0",
            "orderType": order_type,
        }
        resp = self._make_signed_request(
            "POST", self._submit_order_url(), data=params
        )
        return {
            "order_id": resp.get("order_id", resp.get("result", {}).get("orderId")),
            "status": resp.get("status", "FILLED"),
        }

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        resp = self._make_signed_request(
            "GET", self._get_order_url(order_id), params={"orderId": order_id}
        )
        return {
            "status": resp.get("status", "UNKNOWN"),
            "filled_quantity": resp.get("filledQty", 0.0),
        }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        resp = self._make_signed_request(
            "POST", self._cancel_order_url(), data={"orderId": order_id}
        )
        return {"success": resp.get("success", True)}

    def get_balance(self, symbol: str) -> float:
        resp = self._make_signed_request("GET", self._balance_url())
        for asset in ("USDT", symbol.split("USDT")[0] if "USDT" in symbol else symbol):
            if asset in resp:
                return float(resp[asset])
        return 100000.0

    def get_ticker(self, symbol: str) -> dict[str, float]:
        resp = self._make_signed_request(
            "GET", self._ticker_url(symbol), params={"symbol": symbol}
        )
        tickers = resp.get("result", {}).get("list", [resp])
        if tickers:
            tick = tickers[0]
            return {
                "bid": float(tick.get("bidPx", tick.get("bid", 0))),
                "ask": float(tick.get("askPx", tick.get("ask", 0))),
                "last": float(tick.get("lastPx", tick.get("last", 0))),
            }
        return {"bid": 0.0, "ask": 0.0, "last": 0.0}

    def close(self) -> None:
        self._client.close()


class BybitExchangeAdapter(BaseCryptoExchangeAdapter):
    """Bybit V5 API Adapter.

    Signing for POST /v5/order/create:
      signature_string = timestamp + api_key + recv_window + jsonBody
    Headers: X-BAPI-API-KEY, X-BAPI-TIMESTAMP, X-BAPI-SIGN, X-BAPI-RECV-WINDOW
    """

    API_BASE = "https://api.bybit.com"
    _RECV_WINDOW = "5000"

    @property
    def name(self) -> str:
        return "BYBIT"

    def _sign_request(
        self, params: dict[str, Any], timestamp: str, data: dict[str, Any] | None = None
    ) -> dict[str, str]:
        # POST requests: sign timestamp + apiKey + recvWindow + jsonBody
        body_str = json.dumps(data, separators=(",", ":")) if data else ""
        signature_string = f"{timestamp}{self._api_key}{self._RECV_WINDOW}{body_str}"
        signature = hmac.new(
            self._api_secret.encode(),
            signature_string.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
        }

    def _build_headers(self) -> dict[str, str]:
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-RECV-WINDOW": self._RECV_WINDOW,
            "Content-Type": "application/json",
        }

    def _submit_order_url(self) -> str:
        return f"{self.API_BASE}/v5/order/create"

    def _get_order_url(self, order_id: str) -> str:
        return f"{self.API_BASE}/v5/order/realtime"

    def _cancel_order_url(self) -> str:
        return f"{self.API_BASE}/v5/order/cancel"

    def _balance_url(self) -> str:
        return f"{self.API_BASE}/v5/account/wallet"

    def _ticker_url(self, symbol: str) -> str:
        return f"{self.API_BASE}/v5/market/tickers"


class BitgetExchangeAdapter(BaseCryptoExchangeAdapter):
    """Bitget V3 (UTA) API Adapter.

    Signing: timestamp + METHOD + requestPath + body  (base64-encoded HMAC)
    Headers: ACCESS-KEY, ACCESS-SIGN, ACCESS-TIMESTAMP, ACCESS-PASSPHRASE
    """

    API_BASE = "https://api.bitget.com"

    @property
    def name(self) -> str:
        return "BITGET"

    def _sign_request(
        self, params: dict[str, Any], timestamp: str, data: dict[str, Any] | None = None
    ) -> dict[str, str]:
        # Use the actual request path; params are for GET query strings only
        path = "/api/v3/trade/place-order"
        body_str = json.dumps(data, separators=(",", ":")) if data else ""
        prehash = f"{timestamp}POST{path}{body_str}"
        digest = hmac.new(
            self._api_secret.encode(),
            prehash.encode(),
            hashlib.sha256,
        ).digest()
        signature = base64.b64encode(digest).decode("utf-8")
        return {
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
        }

    def _build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "ACCESS-KEY": self._api_key,
            "ACCESS-PASSPHRASE": self._passphrase or self._api_secret,
        }

    def _submit_order_url(self) -> str:
        return f"{self.API_BASE}/api/v3/trade/place-order"

    def _get_order_url(self, order_id: str) -> str:
        return f"{self.API_BASE}/api/v2/spot/order/detail"

    def _cancel_order_url(self) -> str:
        return f"{self.API_BASE}/api/v3/spot/order/cancel"

    def _balance_url(self) -> str:
        return f"{self.API_BASE}/api/v2/spot/account/balance"

    def _ticker_url(self, symbol: str) -> str:
        return f"{self.API_BASE}/api/v2/spot/market/ticker"