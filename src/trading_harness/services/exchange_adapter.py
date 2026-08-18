"""ExchangeAdapter — polymorphes Interface für Exchange-Integration.

Kein konkreter Exchange wird im MVP fest integriert.
Dieses Interface ermöglicht zukünftige CCXT- oder andere Adapter-Integrationen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExchangeAdapterError(Exception):
    """Fehler beim Austausch mit der Exchange."""


class RateLimitError(ExchangeAdapterError):
    """Request rate-limited by the exchange (HTTP 429)."""

    retry_after: float = 1.0


class ConnectionError(ExchangeAdapterError):
    """Network/connection error (5xx, timeout, DNS, SSL)."""


class AuthenticationError(ExchangeAdapterError):
    """Authentication failed (HTTP 401/403)."""


class ResponseValidationError(ExchangeAdapterError):
    """Exchange returned error code in response body."""

    code: str = ""
    message: str = ""

    def __init__(self, code: str = "", message: str = "") -> None:
        self.code = code
        self.message = message
        super().__init__(f"Response error {code}: {message}")


class ExchangeAdapter(ABC):
    """Abstrakte Basisklasse für Exchange-Adapter.

    Jedes konkrete Exchange-Interface (CCXT, GDAX, etc.) muss diese
    Schnittstelle implementieren. Im MVP wird nur die Stub-Implementierung
    verwendet.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name des Exchange-Adapters."""
        ...

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
        exchange_name: str | None = None,
    ) -> dict[str, Any]:
        """Order an die Exchange senden.

        Args:
            symbol: Trading symbol (e.g. "BTCUSDT")
            side: "BUY" or "SELL"
            quantity: Order quantity
            price: Order price (0 for market orders)
            order_type: "MARKET", "LIMIT", etc.
            exchange_name: Optional exchange name for routing

        Returns:
            Dict mit order_id, status, und optional error details

        Raises:
            ExchangeAdapterError: Wenn die Order nicht gesendet werden kann
        """
        ...

    @abstractmethod
    def get_order_status(self, order_id: str, exchange_name: str | None = None) -> dict[str, Any]:
        """Status einer Order abfragen.

        Args:
            order_id: ID der Order von der Exchange
            exchange_name: Optional exchange name for routing

        Returns:
            Dict mit filled_quantity, remaining_quantity, status
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str, exchange_name: str | None = None) -> dict[str, Any]:
        """Order stornieren.

        Args:
            order_id: ID der Order von der Exchange
            exchange_name: Optional exchange name for routing

        Returns:
            Dict mit success boolean und error details
        """
        ...

    @abstractmethod
    def get_balance(self, symbol: str) -> float:
        """Verfügbares Guthaben für ein Symbol abfragen.

        Args:
            symbol: Trading symbol

        Returns:
            Verfügbares Guthaben
        """
        ...

    @abstractmethod
    def get_ticker(self, symbol: str) -> dict[str, float]:
        """Aktuellen Marktpreis abfragen.

        Args:
            symbol: Trading symbol

        Returns:
            Dict mit bid, ask, last price
        """
        ...


class StubExchangeAdapter(ExchangeAdapter):
    """Stub-Implementierung — gibt NOT_IMPLEMENTED zurück.

    Wird im MVP verwendet und ersetzt durch echte Adapter später.
    """

    @property
    def name(self) -> str:
        return "STUB"

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
        exchange_name: str | None = None,
    ) -> dict[str, Any]:
        return {
            "order_id": None,
            "status": "NOT_IMPLEMENTED",
            "error": "NO_EXCHANGE_ADAPTER_IMPLEMENTED",
        }

    def get_order_status(self, order_id: str, exchange_name: str | None = None) -> dict[str, Any]:
        return {"status": "NOT_IMPLEMENTED", "error": "NO_EXCHANGE_ADAPTER_IMPLEMENTED"}

    def cancel_order(self, order_id: str, exchange_name: str | None = None) -> dict[str, Any]:
        return {"success": False, "error": "NO_EXCHANGE_ADAPTER_IMPLEMENTED"}

    def get_balance(self, symbol: str) -> float:
        return 0.0

    def get_ticker(self, symbol: str) -> dict[str, float]:
        return {"bid": 0.0, "ask": 0.0, "last": 0.0}
