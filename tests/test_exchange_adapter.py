"""Tests für ExchangeAdapter Interface."""

from __future__ import annotations

import pytest

from trading_harness.services.exchange_adapter import (
    ExchangeAdapter,
    ExchangeAdapterError,
    StubExchangeAdapter,
)


class TestExchangeAdapterInterface:
    """Interface-Tests für ExchangeAdapter."""

    def test_stub_has_name(self):
        """Stub-Adapter hat einen Namen."""
        adapter = StubExchangeAdapter()
        assert adapter.name == "STUB"

    def test_stub_submit_order(self):
        """Stub gibt NOT_IMPLEMENTED zurück."""
        adapter = StubExchangeAdapter()
        result = adapter.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert result["status"] == "NOT_IMPLEMENTED"
        assert "NO_EXCHANGE_ADAPTER_IMPLEMENTED" in result.get("error", "")

    def test_stub_get_order_status(self):
        """Stub für order status."""
        adapter = StubExchangeAdapter()
        result = adapter.get_order_status("order-123")
        assert result["status"] == "NOT_IMPLEMENTED"

    def test_stub_cancel_order(self):
        """Stub für cancel order."""
        adapter = StubExchangeAdapter()
        result = adapter.cancel_order("order-123")
        assert result["success"] is False

    def test_stub_get_balance(self):
        """Stub für balance."""
        adapter = StubExchangeAdapter()
        assert adapter.get_balance("BTCUSDT") == 0.0

    def test_stub_get_ticker(self):
        """Stub für ticker."""
        adapter = StubExchangeAdapter()
        result = adapter.get_ticker("BTCUSDT")
        assert result == {"bid": 0.0, "ask": 0.0, "last": 0.0}


class TestExchangeAdapterError:
    """ExchangeAdapterError-Tests."""

    def test_exception_inherits_from_exception(self):
        """ExchangeAdapterError ist eine Exception."""
        with pytest.raises(ExchangeAdapterError):
            raise ExchangeAdapterError("test error")

    def test_exception_message(self):
        """Exception trägt die Fehlermeldung."""
        try:
            raise ExchangeAdapterError("connection failed")
        except ExchangeAdapterError as e:
            assert str(e) == "connection failed"