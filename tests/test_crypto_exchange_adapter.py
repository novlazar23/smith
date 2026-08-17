"""Tests für Bybit & Bitget Crypto Exchange Adapters."""

from __future__ import annotations

from trading_harness.services.crypto_exchange_adapter import (
    BitgetExchangeAdapter,
    BybitExchangeAdapter,
)


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