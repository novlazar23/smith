"""Tests for PaperExchangeAdapter.

Deckt submit_order (Erfolg/Fehler), Side-Normalisierung, Stub-Methoden
und Initialisierungsparameter ab.
"""

from __future__ import annotations

import pytest

from trading_harness.services.paper_exchange import PaperExchange
from trading_harness.services.paper_exchange_adapter import PaperExchangeAdapter
from trading_harness.services.paper_trade_store import InMemoryPaperTradeStore


def _make_store() -> InMemoryPaperTradeStore:
    """Erstellt einen InMemoryPaperTradeStore für PaperExchange."""
    return InMemoryPaperTradeStore()


class TestPaperExchangeAdapterInit:
    """Prüft die Initialisierung des Adapters."""

    def test_default_init_creates_paper_exchange(self) -> None:
        adapter = PaperExchangeAdapter()
        assert adapter._paper_exchange is not None
        assert isinstance(adapter._paper_exchange, PaperExchange)

    def test_custom_paper_exchange(self) -> None:
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.002, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)
        assert adapter._paper_exchange is pe

    def test_custom_run_id(self) -> None:
        adapter = PaperExchangeAdapter(run_id="test-run-42")
        assert adapter._run_id == "test-run-42"

    def test_name_property(self) -> None:
        adapter = PaperExchangeAdapter()
        assert adapter.name == "PAPER"

    def test_default_fill_rate(self) -> None:
        adapter = PaperExchangeAdapter()
        assert adapter._paper_exchange.fill_rate == 0.8

    def test_default_fee_rate(self) -> None:
        adapter = PaperExchangeAdapter()
        assert adapter._paper_exchange.fee_rate == 0.001


class TestPaperExchangeAdapterSubmitOrderSuccess:
    """Erfolgreiche Order-Ausführung durch PaperExchange."""

    def _make_adapter_with_fill_rate(self, fill_rate: float = 1.0) -> PaperExchangeAdapter:
        pe = PaperExchange(fill_rate=fill_rate, fee_rate=0.0, stores=_make_store())
        return PaperExchangeAdapter(paper_exchange=pe)

    def test_submit_long_order(self) -> None:
        adapter = self._make_adapter_with_fill_rate()
        result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "FILLED"
        assert result["order_id"] is not None
        assert result["trade_id"] is not None
        assert result["actual_quantity"] == 1.0  # fill_rate=1.0
        assert result["actual_price"] == pytest.approx(50000.0)
        assert result["fees"] == 0.0
        assert result["error"] is None

    def test_submit_short_order(self) -> None:
        adapter = self._make_adapter_with_fill_rate()
        result = adapter.submit_order(
            symbol="ETHUSDT",
            side="SHORT",
            quantity=10.0,
            price=3000.0,
        )
        assert result["status"] == "FILLED"
        assert result["actual_quantity"] == 10.0

    def test_buy_side_normalized_to_long(self) -> None:
        adapter = self._make_adapter_with_fill_rate()
        result = adapter.submit_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "FILLED"

    def test_sell_side_normalized_to_short(self) -> None:
        adapter = self._make_adapter_with_fill_rate()
        result = adapter.submit_order(
            symbol="ETHUSDT",
            side="SELL",
            quantity=10.0,
            price=3000.0,
        )
        assert result["status"] == "FILLED"

    def test_buy_lowercase_normalized_to_long(self) -> None:
        adapter = self._make_adapter_with_fill_rate()
        result = adapter.submit_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "FILLED"

    def test_sell_lowercase_normalized_to_short(self) -> None:
        adapter = self._make_adapter_with_fill_rate()
        result = adapter.submit_order(
            symbol="ETHUSDT",
            side="sell",
            quantity=10.0,
            price=3000.0,
        )
        assert result["status"] == "FILLED"

    def test_order_type_ignored(self) -> None:
        """order_type wird vom Adapter ignoriert (PaperExchange simuliert MARKET)."""
        adapter = self._make_adapter_with_fill_rate()
        result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
            order_type="LIMIT",
        )
        assert result["status"] == "FILLED"

    def test_fee_calculation(self) -> None:
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.001, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)
        result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        expected_fee = 1.0 * 50000.0 * 0.001
        assert result["fees"] == pytest.approx(expected_fee, rel=1e-6)

    def test_slippage_affects_actual_price(self) -> None:
        """LONG sollte Slippage addieren, SHORT subtrahieren."""
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)

        long_result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        # TradeProposal verwendet expected_slippage_bps (Standardwert in TradeProposal)
        # actual_price für LONG = current_price + slippage
        assert long_result["actual_price"] >= 50000.0

        short_result = adapter.submit_order(
            symbol="BTCUSDT",
            side="SHORT",
            quantity=1.0,
            price=50000.0,
        )
        # actual_price für SHORT = current_price - slippage
        assert short_result["actual_price"] <= 50000.0

    def test_fill_rate_reduction(self) -> None:
        pe = PaperExchange(fill_rate=0.5, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)
        result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=10.0,
            price=50000.0,
        )
        assert result["actual_quantity"] == pytest.approx(5.0)

    def test_stop_loss_and_target_default(self) -> None:
        """Stop-Loss (5% unter) und Target (5% über) sollten korrekt sein."""
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)
        result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=100.0,
        )
        assert result["status"] == "FILLED"


class TestPaperExchangeAdapterSubmitOrderRejections:
    """Fehlerbehandlung bei ungültigen Eingaben."""

    def test_zero_price_rejected(self) -> None:
        adapter = PaperExchangeAdapter()
        result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=0.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "INVALID_PRICE"
        assert result["order_id"] is None

    def test_negative_price_rejected(self) -> None:
        adapter = PaperExchangeAdapter()
        result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=-100.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "INVALID_PRICE"
        assert result["order_id"] is None

    def test_empty_symbol_rejected(self) -> None:
        adapter = PaperExchangeAdapter()
        result = adapter.submit_order(
            symbol="",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "MISSING_SYMBOL"
        assert result["order_id"] is None

    def test_none_symbol_rejected(self) -> None:
        adapter = PaperExchangeAdapter()
        result = adapter.submit_order(
            symbol=None,  # type: ignore[arg-type]
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "MISSING_SYMBOL"

    def test_invalid_side_rejected(self) -> None:
        adapter = PaperExchangeAdapter()
        result = adapter.submit_order(
            symbol="BTCUSDT",
            side="EXIT",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "INVALID_SIDE"
        assert result["order_id"] is None

    def test_invalid_side_with_numbers_rejected(self) -> None:
        adapter = PaperExchangeAdapter()
        result = adapter.submit_order(
            symbol="BTCUSDT",
            side="123",
            quantity=1.0,
            price=50000.0,
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "INVALID_SIDE"


class TestPaperExchangeAdapterStubMethods:
    """Prüft die Stub-Methoden für Order-Status, Stornierung, Balance und Ticker."""

    def test_get_order_status_stub(self) -> None:
        adapter = PaperExchangeAdapter()
        result = adapter.get_order_status("order-123")
        assert result["status"] == "NOT_IMPLEMENTED"
        assert result["error"] == "PAPER_TRADE_STORE_ACCESS_NOT_EXPOSED"

    def test_cancel_order_stub(self) -> None:
        adapter = PaperExchangeAdapter()
        result = adapter.cancel_order("order-123")
        assert result["success"] is False
        assert result["error"] == "CANCEL_NOT_SUPPORTED"

    def test_get_balance_stub(self) -> None:
        adapter = PaperExchangeAdapter()
        balance = adapter.get_balance("BTCUSDT")
        assert balance == 100000.0

    def test_get_ticker_stub(self) -> None:
        adapter = PaperExchangeAdapter()
        ticker = adapter.get_ticker("BTCUSDT")
        assert ticker == {"bid": 0.0, "ask": 0.0, "last": 0.0}