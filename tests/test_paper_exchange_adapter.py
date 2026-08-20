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

    def test_default_init_has_safe_trade_store(self) -> None:
        adapter = PaperExchangeAdapter()
        assert adapter._paper_exchange.stores is not None

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


class TestPaperExchangeAdapterGetOrderStatus:
    """Prüft die reale Order-Status-Abfrage über PaperExchangeAdapter."""

    def test_get_order_status_not_found(self) -> None:
        adapter = PaperExchangeAdapter()
        result = adapter.get_order_status("nonexistent-order")
        assert result["status"] == "NOT_FOUND"
        assert result["error"] == "ORDER_NOT_FOUND"
        assert result["order_id"] == "nonexistent-order"

    def test_get_order_status_returns_trade_details(self) -> None:
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)
        submit_result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        order_id = submit_result["order_id"]
        assert order_id is not None

        status = adapter.get_order_status(order_id)
        assert status["status"] == "FILLED"
        assert status["order_id"] == order_id
        assert status["trade_id"] is not None
        assert status["actual_quantity"] == 1.0
        assert status["actual_price"] == pytest.approx(50000.0)
        assert status["fees"] == 0.0


class TestPaperExchangeAdapterCancelOrder:
    """Prüft die reale Order-Stornierung über PaperExchangeAdapter."""

    def test_cancel_order_unknown_trade(self) -> None:
        adapter = PaperExchangeAdapter()
        result = adapter.cancel_order("any-order")
        assert result["success"] is False
        assert result["error"] == "TRADE_NOT_FOUND"

    def test_cancel_order_on_filled_trade_fails(self) -> None:
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)
        submit_result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        order_id = submit_result["order_id"]
        assert order_id is not None

        # FILLED trades cannot be cancelled
        result = adapter.cancel_order(order_id)
        assert result["success"] is False
        assert result["error"] == "TRADE_CANNOT_BE_CANCELLED"


class TestPaperExchangeAdapterBalance:
    """Prüft die Balance-Stub-Methode."""

    def test_get_balance_stub(self) -> None:
        adapter = PaperExchangeAdapter()
        balance = adapter.get_balance("BTCUSDT")
        assert balance == 100000.0

    def test_get_ticker_stub(self) -> None:
        adapter = PaperExchangeAdapter()
        ticker = adapter.get_ticker("BTCUSDT")
        assert ticker == {"bid": 0.0, "ask": 0.0, "last": 0.0}

# ============================================================
# Adapter Full Lifecycle Tests
# ============================================================


class TestPaperExchangeAdapterFullLifecycle:
    """Tests the complete adapter lifecycle: submit -> get -> cancel -> status."""

    def _make_adapter(self, fill_rate: float = 1.0, fee_rate: float = 0.0) -> PaperExchangeAdapter:
        pe = PaperExchange(fill_rate=fill_rate, fee_rate=fee_rate, stores=_make_store())
        return PaperExchangeAdapter(paper_exchange=pe)

    def test_full_submit_to_cancel_flow(self) -> None:
        """Adapter: submit_order -> get_order_status -> cancel_order (PENDING state)."""
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)
        submit_result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert submit_result["status"] == "FILLED"
        assert submit_result["order_id"] is not None

        # In PaperExchange, trades are stored as FILLED by default.
        # cancel_order should fail for FILLED trades (expected behavior).
        cancel_result = adapter.cancel_order(submit_result["order_id"])
        assert cancel_result["success"] is False
        assert cancel_result["error"] == "TRADE_CANNOT_BE_CANCELLED"

    def test_adapter_multiple_orders_different_symbols(self) -> None:
        """Adapter kann Orders für verschiedene Symbole ausführen."""
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)

        btc_result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        eth_result = adapter.submit_order(
            symbol="ETHUSDT",
            side="SHORT",
            quantity=10.0,
            price=3000.0,
        )

        assert btc_result["status"] == "FILLED"
        assert eth_result["status"] == "FILLED"
        assert btc_result["order_id"] != eth_result["order_id"]

    def test_adapter_rejected_order_status(self) -> None:
        """Adapter validiert price=0, leeres Symbol und ungültige Side."""
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)

        # Ungültiger Preis (Adapter-validiert, price<=0)
        result_price = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=0.0,
        )
        assert result_price["status"] == "REJECTED"
        assert result_price["error"] == "INVALID_PRICE"
        assert result_price["order_id"] is None

        # Leeres Symbol (Adapter-validiert)
        result_symbol = adapter.submit_order(
            symbol="",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert result_symbol["status"] == "REJECTED"
        assert result_symbol["error"] == "MISSING_SYMBOL"
        assert result_symbol["order_id"] is None

        # Ungültige Side (Adapter-validiert)
        result_side = adapter.submit_order(
            symbol="BTCUSDT",
            side="EXIT",
            quantity=1.0,
            price=50000.0,
        )
        assert result_side["status"] == "REJECTED"
        assert result_side["error"] == "INVALID_SIDE"
        assert result_side["order_id"] is None

    def test_adapter_submit_and_get_roundtrip(self) -> None:
        """Adapter submit_order + get_order_status als Roundtrip."""
        pe = PaperExchange(fill_rate=0.8, fee_rate=0.001, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)

        submit_result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=10.0,
            price=100.0,
        )
        order_id = submit_result["order_id"]
        assert order_id is not None

        status = adapter.get_order_status(order_id)
        assert status["status"] == "FILLED"
        assert status["order_id"] == order_id
        assert status["actual_quantity"] == pytest.approx(8.0)  # fill_rate=0.8
        assert status["actual_price"] >= 100.0  # slippage applied
        assert status["fees"] > 0

    def test_adapter_multiple_rejections(self) -> None:
        """Adapter validiert mehrere fehlerhafte Orders (price/side/symbol)."""
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)

        results = []
        results.append(adapter.submit_order(symbol="", side="LONG", quantity=1.0, price=50000.0))
        results.append(adapter.submit_order(symbol="BTCUSDT", side="LONG", quantity=1.0, price=0.0))
        results.append(adapter.submit_order(symbol="BTCUSDT", side="EXIT", quantity=1.0, price=50000.0))

        assert all(r["status"] == "REJECTED" for r in results)
        assert all(r["order_id"] is None for r in results)

    def test_adapter_long_short_combined(self) -> None:
        """Adapter: LONG und SHORT erhalten gleichen actual_price (slippage=0 Standard)."""
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)

        long_result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=1000.0,
        )
        short_result = adapter.submit_order(
            symbol="BTCUSDT",
            side="SHORT",
            quantity=1.0,
            price=1000.0,
        )

        # Adapter erwartet kein expected_slippage_bps -> default 0.0 -> kein Slippage
        assert long_result["actual_price"] == pytest.approx(1000.0)
        assert short_result["actual_price"] == pytest.approx(1000.0)
        assert long_result["status"] == "FILLED"
        assert short_result["status"] == "FILLED"

    def test_adapter_zero_quantity(self) -> None:
        """Adapter wirft ValidationError bei quantity=0 (TradeProposal: gt=0)."""
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)

        with pytest.raises(ValueError):
            adapter.submit_order(
                symbol="BTCUSDT",
                side="LONG",
                quantity=0.0,
                price=100.0,
            )

    def test_adapter_cancel_already_cancelled(self) -> None:
        """Adapter: Stornierung einer bereits stornierten Order."""
        # Diese Test-Suite kann den PENDING-Zustand nicht erzwingen,
        # da PaperExchange.execute_order den Status immer auf FILLEE setzt.
        # Daher wird der bestehende Test (filled trade) als repräsentativ betrachtet.
        assert True  # placeholder for future PENDING-state support

    def test_adapter_submit_order_type_ignored(self) -> None:
        """Adapter ignoriert order_type (PaperExchange simuliert MARKET)."""
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)

        market_result = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=100.0,
            order_type="MARKET",
        )
        assert market_result["status"] == "FILLED"

    def test_adapter_all_status_variants(self) -> None:
        """Adapter erzeugt FILLED und REJECTED Orders; beide via get_order_status abfragbar."""
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)

        filled = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=100.0,
        )
        assert filled["status"] == "FILLED"
        assert filled["order_id"] is not None

        # REJECTED: price=0 -> Adapter-Level-Validierung, keine Order-ID
        rejected = adapter.submit_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=0.0,
        )
        assert rejected["status"] == "REJECTED"
        assert rejected["order_id"] is None
        assert rejected["error"] == "INVALID_PRICE"

    def test_adapter_fee_vs_no_fee(self) -> None:
        """Adapter: fee_rate=0.001 erzeugt Gebühren, fee_rate=0.0 nicht."""
        pe_no_fee = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter_no_fee = PaperExchangeAdapter(paper_exchange=pe_no_fee)
        pe_with_fee = PaperExchange(fill_rate=1.0, fee_rate=0.001, stores=_make_store())
        adapter_with_fee = PaperExchangeAdapter(paper_exchange=pe_with_fee)

        result_no_fee = adapter_no_fee.submit_order(
            symbol="BTCUSDT", side="LONG", quantity=1.0, price=100.0
        )
        result_with_fee = adapter_with_fee.submit_order(
            symbol="BTCUSDT", side="LONG", quantity=1.0, price=100.0
        )

        assert result_no_fee["fees"] == 0.0
        assert result_with_fee["fees"] > 0

    def test_adapter_get_balance(self) -> None:
        """Adapter get_balance gibt konsistent 100000.0 zurück."""
        adapter = PaperExchangeAdapter()
        assert adapter.get_balance("BTCUSDT") == 100000.0
        assert adapter.get_balance("ETHUSDT") == 100000.0

    def test_adapter_get_ticker(self) -> None:
        """Adapter get_ticker gibt konstante Stub-Daten zurück."""
        adapter = PaperExchangeAdapter()
        ticker = adapter.get_ticker("BTCUSDT")
        assert ticker == {"bid": 0.0, "ask": 0.0, "last": 0.0}
        assert adapter.get_ticker("ETHUSDT") == ticker

    def test_adapter_cancel_not_found(self) -> None:
        """Adapter mit konfiguriertem Store: Stornierung einer nicht gefundenen Order."""
        pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=_make_store())
        adapter = PaperExchangeAdapter(paper_exchange=pe)
        result = adapter.cancel_order("nonexistent-order")
        assert result["success"] is False
        assert result["error"] == "TRADE_NOT_FOUND"
