"""Tests für ShadowModeLogger und ShadowModeAdapter."""

from __future__ import annotations

from trading_harness.services.shadow_mode_logger import (
    ShadowModeAdapter,
    ShadowModeLogger,
)


class TestShadowModeLogger:
    """ShadowModeLogger Tests."""

    def test_log_order_creates_record(self):
        """Loggt Order und erstellt Record."""
        logger = ShadowModeLogger()
        record = logger.log_order(
            decision_id="dec-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert record.decision_id == "dec-1"
        assert record.symbol == "BTCUSDT"
        assert record.side == "LONG"
        assert record.quantity == 1.0
        assert record.price == 50000.0
        assert record.simulated_status == "FILLED"

    def test_log_order_creates_multiple(self):
        """Mehrere Orders werden geloggt."""
        logger = ShadowModeLogger()
        logger.log_order("dec-1", "BTCUSDT", "LONG", 1.0, 50000.0)
        logger.log_order("dec-2", "ETHUSDT", "SHORT", 2.0, 3000.0)
        assert logger.record_count == 2

    def test_get_records_filters(self):
        """Records können nach decision_id gefiltert werden."""
        logger = ShadowModeLogger()
        logger.log_order("dec-1", "BTCUSDT", "LONG", 1.0, 50000.0)
        logger.log_order("dec-2", "BTCUSDT", "SHORT", 1.0, 50000.0)
        logger.log_order("dec-3", "ETHUSDT", "LONG", 1.0, 3000.0)

        records = logger.get_records(decision_id="dec-2")
        assert len(records) == 1
        assert records[0].symbol == "BTCUSDT"

    def test_get_records_by_symbol(self):
        """Records können nach Symbol gefiltert werden."""
        logger = ShadowModeLogger()
        logger.log_order("dec-1", "BTCUSDT", "LONG", 1.0, 50000.0)
        logger.log_order("dec-2", "ETHUSDT", "SHORT", 2.0, 3000.0)

        records = logger.get_records(symbol="BTCUSDT")
        assert len(records) == 1
        assert records[0].symbol == "BTCUSDT"

    def test_commission_calculation(self):
        """Commission wird basierend auf Fill-Preis berechnet."""
        logger = ShadowModeLogger(default_commission=0.001)
        record = logger.log_order(
            decision_id="dec-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        # Commission = fill_price * quantity * commission
        # fill_price = 50000 + (50000 * 0.0005) = 50025
        assert abs(record.simulated_commission - 50.025) < 0.001

    def test_slippage_calculation(self):
        """Slippage wird korrekt berechnet."""
        logger = ShadowModeLogger(default_slippage=0.0005)
        record = logger.log_order(
            decision_id="dec-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        expected_slippage = 50000.0 * 0.0005
        assert record.simulated_slippage == expected_slippage

    def test_pnl_estimate(self):
        """Geschätzter PnL wird korrekt berechnet."""
        logger = ShadowModeLogger(
            default_slippage=0.0,
            default_commission=0.0,
            auto_fill_price=False,
        )
        record = logger.log_order(
            decision_id="dec-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            price=50000.0,
        )
        assert record.pnl_estimate == 0.0

    def test_summary(self):
        """Zusammenfassung aller Orders."""
        logger = ShadowModeLogger()
        logger.log_order("dec-1", "BTCUSDT", "LONG", 1.0, 50000.0)
        logger.log_order("dec-2", "ETHUSDT", "SHORT", 2.0, 3000.0)

        summary = logger.summary()
        assert summary["total_orders"] == 2
        assert summary["filled"] == 2
        assert summary["rejected"] == 0
        assert summary["total_slippage"] > 0
        assert summary["total_commission"] > 0

    def test_total_commission(self):
        """Gesamt-Commission (inkl. Slippage-Effekt)."""
        logger = ShadowModeLogger(default_commission=0.001)
        logger.log_order("dec-1", "BTCUSDT", "LONG", 1.0, 50000.0)
        logger.log_order("dec-2", "BTCUSDT", "LONG", 2.0, 50000.0)
        # Commission = fill_price * quantity * 0.001
        # fill_price = 50025 (with slippage)
        # commission1 = 50.025, commission2 = 100.05
        assert abs(logger.total_commission - 150.075) < 0.001

    def test_total_slippage(self):
        """Gesamt-Slippage."""
        logger = ShadowModeLogger(default_slippage=0.0005)
        logger.log_order("dec-1", "BTCUSDT", "LONG", 1.0, 50000.0)
        logger.log_order("dec-2", "BTCUSDT", "LONG", 1.0, 50000.0)
        assert logger.total_slippage == 50.0


class TestShadowModeAdapter:
    """ShadowModeAdapter Tests."""

    def test_shadow_submit_order(self):
        """ShadowModeAdapter loggt Order."""
        adapter = ShadowModeAdapter()
        result = adapter.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert result["status"] == "FILLED"
        assert result["order_id"] is not None

    def test_shadow_get_balance(self):
        """ShadowModeAdapter gibt simulierte Balance."""
        adapter = ShadowModeAdapter()
        balance = adapter.get_balance("BTCUSDT")
        assert balance == 100000.0

    def test_shadow_get_ticker(self):
        """ShadowModeAdapter gibt simulierte Ticker-Daten."""
        adapter = ShadowModeAdapter()
        ticker = adapter.get_ticker("BTCUSDT")
        assert ticker["bid"] == 50000.0
        assert ticker["ask"] == 50001.0
        assert ticker["last"] == 50000.5

    def test_shadow_order_in_logger(self):
        """ShadowModeAdapter loggt Order im Logger."""
        logger = ShadowModeLogger()
        adapter = ShadowModeAdapter(shadow=logger)
        adapter.submit_order("BTCUSDT", "BUY", 1.0, 50000.0)
        assert logger.record_count == 1