"""Tests für die Decision→Trade-Mapping (long-only) inklusive Konfidenz-Gate."""

from __future__ import annotations

import pytest
from apps.demo_trader.service import DemoTraderConfig, plan_trade

from .conftest import (
    BTC,
    ETH,
    FakeConnection,
    StubCandleSource,
    StubPipeline,
    make_ohlcv,
    make_result,
    make_trader,
)


class TestPlanTrade:
    """Reine Mapping-Funktion: decision + confidence → Plan."""

    def test_long_bias_above_gate_plans_buy(self) -> None:
        """LONG_BIAS über dem Gate → Market-Buy mit quantity = notional/close."""
        plan = plan_trade("LONG_BIAS", 0.6, 0.4, 100.0, 2000.0)
        assert plan.action == "BUY"
        assert plan.quantity == pytest.approx(20.0)
        assert plan.price == 100.0

    def test_long_bias_at_gate_plans_buy(self) -> None:
        """Konfidenz exakt am Gate (>=) → Trade wird geplant."""
        plan = plan_trade("LONG_BIAS", 0.4, 0.4, 100.0, 2000.0)
        assert plan.action == "BUY"

    def test_long_bias_below_gate_is_no_trade(self) -> None:
        """LONG_BIAS unter dem Gate → kein Trade mit Gate-Meldung."""
        plan = plan_trade("LONG_BIAS", 0.39, 0.4, 100.0, 2000.0)
        assert plan.action == "NONE"
        assert plan.reason == "Signal unter Konfidenz-Gate → kein Trade"

    def test_short_bias_above_gate_plans_sell(self) -> None:
        """SHORT_BIAS über dem Gate → Glattstellung (SELL)."""
        plan = plan_trade("SHORT_BIAS", 0.7, 0.4, 100.0, 2000.0)
        assert plan.action == "SELL"
        assert plan.reason == "SHORT_BIAS → Position glattstellen"

    def test_short_bias_below_gate_is_no_trade(self) -> None:
        """SHORT_BIAS unter dem Gate → kein Trade (Position bleibt offen)."""
        plan = plan_trade("SHORT_BIAS", 0.1, 0.4, 100.0, 2000.0)
        assert plan.action == "NONE"
        assert plan.reason == "Signal unter Konfidenz-Gate → kein Trade"

    def test_no_trade_decision_is_noop(self) -> None:
        """NO_TRADE → immer kein Trade, egal wie hoch die Konfidenz."""
        plan = plan_trade("NO_TRADE", 0.9, 0.4, 100.0, 2000.0)
        assert plan.action == "NONE"
        assert plan.reason == "NO_TRADE → kein Trade"

    def test_range_decision_is_noop(self) -> None:
        """RANGE (sonstige Entscheidung) → kein Trade."""
        plan = plan_trade("RANGE", 0.9, 0.4, 100.0, 2000.0)
        assert plan.action == "NONE"

    def test_long_bias_with_invalid_close_is_noop(self) -> None:
        """LONG_BIAS mit nicht-positivem Schlusskurs → kein Trade."""
        plan = plan_trade("LONG_BIAS", 0.9, 0.4, 0.0, 2000.0)
        assert plan.action == "NONE"


class TestDecisionToPaperTrade:
    """End-to-End durch DemoTrader._run_instrument mit echtem PaperExecutor."""

    def test_long_bias_executes_buy_and_persists(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """LONG_BIAS → Paper-Buy, Position entsteht, Trade-Zeile wird persistiert."""
        window = make_ohlcv(100, start_price=100.0)
        provider = StubCandleSource({BTC: window})
        pipeline = StubPipeline({BTC: make_result(decision="LONG_BIAS", confidence=0.6)})
        trader = make_trader(config, provider, fake_conn, pipeline)

        executed = trader._run_instrument(BTC)

        assert executed == 1
        last_close = float(window.close[-1])
        params = fake_conn.executed[0][1]
        assert params["instrument"] == BTC
        assert params["direction"] == "BUY"
        assert params["quantity"] == pytest.approx(2000.0 / last_close)
        # Slippage 0.1 % → Fill über dem Markt-Preis
        assert params["filled_price"] > last_close
        assert params["status"] == "filled"
        assert BTC in trader.account.positions

    def test_short_bias_flattens_open_position(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """LONG_BIAS kauft ein, SHORT_BIAS stellt die Position glatt."""
        provider = StubCandleSource({BTC: make_ohlcv(100, start_price=100.0)})
        pipeline = StubPipeline({BTC: make_result(decision="LONG_BIAS", confidence=0.6)})
        trader = make_trader(config, provider, fake_conn, pipeline)

        assert trader._run_instrument(BTC) == 1
        assert BTC in trader.account.positions

        pipeline.results = {BTC: make_result(decision="SHORT_BIAS", confidence=0.6)}
        assert trader._run_instrument(BTC) == 1

        assert BTC not in trader.account.positions
        sell_params = fake_conn.executed[1][1]
        assert sell_params["direction"] == "SELL"
        assert fake_conn.commits == 2

    def test_short_bias_without_position_is_noop(
        self, config: DemoTraderConfig, fake_conn: FakeConnection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SHORT_BIAS ohne offene Position → kein Trade, Grund wird geloggt."""
        provider = StubCandleSource({BTC: make_ohlcv(100, start_price=100.0)})
        pipeline = StubPipeline({BTC: make_result(decision="SHORT_BIAS", confidence=0.6)})
        trader = make_trader(config, provider, fake_conn, pipeline)

        with caplog.at_level("INFO"):
            executed = trader._run_instrument(BTC)

        assert executed == 0
        assert fake_conn.executed == []
        assert "keine offene Position" in caplog.text

    def test_confidence_gate_blocks_buy(
        self, config: DemoTraderConfig, fake_conn: FakeConnection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Signal unter dem Konfidenz-Gate → kein Trade, Gate wird geloggt."""
        provider = StubCandleSource({BTC: make_ohlcv(100, start_price=100.0)})
        pipeline = StubPipeline({BTC: make_result(decision="LONG_BIAS", confidence=0.39)})
        trader = make_trader(config, provider, fake_conn, pipeline)

        with caplog.at_level("INFO"):
            executed = trader._run_instrument(BTC)

        assert executed == 0
        assert fake_conn.executed == []
        assert BTC not in trader.account.positions
        assert "Signal unter Konfidenz-Gate → kein Trade" in caplog.text

    def test_no_trade_decision_is_noop(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """NO_TRADE → weder Trade noch Persistenz."""
        provider = StubCandleSource({BTC: make_ohlcv(100, start_price=100.0)})
        pipeline = StubPipeline({BTC: make_result(decision="NO_TRADE", confidence=0.9)})
        trader = make_trader(config, provider, fake_conn, pipeline)

        executed = trader._run_instrument(BTC)

        assert executed == 0
        assert fake_conn.executed == []
        assert BTC not in trader.account.positions

    def test_position_persists_across_instruments(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """Kauf bei BTC bleibt beim Durchlauf für ETH unverändert erhalten."""
        provider = StubCandleSource(
            {BTC: make_ohlcv(100, start_price=100.0), ETH: make_ohlcv(100, start_price=50.0)}
        )
        pipeline = StubPipeline(
            {
                BTC: make_result(decision="LONG_BIAS", confidence=0.6),
                ETH: make_result(decision="NO_TRADE", confidence=0.0),
            }
        )
        trader = make_trader(config, provider, fake_conn, pipeline)

        assert trader._run_instrument(BTC) == 1
        assert trader._run_instrument(ETH) == 0

        assert BTC in trader.account.positions
        assert ETH not in trader.account.positions
        assert len(fake_conn.executed) == 1
