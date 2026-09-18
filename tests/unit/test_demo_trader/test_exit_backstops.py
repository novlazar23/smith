"""Tests für die deterministischen Exit-Backstops des Demo-Traders.

Stop-Loss, Max-Haltezeit und Flat-Size — die Backstops überschreiben
den Trade-Plan (kein Agenten-Vote nötig/erlaubt).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from apps.demo_trader.service import (
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_STOP_LOSS_PCT,
    DemoTrader,
    DemoTraderConfig,
)
from apps.orchestrator_service.service import CandleWindow
from packages.paper import TradeDirection

from .conftest import BTC, FakeConnection, StubCandleSource, StubPipeline, make_result, make_trader


def _flat_window(price: float, n: int = 50) -> CandleWindow:
    """Flacher Kurs mit kleiner fester Halbbreite ±0,5 (ATR > 0, damit das
    ATR-basierte Cost-Margin-Gate durchläuft; Exit-Backstops prüfen Close)."""
    close = np.full(n, price)
    return CandleWindow(
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=np.full(n, 1000.0),
        timestamps=np.arange(n, dtype=np.int64) * 300_000_000_000,
    )


def _build(
    window: CandleWindow,
    *,
    decision: str = "RANGE",
    flat_size: bool = True,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    max_holding_hours: float = DEFAULT_MAX_HOLDING_HOURS,
) -> tuple:
    config = DemoTraderConfig(
        instruments=(BTC,),
        stop_loss_pct=stop_loss_pct,
        max_holding_hours=max_holding_hours,
        flat_size=flat_size,
    )
    pipeline = StubPipeline({BTC: make_result(decision, confidence=0.9)})
    conn = FakeConnection()
    trader = make_trader(config, StubCandleSource({BTC: window}), conn, pipeline)
    return trader, pipeline, conn


def _open_position(trader: DemoTrader, quantity: float, price: float) -> None:
    trader._executor.submit_order(trader.account, BTC, TradeDirection.BUY, quantity, price)


class TestStopLoss:
    def test_stop_loss_closes_position_and_skips_pipeline(self) -> None:
        # Position avg 100, Close 91 (-9 %) -> Stop 8 % feuert, obwohl der
        # Konsens (LONG_BIAS) einen Kauf signalisieren würde.
        trader, pipeline, _conn = _build(_flat_window(91.0), decision="LONG_BIAS")
        _open_position(trader, 0.1, 100.0)

        executed = trader.run_cycle()

        assert executed == 1
        assert BTC not in trader.account.positions
        assert pipeline.calls == []

    def test_no_exit_when_close_above_stop(self) -> None:
        # Close 99 (-1 %) -> kein Stop; RANGE-Entscheidung -> kein Trade.
        trader, pipeline, _conn = _build(_flat_window(99.0))
        _open_position(trader, 0.1, 100.0)

        executed = trader.run_cycle()

        assert executed == 0
        assert BTC in trader.account.positions
        assert len(pipeline.calls) == 1

    def test_stop_loss_disabled_with_zero(self) -> None:
        trader, pipeline, _conn = _build(_flat_window(91.0), stop_loss_pct=0.0)
        _open_position(trader, 0.1, 100.0)

        executed = trader.run_cycle()

        assert executed == 0  # RANGE → kein Trade, Stop deaktiviert
        assert BTC in trader.account.positions
        assert len(pipeline.calls) == 1


class TestMaxHolding:
    def test_max_holding_closes_old_position(self) -> None:
        trader, pipeline, _conn = _build(_flat_window(101.0))
        _open_position(trader, 0.1, 100.0)
        trader.account.positions[BTC].opened_at = datetime.now(UTC) - timedelta(days=8)

        executed = trader.run_cycle()

        assert executed == 1
        assert BTC not in trader.account.positions
        assert pipeline.calls == []

    def test_fresh_position_not_closed_by_max_holding(self) -> None:
        trader, _pipeline, _conn = _build(_flat_window(101.0))
        _open_position(trader, 0.1, 100.0)

        executed = trader.run_cycle()

        assert executed == 0
        assert BTC in trader.account.positions


class TestFlatSize:
    def test_flat_size_blocks_addon_buy(self) -> None:
        trader, pipeline, _conn = _build(_flat_window(100.0), decision="LONG_BIAS")
        _open_position(trader, 0.1, 100.0)

        executed = trader.run_cycle()

        assert executed == 0
        assert trader.account.positions[BTC].quantity == pytest.approx(0.1)
        assert len(pipeline.calls) == 1  # Konsens lief, Buy wurde unterdrückt

    def test_pyramiding_allowed_when_flat_size_disabled(self) -> None:
        trader, _pipeline, _conn = _build(_flat_window(100.0), decision="LONG_BIAS", flat_size=False)
        _open_position(trader, 0.1, 100.0)

        executed = trader.run_cycle()

        assert executed == 1
        assert trader.account.positions[BTC].quantity > 0.1
