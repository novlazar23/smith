"""Tests für die Demo-Trader-Prometheus-Metriken (update_metrics)."""

from __future__ import annotations

from datetime import UTC, datetime

from apps.demo_trader.metrics import REGISTRY, update_metrics
from apps.demo_trader.service import DemoTrader, DemoTraderConfig
from packages.paper import PaperPosition
from prometheus_client import generate_latest

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


def _trader_with_cycle(config: DemoTraderConfig, conn: FakeConnection) -> DemoTrader:
    """Builder + Zyklus mit Konfidenz 0.42 (BTC) bzw. 0.55 (ETH), ohne Trades."""
    pipeline = StubPipeline(
        {
            BTC: make_result("NO_TRADE", confidence=0.42),
            ETH: make_result("NO_TRADE", confidence=0.55),
        }
    )
    provider = StubCandleSource({BTC: make_ohlcv(50), ETH: make_ohlcv(50, start_price=3000.0)})
    trader = make_trader(config, provider, conn, pipeline)
    trader.run_cycle()
    return trader


def test_metrics_reflect_account_state(config: DemoTraderConfig, fake_conn: FakeConnection) -> None:
    """Nach einem Trade-losen Zyklus: 0 Positionen, 0 Drawdown, Konfidenzen pro Instrument."""
    trader = _trader_with_cycle(config, fake_conn)
    update_metrics(trader)
    text = generate_latest(REGISTRY).decode()
    assert 'trading_open_positions 0.0' in text
    assert 'trading_portfolio_drawdown 0.0' in text
    assert 'trading_config_max_open_positions 2.0' in text
    assert 'trading_signal_confidence{instrument="BTC/USDT"} 0.42' in text
    assert 'trading_signal_confidence{instrument="ETH/USDT"} 0.55' in text


def test_metrics_show_open_position_and_drawdown(config: DemoTraderConfig, fake_conn: FakeConnection) -> None:
    """Eine rote offene Position (Mark unter avg) zeigt Position + Drawdown."""
    trader = _trader_with_cycle(config, fake_conn)
    # 20 BTC @ 100 (Notional 2000): Cash wird belastet, Mark 90 -> uPnL -200
    trader._account.cash = 98000.0
    trader._account.positions[BTC] = PaperPosition(
        symbol=BTC,
        quantity=20.0,
        avg_price=100.0,
        mark_price=90.0,
        unrealized_pnl=-200.0,
        opened_at=datetime.now(UTC),
    )
    update_metrics(trader)
    text = generate_latest(REGISTRY).decode()
    assert 'trading_open_positions 1.0' in text
    # peak 100000 → equity 98000 + 1800 = 99800 → drawdown 0.002
    assert 'trading_portfolio_drawdown 0.002' in text
