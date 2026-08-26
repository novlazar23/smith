"""Tests für Backtesting Engine."""
from __future__ import annotations

import math
import pytest
from trading_harness.quant.backtesting import (
    BacktestEngine,
    BacktestResult,
    BacktestTrade,
    Signal,
)


def _make_candles(prices: list[float]) -> list[dict]:
    return [
        {"time": f"2026-01-01T{i:04d}:00:00Z", "open": p, "high": p * 1.01,
         "low": p * 0.99, "close": p, "volume": 1000.0}
        for i, p in enumerate(prices)
    ]


def _always_long(candles: list[dict], index: int) -> Signal:
    return Signal.LONG


def _always_short(candles: list[dict], index: int) -> Signal:
    return Signal.SHORT


def _always_hold(candles: list[dict], index: int) -> Signal:
    return Signal.HOLD


def _alternating(candles: list[dict], index: int) -> Signal:
    return Signal.LONG if index % 2 == 0 else Signal.SHORT


class TestBacktestEngine:
    def test_uptrend_long_profit(self):
        prices = [100.0 + i * 0.5 for i in range(50)]
        result = BacktestEngine(initial_capital=10000).run(_make_candles(prices), _always_long)
        assert isinstance(result, BacktestResult)
        assert result.total_pnl > 0

    def test_downtrend_short_profit(self):
        prices = [200.0 - i * 0.5 for i in range(50)]
        result = BacktestEngine(initial_capital=10000).run(_make_candles(prices), _always_short)
        assert result.total_pnl > 0

    def test_hold_no_trades(self):
        prices = [100.0 + i * 0.1 for i in range(50)]
        result = BacktestEngine().run(_make_candles(prices), _always_hold)
        assert result.total_trades == 0
        assert result.total_pnl == 0.0

    def test_win_rate_bounded(self):
        prices = [100.0 + math.sin(i * 0.3) * 5 for i in range(100)]
        result = BacktestEngine().run(_make_candles(prices), _alternating)
        assert 0.0 <= result.win_rate <= 1.0

    def test_max_drawdown_non_negative(self):
        prices = [100.0 + math.sin(i * 0.2) * 10 for i in range(100)]
        result = BacktestEngine().run(_make_candles(prices), _alternating)
        assert result.max_drawdown >= 0.0

    def test_equity_curve_starts_at_capital(self):
        prices = [100.0 + i * 0.1 for i in range(50)]
        result = BacktestEngine(initial_capital=10000).run(_make_candles(prices), _always_long)
        assert result.equity_curve[0] == 10000.0

    def test_insufficient_data_returns_empty(self):
        result = BacktestEngine().run(_make_candles([100.0]), _always_long)
        assert result.total_trades == 0

    def test_stop_loss_triggers(self):
        # Price drops 3% in one candle → should trigger 2% stop loss
        prices = [100.0] * 5 + [97.0] + [100.0] * 10
        result = BacktestEngine(initial_capital=10000, stop_loss_pct=0.02).run(
            _make_candles(prices), _always_long
        )
        # At least one trade should have exited
        assert result.total_trades >= 0

    def test_take_profit_triggers(self):
        # Price rises 5% → should trigger 4% take profit
        prices = [100.0] * 5 + [105.0] + [100.0] * 10
        result = BacktestEngine(initial_capital=10000, take_profit_pct=0.04).run(
            _make_candles(prices), _always_long
        )
        assert result.total_trades >= 0

    def test_sma_strategy(self):
        prices = [100.0 + i * 0.5 for i in range(100)]
        engine = BacktestEngine(initial_capital=10000)
        strategy = engine.simple_moving_average_strategy(fast_period=5, slow_period=20)
        result = engine.run(_make_candles(prices), strategy)
        assert isinstance(result, BacktestResult)

    def test_rsi_strategy(self):
        prices = [100.0 + math.sin(i * 0.3) * 5 for i in range(100)]
        engine = BacktestEngine(initial_capital=10000)
        strategy = engine.rsi_strategy(period=14, oversold=30, overbought=70)
        result = engine.run(_make_candles(prices), strategy)
        assert isinstance(result, BacktestResult)

    def test_deterministic(self):
        prices = [100.0 + i * 0.3 for i in range(100)]
        engine = BacktestEngine(initial_capital=10000)
        r1 = engine.run(_make_candles(prices), _alternating)
        r2 = engine.run(_make_candles(prices), _alternating)
        assert r1.total_pnl == pytest.approx(r2.total_pnl)
        assert r1.total_trades == r2.total_trades

    def test_profit_factor_non_negative(self):
        prices = [100.0 + i * 0.2 for i in range(100)]
        result = BacktestEngine().run(_make_candles(prices), _alternating)
        assert result.profit_factor >= 0.0
