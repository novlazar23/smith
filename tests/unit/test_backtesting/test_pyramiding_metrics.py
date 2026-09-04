"""Tests für allow_pyramiding (Flatsize) und korrekte Trade-Return-Metriken."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from packages.backtesting.core import BacktestConfig, BacktestResult, Candle
from packages.backtesting.datafeed import MemoryDataFeed
from packages.backtesting.engine import BacktestEngine
from packages.backtesting.strategies import BaseStrategy, SignalAction, StrategySignal


def _make_candles(closes: list[float]) -> list[Candle]:
    base = datetime(2021, 1, 1, tzinfo=UTC)
    return [
        Candle(
            timestamp=base + timedelta(minutes=i),
            symbol="BTC/USD",
            open=close,
            high=close * 1.001,
            low=close * 0.999,
            close=close,
            volume=100.0,
        )
        for i, close in enumerate(closes)
    ]


class ScriptedStrategy(BaseStrategy):
    """Gibt BUY an festen Bar-Indizes (auch bei offener Position) und SELL am Ende."""

    def __init__(self, buy_bars: frozenset[int], sell_bars: frozenset[int]) -> None:
        super().__init__(name="scripted")
        self._buys = buy_bars
        self._sells = sell_bars
        self._i = 0

    def on_bar(self, candle: Candle) -> StrategySignal | None:
        i = self._i
        self._i += 1
        action = (
            SignalAction.BUY if i in self._buys else SignalAction.SELL if i in self._sells else SignalAction.HOLD
        )
        return StrategySignal(
            action=action,
            symbol=candle.symbol,
            confidence=0.5,
            reason=f"scripted:{action.value}",
            position_size=0.1,
            timestamp=candle.timestamp,
        )


def _run(buy_bars: frozenset[int], sell_bars: frozenset[int], *, allow_pyramiding: bool) -> BacktestResult:
    # 8 Bars zu 100, dann SELL-Phase zu 110
    feed = MemoryDataFeed(candles=_make_candles([100.0] * 8 + [110.0, 110.0]))
    cfg = BacktestConfig(symbol="BTC/USD", warmup_bars=0, allow_pyramiding=allow_pyramiding)
    result = BacktestEngine(config=cfg).run(feed, ScriptedStrategy(buy_bars, sell_bars), warmup_bars=0)
    return result


def test_pyramiding_default_stacks_position() -> None:
    result = _run(frozenset({2, 5}), frozenset({8}), allow_pyramiding=True)
    rts = result.metadata["round_trips"]
    assert len(rts) == 1
    # Zweiter BUY verdoppelt die Position (10 % von Equity zweimal)
    assert rts[0]["quantity"] > 150


def test_no_pyramiding_keeps_flat_position() -> None:
    flat = _run(frozenset({2, 5}), frozenset({8}), allow_pyramiding=False)
    pyramided = _run(frozenset({2, 5}), frozenset({8}), allow_pyramiding=True)
    flat_rts = flat.metadata["round_trips"]
    pyr_rts = pyramided.metadata["round_trips"]
    assert len(flat_rts) == 1
    # Flatsize: nur der erste BUY zählt, zweiter wird ignoriert
    assert flat_rts[0]["quantity"] < 0.6 * pyr_rts[0]["quantity"]
    # Fills: 1 BUY + 1 SELL statt 2 BUYs + 1 SELL
    assert len(flat.trades) == 2
    assert len(pyramided.trades) == 3


def test_trade_return_pct_is_fraction_of_notional() -> None:
    result = _run(frozenset({2}), frozenset({8}), allow_pyramiding=False)
    m = result.metrics
    # 100 → 110 (vor Slippage) ≈ +10 % des Notentials — nicht $-PnL mal 100
    assert 9.5 < m["avg_trade_return_pct"] < 10.5
    assert 9.5 < m["best_trade_return_pct"] < 10.5
    assert 9.5 < m["avg_win_return_pct"] < 10.5
    assert m["total_trades"] == 1
    assert m["win_rate_pct"] == 100.0


def test_trade_return_pct_negative_on_loss() -> None:
    # 110 → 100: ≈ -9,1 % des Notentials
    feed = MemoryDataFeed(candles=_make_candles([110.0] * 8 + [100.0, 100.0]))
    cfg = BacktestConfig(symbol="BTC/USD", warmup_bars=0)
    result = BacktestEngine(config=cfg).run(
        feed, ScriptedStrategy(frozenset({2}), frozenset({8})), warmup_bars=0
    )
    m = result.metrics
    assert -9.6 < m["avg_trade_return_pct"] < -8.6
    assert m["win_rate_pct"] == 0.0
    assert m["profit_factor"] == 0.0
