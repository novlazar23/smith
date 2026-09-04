"""Tests für die Regel-Strategien: Warmup-Stille, Signal-Verhalten, Felder, Determinismus."""

from __future__ import annotations

from datetime import timedelta

import pytest
from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction
from packages.strategies import create_strategy, list_strategies
from packages.strategies._common import RuleStrategy
from tests.unit.test_strategies.conftest import BASE_TIME, BTC, make_candles, make_sine_candles

UP = SignalAction.BUY
DN = SignalAction.SELL


def _feed(strategy: RuleStrategy, candles: list[Candle]) -> list:
    """Füttert Kerzen barweise und sammelt alle Signale."""
    signals = []
    for candle in candles:
        signal = strategy.on_bar(candle)
        if signal is not None:
            signals.append(signal)
    return signals


def _phases(
    n_flat: int,
    n_up: int,
    step_up: float,
    n_down: int = 0,
    step_down: float = -0.8,
    start_price: float = 100.0,
) -> list[Candle]:
    """Flat → Up (→ Down) Phasen-Kerzen für Crossover-Tests."""
    candles = make_candles(n_flat)
    if n_flat:
        start_price = candles[-1].close
    price = start_price
    start = BASE_TIME + timedelta(minutes=n_flat)
    for i in range(n_up):
        close = price + step_up
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=i),
                symbol=BTC,
                open=price,
                high=max(price, close) + 0.5,
                low=min(price, close) - 0.5,
                close=close,
                volume=1000.0,
            )
        )
        price = close
    for i in range(n_down):
        close = price + step_down
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=n_up + i),
                symbol=BTC,
                open=price,
                high=max(price, close) + 0.5,
                low=min(price, close) - 0.5,
                close=close,
                volume=1000.0,
            )
        )
        price = close
    return candles


# ── Warmup: keine Signale vor min_bars ─────────────────────────────────────


@pytest.mark.parametrize("name", list_strategies())
def test_no_signals_before_min_bars(name: str) -> None:
    # min_bars-1 Kerzen: das Fenster ist nicht voll → garantiert keine Auswertung.
    strategy = create_strategy(name, BTC)
    candles = make_candles(strategy.min_bars - 1, step=0.5)
    assert _feed(strategy, candles) == []
    assert strategy.n_buy_signals == 0
    assert strategy.n_sell_signals == 0


# ── Signal-Verhalten pro Strategie ─────────────────────────────────────────


def test_ema_cross_buys_on_uptrend_and_sells_on_downtrend() -> None:
    strategy = create_strategy("ema_cross", BTC)
    signals = _feed(strategy, _phases(200, 120, 0.8, 120, -0.8))
    actions = [s.action for s in signals]
    assert UP in actions
    assert DN in actions
    assert actions.index(UP) < actions.index(DN)


def test_macd_cross_fires_on_regime_change() -> None:
    strategy = create_strategy("macd_cross", BTC)
    signals = _feed(strategy, _phases(200, 120, 0.8, 120, -0.8))
    actions = [s.action for s in signals]
    assert UP in actions
    assert DN in actions


def test_supertrend_flips_direction() -> None:
    down = make_candles(200, step=-0.3)
    up = _phases(0, 200, 0.8, start_price=down[-1].close)
    strategy = create_strategy("supertrend", BTC)
    signals = _feed(strategy, down + up)
    assert any(s.action == UP for s in signals)


def test_donchian_breakout_buys_on_spike() -> None:
    strategy = create_strategy("donchian_breakout", BTC)
    signals = _feed(strategy, _phases(300, 50, 1.0))
    assert any(s.action == UP for s in signals)


def test_rsi_mean_reversion_buys_oversold_exit() -> None:
    crash = make_candles(250, step=-0.3)
    recovery = _phases(0, 40, 0.3, start_price=crash[-1].close)
    strategy = create_strategy("rsi_mean_reversion", BTC)
    signals = _feed(strategy, crash + recovery)
    assert any(s.action == UP for s in signals)


def test_rsi_mean_reversion_sells_overbought_exit() -> None:
    pump = make_candles(250, step=0.5)
    fade = _phases(0, 40, -0.3, start_price=pump[-1].close)
    strategy = create_strategy("rsi_mean_reversion", BTC)
    signals = _feed(strategy, pump + fade)
    assert any(s.action == DN for s in signals)


def _dip_then_rebound(n_flat: int, dip: float, rebound: float) -> list[Candle]:
    candles = make_candles(n_flat)
    base = BASE_TIME + timedelta(minutes=n_flat)
    price = 100.0
    dip_close = price + dip
    candles.append(
        Candle(
            timestamp=base,
            symbol=BTC,
            open=price,
            high=max(price, dip_close) + 0.5,
            low=min(price, dip_close) - 0.5,
            close=dip_close,
            volume=1000.0,
        )
    )
    price = dip_close
    rebound_close = price + rebound
    candles.append(
        Candle(
            timestamp=base + timedelta(minutes=1),
            symbol=BTC,
            open=price,
            high=max(price, rebound_close) + 0.5,
            low=min(price, rebound_close) - 0.5,
            close=rebound_close,
            volume=1000.0,
        )
    )
    return candles


def test_bollinger_reversion_buys_rebound_into_band() -> None:
    strategy = create_strategy("bollinger_reversion", BTC)
    signals = _feed(strategy, _dip_then_rebound(250, -3.0, 2.5))
    assert any(s.action == UP for s in signals)


def test_keltner_breakout_buys_band_break() -> None:
    strategy = create_strategy("keltner_breakout", BTC)
    signals = _feed(strategy, _phases(250, 30, 2.5))
    assert any(s.action == UP for s in signals)


def test_vwap_reversion_buys_reversion_to_vwap() -> None:
    flat = make_candles(300)
    drop = _phases(0, 10, -1.2)
    rebound = _phases(0, 20, 1.5, start_price=drop[-1].close)
    strategy = create_strategy("vwap_reversion", BTC)
    signals = _feed(strategy, flat + drop + rebound)
    assert any(s.action == UP for s in signals)


def test_stochastics_buys_oversold_cross() -> None:
    flat = make_candles(250)
    drop = _phases(0, 20, -1.0)
    rebound = _phases(0, 10, 0.8, start_price=drop[-1].close)
    strategy = create_strategy("stochastics", BTC)
    signals = _feed(strategy, flat + drop + rebound)
    assert any(s.action == UP for s in signals)


def test_momentum_roc_buys_uptrend_momentum() -> None:
    strategy = create_strategy("momentum_roc", BTC)
    signals = _feed(strategy, _phases(120, 60, 0.5))
    assert any(s.action == UP for s in signals)


# ── Signal-Felder (Einheitsschnitt) ────────────────────────────────────────


def _any_signal() -> tuple:
    strategy = create_strategy("ema_cross", BTC)
    signals = _feed(strategy, _phases(200, 120, 0.8, 120, -0.8))
    assert signals, "Voraussetzung: ema_cross erzeugt auf Phasen-Kerzen Signale"
    return strategy, signals[0]


def test_signal_fields_are_well_formed() -> None:
    strategy, signal = _any_signal()
    assert signal.action in (UP, DN)
    assert 0.5 <= signal.confidence <= 0.85
    assert signal.position_size == pytest.approx(strategy.trade_notional / strategy.initial_capital)
    assert signal.symbol == BTC
    assert signal.reason.startswith("ema_cross")
    assert signal.metadata["strategy"] == "ema_cross"
    assert signal.metadata["params"] == strategy.params


def test_signal_timestamp_is_signal_bar() -> None:
    strategy = create_strategy("ema_cross", BTC)
    candles = _phases(200, 120, 0.8)
    for candle in candles:
        signal = strategy.on_bar(candle)
        if signal is not None:
            assert signal.timestamp == candle.timestamp
            return
    pytest.fail("kein Signal erzeugt")


# ── Determinismus und Robustheit ───────────────────────────────────────────


@pytest.mark.parametrize("name", list_strategies())
def test_deterministic_signal_sequence(name: str) -> None:
    candles = make_sine_candles(400)
    first = [ (s.action, s.confidence) for s in _feed(create_strategy(name, BTC), candles) ]
    second = [ (s.action, s.confidence) for s in _feed(create_strategy(name, BTC), candles) ]
    assert first == second


@pytest.mark.parametrize("name", list_strategies())
def test_runs_on_all_regimes_without_error(name: str) -> None:
    # Schritte so gewählt, dass der Kurs positiv bleibt (100 → 220 / 100 → 20).
    regimes = [
        make_candles(400, step=0.3),
        make_candles(400, step=-0.2),
        make_candles(400, step=0.0),
        make_sine_candles(400),
    ]
    for candles in regimes:
        _feed(create_strategy(name, BTC), candles)  # darf keine Exception werfen


def test_signal_counters_increment() -> None:
    strategy = create_strategy("ema_cross", BTC)
    signals = _feed(strategy, _phases(200, 120, 0.8, 120, -0.8))
    buys = sum(1 for s in signals if s.action == UP)
    sells = sum(1 for s in signals if s.action == DN)
    assert strategy.n_buy_signals == buys
    assert strategy.n_sell_signals == sells
    assert buys >= 1
    assert sells >= 1
