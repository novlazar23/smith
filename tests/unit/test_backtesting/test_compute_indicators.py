"""Equivalenz-Tests: compute_indicators (O(n)) vs. Legacy-Referenz (O(n²)).

Die Legacy-Hilfsfunktionen sind hier als Referenz implementiert (bit-exakte
Kopie der alten pro-Bar-Implementierung). Beide müssen auf denselben
Eingaben identische Werte liefern.
"""

from __future__ import annotations

import random

from packages.backtesting.core import Candle
from packages.backtesting.datafeed import compute_indicators

# ── Legacy-Referenz (alte pro-Bar-Implementierung, unverändert) ─────────────


def _legacy_ema(values: list[float], idx: int, period: int) -> float | None:
    if idx < period - 1:
        return None
    if idx == 0:
        return values[0]
    k = 2.0 / (period + 1)
    result = None
    for i in range(idx + 1):
        if i < period - 1:
            result = (result if result is not None else 0.0) + values[i]
            if i == period - 2:
                result /= period
        else:
            result = values[i] if result is None else values[i] * k + result * (1 - k)
    return result


def _legacy_macd(
    values: list[float], idx: int
) -> tuple[float | None, float | None, float | None]:
    if idx < 33:
        return None, None, None
    ema12 = _legacy_ema(values, idx, 12)
    ema26 = _legacy_ema(values, idx, 26)
    if ema12 is None or ema26 is None:
        return None, None, None
    macd_line = ema12 - ema26
    macd_values: list[float] = []
    for j in range(max(0, idx - 33), idx + 1):
        e12 = _legacy_ema(values, j, 12)
        e26 = _legacy_ema(values, j, 26)
        if e12 is not None and e26 is not None:
            macd_values.append(e12 - e26)
    if len(macd_values) < 9:
        return macd_line, None, macd_line
    signal = sum(macd_values[-9:]) / min(9, len(macd_values))
    return macd_line, signal, macd_line - signal


def _make_candles(closes: list[float]) -> list[Candle]:
    import math
    from datetime import datetime, timedelta

    base = datetime(2021, 1, 1)
    candles = []
    for i, close in enumerate(closes):
        open_ = closes[i - 1] if i > 0 else close
        high = max(open_, close) * (1 + 0.001)
        low = min(open_, close) * (1 - 0.001)
        candles.append(
            Candle(
                timestamp=base + timedelta(minutes=i),
                symbol="BTC/USDT",
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=1000.0 + math.sin(i) * 10.0,
            )
        )
    return candles


def _random_closes(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    price = 60000.0
    out = []
    for _ in range(n):
        price *= 1 + rng.uniform(-0.005, 0.005)
        out.append(price)
    return out


# ── Tests ───────────────────────────────────────────────────────────────────


def test_ema_and_macd_bit_identical_on_random_walk() -> None:
    closes = _random_closes(400, seed=7)
    candles = compute_indicators(_make_candles(closes))

    for i, candle in enumerate(candles):
        exp_ema12 = _legacy_ema(closes, i, 12)
        exp_ema26 = _legacy_ema(closes, i, 26)
        exp_line, exp_signal, exp_hist = _legacy_macd(closes, i)
        assert candle.ema_12 == exp_ema12, f"ema_12 @ {i}"
        assert candle.ema_26 == exp_ema26, f"ema_26 @ {i}"
        assert candle.macd_line == exp_line, f"macd_line @ {i}"
        assert candle.macd_signal == exp_signal, f"macd_signal @ {i}"
        assert candle.macd_histogram == exp_hist, f"macd_histogram @ {i}"


def test_ema_none_semantics_before_warmup() -> None:
    closes = _random_closes(100, seed=11)
    candles = compute_indicators(_make_candles(closes))
    for i in range(100):
        assert (candles[i].ema_12 is None) == (i < 11)
        assert (candles[i].ema_26 is None) == (i < 25)
        assert (candles[i].macd_line is None) == (i < 33)


def test_short_series_all_macd_none() -> None:
    closes = _random_closes(33, seed=3)
    candles = compute_indicators(_make_candles(closes))
    for i, candle in enumerate(candles):
        exp_line, exp_signal, exp_hist = _legacy_macd(closes, i)
        assert candle.macd_line == exp_line == None  # noqa: E711
        assert candle.macd_signal == exp_signal == None  # noqa: E711
        assert candle.macd_histogram == exp_hist == None  # noqa: E711


def test_macd_signal_window_uses_last_nine_defined() -> None:
    # Konstruierte Serie: MACD-Werte ab Index 25 definiert; am Index 33 ist
    # das 9er-Fenster exakt [25..33].
    closes = _random_closes(60, seed=21)
    candles = compute_indicators(_make_candles(closes))
    i = 33
    exp_line, exp_signal, exp_hist = _legacy_macd(closes, i)
    assert candles[i].macd_line == exp_line
    assert candles[i].macd_signal == exp_signal
    assert candles[i].macd_histogram == exp_hist
