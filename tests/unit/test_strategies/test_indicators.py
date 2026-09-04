"""Tests für die Indikator-Bibliothek: exakte Werte + No-Lookahead-Eigenschaft.

Die Kern-Eigenschaft: Wenn man an das Ende einer Serie weitere (Zukunfts-)
Daten anhängt, dürfen sich die bereits berechneten Werte NICHT ändern.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from packages.strategies import indicators as ta


def _no_lookahead(fn: Callable[..., Any], *arrays: np.ndarray, **kwargs: Any) -> None:
    """Hängt eine zukunfts-Bar ab und prüft Stabilität der Vorwerte (Last-Bar-Prinzip)."""
    full = fn(*arrays, **kwargs)
    n = len(arrays[0])
    trimmed = fn(*(a[: n - 1] for a in arrays), **kwargs)
    full = full if isinstance(full, tuple) else (full,)
    trimmed = trimmed if isinstance(trimmed, tuple) else (trimmed,)
    for full_series, trimmed_series in zip(full, trimmed, strict=True):
        assert np.allclose(full_series[: n - 1], trimmed_series, equal_nan=True), (
            f"{fn.__name__}: Anhängen zukünftiger Daten verändert Vergangenheitswerte"
        )


def test_sma_exact_values() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ta.sma(values, 3)
    # Warmup: expandierendes Mittel, dann Fenstergröße 3
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(1.5)
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)
    assert out[4] == pytest.approx(4.0)


def test_ema_seed_is_sma_of_first_period() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = ta.ema(values, 3)
    # Seed = SMA der ersten 3 Kerzen (2.0), auf die Seed-Bars gebroadcastet
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(2.0)
    assert out[2] == pytest.approx(2.0)
    # Rekursion: alpha = 0.5 → out[3] = 0.5*4 + 0.5*2 = 3.0
    assert out[3] == pytest.approx(3.0)


def test_rsi_extremes() -> None:
    rising = np.arange(1.0, 31.0)
    falling = np.arange(30.0, 0.0, -1.0)
    flat = np.full(30, 100.0)
    assert ta.rsi(rising, 14) == pytest.approx(100.0)
    assert ta.rsi(falling, 14) == pytest.approx(0.0)
    assert ta.rsi(flat, 14) == pytest.approx(50.0)


def test_atr_constant_range() -> None:
    # Constante Kerzen: high-low = 2.0, keine Close-Sprünge → ATR = 2.0
    high = np.full(30, 102.0)
    low = np.full(30, 100.0)
    close = np.full(30, 101.0)
    assert ta.atr(high, low, close, 14) == pytest.approx(2.0)


def test_rolling_vwap_uniform_price() -> None:
    # Gleichbleibender Preis und Volumen → VWAP = Preis
    n = 60
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    volume = np.full(n, 1000.0)
    out = ta.rolling_vwap(high, low, close, volume, 50)
    assert np.allclose(out, 100.0)


def test_bollinger_flat_series() -> None:
    close = np.full(40, 100.0)
    mid, upper, lower = ta.bollinger(close, 20, 2.0)
    assert np.allclose(mid, 100.0)
    assert np.allclose(upper, 100.0)
    assert np.allclose(lower, 100.0)


def test_macd_flat_series_is_zero() -> None:
    close = np.full(60, 42.0)
    line, signal, hist = ta.macd(close, 12, 26, 9)
    assert np.allclose(line, 0.0)
    assert np.allclose(signal, 0.0)
    assert np.allclose(hist, 0.0)


def test_stochastic_at_range_extremes() -> None:
    # Close = Fenster-Hoch → %K = 100; Close = Fenster-Tief → %K = 0
    high = np.full(20, 110.0)
    low = np.full(20, 90.0)
    close = np.full(20, 110.0)
    k, _ = ta.stochastic(high, low, close, 14, 3)
    assert k[-1] == pytest.approx(100.0)
    close = np.full(20, 90.0)
    k, _ = ta.stochastic(high, low, close, 14, 3)
    assert k[-1] == pytest.approx(0.0)


def test_donchian_exact_values() -> None:
    high = np.array([1.0, 2.0, 3.0, 2.0, 5.0])
    low = np.array([0.5, 1.5, 2.5, 1.5, 4.5])
    upper, lower = ta.donchian(high, low, 3)
    assert upper[2] == pytest.approx(3.0)  # max(1, 2, 3)
    assert lower[2] == pytest.approx(0.5)  # min(0.5, 1.5, 2.5)
    assert upper[4] == pytest.approx(5.0)  # max(3, 2, 5)


def test_keltner_band_structure() -> None:
    n = 50
    close = np.linspace(100.0, 120.0, n)
    high = close + 1.0
    low = close - 1.0
    mid, upper, lower = ta.keltner(close, high, low, 20, 2.0)
    assert np.all(upper >= mid)
    assert np.all(lower <= mid)
    assert np.allclose(upper - mid, mid - lower)


def test_roc_values() -> None:
    values = np.array([100.0, 110.0, 121.0])
    out = ta.roc(values, 1)
    assert out[0] == 0.0  # Warmup
    assert out[1] == pytest.approx(0.10)
    assert out[2] == pytest.approx(0.10)


def test_supertrend_direction_on_trends() -> None:
    n = 80
    high_up = np.linspace(100.0, 140.0, n)
    low_up = high_up - 1.0
    close_up = high_up - 0.5
    _, direction_up = ta.supertrend(high_up, low_up, close_up, 10, 3.0)
    assert direction_up[-1] == 1

    close_dn = np.linspace(140.0, 100.0, n)
    high_dn = close_dn + 0.5
    low_dn = close_dn - 1.0
    _, direction_dn = ta.supertrend(high_dn, low_dn, close_dn, 10, 3.0)
    assert direction_dn[-1] == -1


# ── No-Lookahead-Eigenschaft (alle Serien-Indikatoren) ─────────────────────


def test_sma_no_lookahead() -> None:
    rng = np.random.default_rng(7)
    values = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    _no_lookahead(ta.sma, values, period=20)


def test_ema_no_lookahead() -> None:
    rng = np.random.default_rng(8)
    values = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    _no_lookahead(ta.ema, values, period=20)


def test_rsi_series_no_lookahead() -> None:
    rng = np.random.default_rng(9)
    values = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    _no_lookahead(ta._rsi_series, values, period=14)


def test_atr_series_no_lookahead() -> None:
    rng = np.random.default_rng(10)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    high = close + rng.uniform(0.1, 2.0, 200)
    low = close - rng.uniform(0.1, 2.0, 200)
    _no_lookahead(ta._atr_series, high, low, close, period=14)


def test_rolling_vwap_no_lookahead() -> None:
    rng = np.random.default_rng(11)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    high = close + rng.uniform(0.1, 2.0, 200)
    low = close - rng.uniform(0.1, 2.0, 200)
    volume = rng.uniform(100.0, 1000.0, 200)
    _no_lookahead(ta.rolling_vwap, high, low, close, volume, period=50)


def test_bollinger_no_lookahead() -> None:
    rng = np.random.default_rng(12)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    _no_lookahead(ta.bollinger, close)


def test_macd_no_lookahead() -> None:
    rng = np.random.default_rng(13)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    _no_lookahead(ta.macd, close)


def test_stochastic_no_lookahead() -> None:
    rng = np.random.default_rng(14)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    high = close + rng.uniform(0.1, 2.0, 200)
    low = close - rng.uniform(0.1, 2.0, 200)
    _no_lookahead(ta.stochastic, high, low, close)


def test_roc_no_lookahead() -> None:
    rng = np.random.default_rng(15)
    values = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    _no_lookahead(ta.roc, values, period=20)


def test_donchian_no_lookahead() -> None:
    rng = np.random.default_rng(16)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    high = close + rng.uniform(0.1, 2.0, 200)
    low = close - rng.uniform(0.1, 2.0, 200)
    _no_lookahead(ta.donchian, high, low, entry_period=20)


def test_keltner_no_lookahead() -> None:
    rng = np.random.default_rng(17)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    high = close + rng.uniform(0.1, 2.0, 200)
    low = close - rng.uniform(0.1, 2.0, 200)
    _no_lookahead(ta.keltner, close, high, low, period=20, multiplier=2.0)


def test_supertrend_no_lookahead() -> None:
    rng = np.random.default_rng(18)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    high = close + rng.uniform(0.1, 2.0, 200)
    low = close - rng.uniform(0.1, 2.0, 200)
    _no_lookahead(ta.supertrend, high, low, close)
