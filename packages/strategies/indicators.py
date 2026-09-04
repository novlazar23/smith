"""Indikatoren der Strategie-Bibliothek (numpy, strikt ohne Lookahead).

Alle Funktionen arbeiten auf 1-D-``float64``-Arrays in der Reihenfolge
älteste → neueste Bar (Index ``-1`` = aktuelle Bar). Der Wert an Index ``i``
verwendet ausschließlich Daten der Indizes <= i. Warmup-Semantik steht in
jeder Funktions-Docstring (typisch: Seed aus den ersten ``period`` Werten,
darunter expandierendes Mittel).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sma(values: FloatArray, period: int) -> FloatArray:
    """Gleitendes arithmetisches Mittel (SMA).

    Warmup: Index ``i < period - 1`` erhält das Mittel der bis dahin
    verfügbaren Werte (expandierend), ab ``i >= period - 1`` das der
    letzten ``period`` Werte.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.empty(n)
    if n == 0:
        return out
    csum = np.cumsum(values)
    for i in range(n):
        start = max(0, i - period + 1)
        window_sum = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        out[i] = window_sum / (i - start + 1)
    return out


def ema(values: FloatArray, period: int) -> FloatArray:
    """Exponentiell geglättetes Mittel (EMA), rekursiv.

    Warmup: die ersten ``period`` Ausgabewerte werden mit dem SMA der
    ersten ``period`` Kerzen gezeichnet (identisch zum repo-internen
    Referenzmuster ``TrendAgent._ema``), danach ``alpha = 2/(period+1)``.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.empty(n)
    if n == 0:
        return out
    seed_len = min(period, n)
    alpha = 2.0 / (period + 1.0)
    seed = float(values[:seed_len].mean())
    out[:seed_len] = seed
    prev = seed
    for i in range(seed_len, n):
        prev = alpha * float(values[i]) + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    """RSI aus Wilder-Mitteln; flache Serie (beide 0) → neutral 50."""
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _rsi_series(values: FloatArray, period: int) -> FloatArray:
    """Wilder-RSI pro Bar; Warmup (Index < period) = neutral 50.0."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.full(n, 50.0)
    if n <= period:
        return out
    delta = np.diff(values)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    avg_gain = float(gain[:period].mean())
    avg_loss = float(loss[:period].mean())
    out[period] = _rsi_from(avg_gain, avg_loss)
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + float(gain[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(loss[i])) / period
        out[i + 1] = _rsi_from(avg_gain, avg_loss)
    return out


def rsi(values: FloatArray, period: int = 14) -> float:
    """Aktueller Wilder-RSI (Letzter Wert der RSI-Reihe)."""
    series = _rsi_series(np.asarray(values, dtype=np.float64), period)
    return float(series[-1]) if len(series) else 50.0


def _atr_series(high: FloatArray, low: FloatArray, close: FloatArray, period: int) -> FloatArray:
    """True-Range-Reihe (Wilder-Glättung); Warmup = expandierendes Mittel.

    Der True Range der allerersten Bar ist nicht definiert (kein
    Vorschließkurs) und wird durch ``high - low`` ersetzt.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    out = np.empty(n)
    if n == 0:
        return out
    prev_close = close[:-1]
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )
    tr = np.concatenate([[float(high[0] - low[0])], tr])
    seed_len = min(period, n)
    seed = float(tr[:seed_len].mean())
    out[:seed_len] = seed
    for i in range(seed_len, n):
        out[i] = (out[i - 1] * (period - 1) + float(tr[i])) / period
    return out


def atr(high: FloatArray, low: FloatArray, close: FloatArray, period: int = 14) -> float:
    """Aktueller Average True Range (Letzter Wert der ATR-Reihe)."""
    series = _atr_series(
        np.asarray(high, dtype=np.float64),
        np.asarray(low, dtype=np.float64),
        np.asarray(close, dtype=np.float64),
        period,
    )
    return float(series[-1]) if len(series) else 0.0


def rolling_vwap(
    high: FloatArray, low: FloatArray, close: FloatArray, volume: FloatArray, period: int
) -> FloatArray:
    """Rollierender VWAP (typical price, volumen-gewichtet).

    Warmup: Index ``i < period - 1`` über die bis dahin verfügbaren Kerzen
    (expandierendes Fenster); Division durch Summen-Präfix, kein Lookahead.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    tp = (high + low + close) / 3.0
    n = len(tp)
    out = np.empty(n)
    if n == 0:
        return out
    csum_tp_v = np.cumsum(tp * volume)
    csum_v = np.cumsum(volume)
    for i in range(n):
        start = max(0, i - period + 1)
        vol_sum = csum_v[i] - (csum_v[start - 1] if start > 0 else 0.0)
        tpv_sum = csum_tp_v[i] - (csum_tp_v[start - 1] if start > 0 else 0.0)
        out[i] = tpv_sum / vol_sum if vol_sum > 0.0 else float(tp[i])
    return out


def bollinger(close: FloatArray, period: int = 20, num_std: float = 2.0) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Bollinger-Bänder (Mittel, Ober, Unter) mit Population-Std (ddof=0).

    Warmup: Index ``i < period - 1`` nutzt Mittel/Std der verfügbaren Werte.
    """
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    mid = sma(close, period)
    std = np.empty(n)
    for i in range(n):
        start = max(0, i - period + 1)
        std[i] = float(close[start : i + 1].std(ddof=0))
    return mid, mid + num_std * std, mid - num_std * std


def macd(
    close: FloatArray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """MACD-Linie, Signal-Linie und Histogramm (EMA-basiert).

    Warmup erbt die EMA-Warmups (Seed = SMA der ersten ``period`` Kerzen).
    """
    close = np.asarray(close, dtype=np.float64)
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    return line, signal_line, line - signal_line


def stochastic(
    high: FloatArray, low: FloatArray, close: FloatArray, k: int = 14, d: int = 3
) -> tuple[FloatArray, FloatArray]:
    """Stochastik %K (Schnell, ungeglättet) und %D (SMA von %K).

    Warmup: Index ``i < k - 1`` → neutral 50.0 (vollständiges k-Fenster
    nicht verfügbar); %D nutzt die SMA-Warmup-Semantik.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    k_vals = np.full(n, 50.0)
    for i in range(k - 1, n):
        highest = float(high[i - k + 1 : i + 1].max())
        lowest = float(low[i - k + 1 : i + 1].min())
        k_vals[i] = (float(close[i]) - lowest) / (highest - lowest) * 100.0 if highest > lowest else 50.0
    return k_vals, sma(k_vals, d)


def roc(values: FloatArray, period: int) -> FloatArray:
    """Rate of Change: ``values[i]/values[i-period] - 1`` (Warmup = 0.0)."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.zeros(n)
    for i in range(period, n):
        base = float(values[i - period])
        if abs(base) > 1e-12:
            out[i] = float(values[i]) / base - 1.0
    return out


def donchian(high: FloatArray, low: FloatArray, entry_period: int) -> tuple[FloatArray, FloatArray]:
    """Donchian-Kanäle (Obers = max(high), Untersch = min(low)) pro Bar.

    Warmup: Index ``i < entry_period - 1`` über die verfügbaren Kerzen.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = len(high)
    upper = np.empty(n)
    lower = np.empty(n)
    for i in range(n):
        start = max(0, i - entry_period + 1)
        upper[i] = float(high[start : i + 1].max())
        lower[i] = float(low[start : i + 1].min())
    return upper, lower


def keltner(
    close: FloatArray, high: FloatArray, low: FloatArray, period: int = 20, multiplier: float = 2.0
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Keltner-Kanäle: Mittel = EMA, Bänder = Mittel ± ``multiplier``-fach ATR."""
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    mid = ema(close, period)
    atr_series = _atr_series(high, low, close, period)
    return mid, mid + multiplier * atr_series, mid - multiplier * atr_series


def supertrend(
    high: FloatArray, low: FloatArray, close: FloatArray, period: int = 10, multiplier: float = 3.0
) -> tuple[FloatArray, FloatArray]:
    """Supertrend: Trendlinie und Richtung (+1 Aufwärtstrend, -1 Abwärts).

    Klassischer Supertrend (Obrigee) auf ATR-Basis; Warmup der ATR-Reihe
    (expandierendes Mittel) propagiert in die Bänder. Die erste Bar startet
    mit Richtung +1.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    trend_line = np.empty(n)
    direction = np.ones(n, dtype=np.int8)
    if n == 0:
        return trend_line, direction
    atr_series = _atr_series(high, low, close, period)
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr_series
    basic_lower = hl2 - multiplier * atr_series
    final_upper = np.empty(n)
    final_lower = np.empty(n)
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    for i in range(1, n):
        final_upper[i] = (
            basic_upper[i]
            if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]
            else final_upper[i - 1]
        )
        final_lower[i] = (
            basic_lower[i]
            if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]
            else final_lower[i - 1]
        )
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close[i] > final_upper[i] else -1
    trend_line = np.where(direction == 1, final_lower, final_upper)
    return trend_line, direction
