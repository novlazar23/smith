"""Deterministischer OHLCV-Marktsnapshot für LLM-Prompts.

Determinismus-Garantie: Für identische Eingabe-Arrays liefert
`summarize_window` **byte-identischen** Text. Alle Indikatoren (SMA/EMA,
Wilder-RSI, Wilder-ATR, Returns, Distanzen, Volumen-Trend) werden lokal
mit NumPy berechnet — kein Netzwerk, keine Wall-Clock, keine Zufallszahlen,
keine Lookahead-Zugriffe (jeder Wert nutzt nur Kerzen bis zur letzten der
``last_n`` Kerzen). Jede Zahl wird über das fixe Format ``f"{x:.4g}"``
(4 signifikante Ziffern) gerendert. Der Text ist damit eine stabile
Cache-Key-Bestandteil.

Kopfzeile: ``bar i of <len>`` ist snapshot-relativ (``i = n-1``, ``<len> = n``
mit ``n = min(Kerzen, last_n)``), damit ein flaches Fenster über
aufeinanderfolgende Evaluations hinweg denselben String liefert.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Anzahl der zuletzt aufgelisteten Kerzen im Abschnitt "recent bars".
_RECENT_BARS = 30


def summarize_window(
    *,
    open: NDArray,  # noqa: A002 — Parametername ist Teil der Schnittstelle (Spezifikation)
    high: NDArray,
    low: NDArray,
    close: NDArray,
    volume: NDArray,
    last_n: int = 60,
) -> str:
    """Baut den deterministischen OHLCV-Snapshot über die letzten ``last_n`` Kerzen.

    Args:
        open: Eröffnungspreise (älteste → neueste).
        high: Höchstpreise.
        low: Tiefstpreise.
        close: Schlusskurse.
        volume: Volumen.
        last_n: Fenstergröße in Kerzen (Standard 60; mehr wird abgeschnitten).

    Returns:
        Zeilenbasierter englischsprachiger Snapshot (s. Modul-Docstring).
    """
    closes = np.asarray(close, dtype=np.float64)
    n = min(len(closes), max(1, int(last_n)))
    if n < 1:
        raise ValueError("summarize_window benötigt mindestens eine Kerze")

    highs = np.asarray(high, dtype=np.float64)[-n:]
    lows = np.asarray(low, dtype=np.float64)[-n:]
    closes = closes[-n:]
    vols = np.asarray(volume, dtype=np.float64)[-n:]

    last_close = float(closes[-1])
    sma12 = _sma(closes, 12)
    sma26 = _sma(closes, 26)
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    rsi = _rsi_wilder(closes)
    atr = _atr_wilder(highs, lows, closes)
    dist_high = _vs_pct(last_close, float(np.max(highs)))
    dist_low = _vs_pct(last_close, float(np.min(lows)))

    lines: list[str] = [
        f"BTC/USDT-style OHLCV snapshot (last {n} bars, 1m) — bar {n - 1} of {n}",
        f"last_close={_fmt(last_close)}",
        f"return_1b={_fmt(_return_pct(closes, 1))}",
        f"return_5b={_fmt(_return_pct(closes, 5))}",
        f"return_20b={_fmt(_return_pct(closes, 20))}",
        f"sma12={_fmt(sma12)}",
        f"sma26={_fmt(sma26)}",
        f"ema12={_fmt(ema12)}",
        f"ema26={_fmt(ema26)}",
        f"close_vs_sma12_pct={_fmt(_vs_pct(last_close, sma12))}",
        f"close_vs_sma26_pct={_fmt(_vs_pct(last_close, sma26))}",
        f"rsi14={_fmt(rsi)}",
        f"atr14_pct={_fmt(_vs_pct(atr, last_close))}",
        f"dist_from_high_lastN_pct={_fmt(dist_high)}",
        f"dist_from_low_lastN_pct={_fmt(dist_low)}",
        f"volume_trend={_fmt(_volume_trend(vols))}",
        "# recent bars (close, volume, ret_1b)",
    ]
    count = min(_RECENT_BARS, n)
    for offset in range(count, 0, -1):
        index = n - offset
        close_i = float(closes[index])
        ret = _ret_1b(closes, index)
        lines.append(
            f"bar -{offset}: close={_fmt(close_i)} vol={_fmt(float(vols[index]))} ret={ret}"
        )
    return "\n".join(lines)


def _fmt(value: float) -> str:
    """Fixes 4-sig-Ziffern-Format (Determinismus)."""
    return f"{value:.4g}"


def _vs_pct(numerator: float, denominator: float) -> float:
    """Relative Abweichung in Prozent (Nenner 0 → 0.0)."""
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator - 1.0) * 100.0


def _return_pct(closes: NDArray, bars: int) -> float:
    """Rückkehr über ``bars`` Kerzen in Prozent (zu wenige Daten → 0.0)."""
    if len(closes) < bars + 1 or float(closes[-1 - bars]) == 0.0:
        return 0.0
    return (float(closes[-1]) / float(closes[-1 - bars]) - 1.0) * 100.0


def _ret_1b(closes: NDArray, index: int) -> str:
    """1-Bar-Rückkehr am Index als Format-String (erste Kerze/Zero → "n/a")."""
    if index == 0:
        return "n/a"
    prev = float(closes[index - 1])
    if prev == 0.0:
        return "n/a"
    return _fmt((float(closes[index]) / prev - 1.0) * 100.0)


def _sma(values: NDArray, period: int) -> float:
    """Simple moving average über die letzten ``min(period, len)`` Werte."""
    return float(np.mean(values[-period:]))


def _ema(values: NDArray, period: int) -> float:
    """Exponentielle moving average (SMA-Saat, Standard-Alpha), kausal."""
    alpha = 2.0 / (period + 1.0)
    seed_len = min(period, len(values))
    ema = float(np.mean(values[:seed_len]))
    for value in values[seed_len:]:
        ema = alpha * float(value) + (1.0 - alpha) * ema
    return ema


def _rsi_wilder(closes: NDArray, period: int = 14) -> float:
    """Wilder-RSI (14) ohne Lookahead; unzureichende Daten → neutrale 50.0."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for gain, loss in zip(gains[period:], losses[period:], strict=True):
        avg_gain = (avg_gain * (period - 1) + float(gain)) / period
        avg_loss = (avg_loss * (period - 1) + float(loss)) / period
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _atr_wilder(highs: NDArray, lows: NDArray, closes: NDArray, period: int = 14) -> float:
    """Wilder-ATR (14) ohne Lookahead; untere Datenmenge → Simple-Mean-Fallback."""
    if len(closes) < 2:
        return 0.0
    prev_close = closes[:-1]
    true_range = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)),
    )
    if len(true_range) < period:
        return float(np.mean(true_range))
    atr = float(np.mean(true_range[:period]))
    for value in true_range[period:]:
        atr = (atr * (period - 1) + float(value)) / period
    return atr


def _volume_trend(vols: NDArray) -> float:
    """Mittel der letzten 5 Kerzen / der 5 davor (Neutral: 1.0 bei <10 Kerzen/Zero)."""
    if len(vols) < 10:
        return 1.0
    prior = float(np.mean(vols[-10:-5]))
    if prior == 0.0:
        return 1.0
    return float(np.mean(vols[-5:])) / prior
