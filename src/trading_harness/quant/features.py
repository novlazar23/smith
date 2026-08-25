"""Feature-Berechnung für OHLCV-Daten (Phase 2).

Deterministischer Indikator-Computer auf reiner Standardbibliothek
(``math`` + ``statistics``) — ohne numpy/pandas. Alle Indikatoren sind reine
Funktionen der übergebenen Kerzenreihe: gleiche Eingabe, gleiche Ausgabe.

Kerzenformat (siehe ``ohlcv_ingestion.py``): Dict mit numerischen
``open``, ``high``, ``low``, ``close``, ``volume`` (optional ``time``).

Semantik der Guard-Klauseln: Bei unzureichender Historie liefert der
betroffene Indikator ``None`` (nie eine halbe Zahl). ``compute`` zählt die
verfügbaren Indikatoren als ``feature_count`` und misst die
Rechenzeit als ``computation_time_ms`` (Metadatenfelder laut ``schema.py``).
"""

from __future__ import annotations

import itertools
import math
import statistics
import time
from typing import Any


class FeatureEngine:
    """Deterministischer Feature-Computer — nur stdlib (math, statistics)."""

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        vol_period: int = 20,
    ) -> None:
        if min(rsi_period, macd_fast, macd_slow, macd_signal, bb_period, atr_period, vol_period) < 1:
            raise ValueError("alle Periode-Parameter müssen >= 1 sein")
        if macd_fast >= macd_slow:
            raise ValueError("macd_fast muss kleiner als macd_slow sein")
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.vol_period = vol_period

    def compute(self, candles: list[dict]) -> dict[str, Any]:
        """Berechnet alle Indikatoren aus einer Kerzenreihe.

        Ergebnis enthält pro Indikator ``None`` bei unzureichender Historie
        sowie die Metadaten ``feature_count`` und ``computation_time_ms``.
        """
        t0 = time.monotonic()
        closes = [c["close"] for c in candles]

        result: dict[str, Any] = {
            "rsi": self._rsi(closes),
            "macd": self._macd(closes),
            "bollinger": self._bollinger(closes),
            "atr": self._atr(candles),
            "volatility": self._volatility(closes),
            "vwap": self._vwap(candles),
        }
        result["feature_count"] = sum(1 for v in result.values() if v is not None)
        result["computation_time_ms"] = (time.monotonic() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    # Indikatoren
    # ------------------------------------------------------------------

    def _rsi(self, closes: list[float]) -> float | None:
        """RSI (Wilder-Glättung) auf der letzten Kerze; < period+1 Werte → None."""
        period = self.rsi_period
        if len(closes) < period + 1:
            return None
        gains: list[float] = []
        losses: list[float] = []
        for prev, curr in itertools.pairwise(closes):
            change = curr - prev
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))
        avg_gain = statistics.fmean(gains[:period])
        avg_loss = statistics.fmean(losses[:period])
        for gain, loss in zip(gains[period:], losses[period:], strict=True):
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _macd(self, closes: list[float]) -> dict[str, float] | None:
        """MACD (fast/slow EMA-Differenz) mit Signal-EMA und Histogramm.

        Liegt bei unzureichender Historie (< slow+signal Werte) keine
        vollständige Signal-Linie vor → None.
        """
        if len(closes) < self.macd_slow + self.macd_signal:
            return None
        fast = self._ema_series(closes, self.macd_fast)
        slow = self._ema_series(closes, self.macd_slow)
        # slow[i] ↔ closes[macd_slow-1+i], fast[i] ↔ closes[macd_fast-1+i]:
        # die Serien ab dem Index, ab dem beide EMAs existieren, aufrichten.
        offset = self.macd_slow - self.macd_fast
        macd_line = [f - s for f, s in zip(fast[offset:], slow, strict=True)]
        signal = self._ema_series(macd_line, self.macd_signal)
        if not signal:
            return None
        macd_value = macd_line[-1]
        signal_value = signal[-1]
        return {
            "macd": macd_value,
            "signal": signal_value,
            "histogram": macd_value - signal_value,
        }

    def _bollinger(self, closes: list[float]) -> dict[str, float] | None:
        """Bollinger-Bänder (Mittel ± bb_std · Populations-StdDev)."""
        if len(closes) < self.bb_period:
            return None
        window = closes[-self.bb_period:]
        middle = statistics.fmean(window)
        std_dev = statistics.pstdev(window)
        upper = middle + self.bb_std * std_dev
        lower = middle - self.bb_std * std_dev
        bandwidth = (upper - lower) / middle if middle != 0 else 0.0
        return {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "std_dev": std_dev,
            "bandwidth": bandwidth,
        }

    def _atr(self, candles: list[dict]) -> float | None:
        """Average True Range (Wilder-Glättung) auf der letzten Kerze."""
        period = self.atr_period
        if len(candles) < period + 1:
            return None
        trs: list[float] = [candles[0]["high"] - candles[0]["low"]]
        for i in range(1, len(candles)):
            high = candles[i]["high"]
            low = candles[i]["low"]
            prev_close = candles[i - 1]["close"]
            trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        atr = statistics.fmean(trs[:period])
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr

    def _volatility(self, closes: list[float]) -> float | None:
        """Populations-Standardabweichung der letzten vol_period Log-Renditen."""
        if len(closes) < self.vol_period + 1:
            return None
        log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        return statistics.pstdev(log_returns[-self.vol_period:])

    def _vwap(self, candles: list[dict]) -> float | None:
        """Volumengewichteter Durchschnittspreis (tp = (H+L+C)/3) über die Reihe."""
        if not candles:
            return None
        cum_pv = 0.0
        cum_vol = 0.0
        for candle in candles:
            tp = (candle["high"] + candle["low"] + candle["close"]) / 3.0
            cum_pv += tp * candle["volume"]
            cum_vol += candle["volume"]
        if cum_vol == 0.0:
            return None
        return cum_pv / cum_vol

    # ------------------------------------------------------------------
    # Hilfsfunktionen
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_series(data: list[float], period: int) -> list[float]:
        """EMA-Serie, seeded mit dem SMA der ersten ``period`` Werte.

        Rückgabe-Eintrag ``i`` entspricht ``data[period - 1 + i]``;
        bei zu kurzen Daten leere Liste.
        """
        if len(data) < period:
            return []
        alpha = 2.0 / (period + 1.0)
        ema = statistics.fmean(data[:period])
        series = [ema]
        for value in data[period:]:
            ema = value * alpha + ema * (1.0 - alpha)
            series.append(ema)
        return series
