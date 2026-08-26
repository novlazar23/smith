"""Regime-Erkennung für Marktdaten (Phase 4).

Klassifiziert Marktphasen (strong_bull, weak_bull, range, weak_bear, strong_bear,
high_volatility, low_volatility, crash, recovery) basierend auf SMA-Crossover,
ADX-ähnlicher Stärke und Volatilitätsklassifikation.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

REGIME_NAMES: tuple[str, ...] = (
    "strong_bull", "weak_bull", "range",
    "weak_bear", "strong_bear",
    "high_volatility", "low_volatility",
    "crash", "recovery",
)


@dataclass
class RegimeResult:
    """Erkanntes Marktregime."""
    regime: str
    confidence: float  # 0.0–1.0
    duration: int  # wie viele Kerzen dieses Regime anhält
    indicators: dict[str, Any] = field(default_factory=dict)


def _sma(values: list[float], period: int) -> float | None:
    """Simple Moving Average."""
    if len(values) < period:
        return None
    return statistics.mean(values[-period:])


def _pstdev(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _log_returns(closes: list[float]) -> list[float]:
    """Logarithmische Renditen."""
    returns: list[float] = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            returns.append(math.log(closes[i] / closes[i - 1]))
    return returns


class RegimeDetector:
    """Deterministische Regime-Erkennung — nur stdlib."""

    def __init__(
        self,
        sma_fast: int = 10,
        sma_slow: int = 50,
        adx_period: int = 14,
        vol_period: int = 20,
        crash_threshold: float = -0.05,
    ) -> None:
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.adx_period = adx_period
        self.vol_period = vol_period
        self.crash_threshold = crash_threshold

    def detect(self, candles: list[dict]) -> RegimeResult:
        """Erkennt das aktuelle Regime aus einer Kerzenreihe."""
        if len(candles) < max(self.sma_slow, self.adx_period, self.vol_period) + 1:
            return RegimeResult(
                regime="range", confidence=0.5, duration=0,
                indicators={"reason": "insufficient_data"},
            )

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]

        # Step 1: Trend
        sma_f = _sma(closes, self.sma_fast)
        sma_s = _sma(closes, self.sma_slow)
        assert sma_f is not None and sma_s is not None  # guaranteed by length check

        trend_ratio = abs(sma_f - sma_s) / sma_s if sma_s != 0 else 0
        is_range = trend_ratio < 0.005
        is_bullish = sma_f > sma_s

        # Step 2: ADX
        adx = self._compute_adx(highs, lows, closes)
        is_strong = adx is not None and adx > 25
        is_weak = adx is not None and adx < 20

        # Step 3: Volatilität
        vol_class = self._classify_volatility(closes)

        # Step 4: Crash/Recovery
        returns = _log_returns(closes)
        last_return = returns[-1] if returns else 0.0
        crash_detected = last_return < self.crash_threshold

        recovery_detected = False
        if len(returns) >= 10:
            recent = returns[-10:]
            had_crash = any(r < self.crash_threshold for r in recent)
            positive_since = all(r >= 0 for r in recent[recent.index(min(recent)):])
            if had_crash and last_return > 0 and positive_since:
                recovery_detected = True

        # Step 5: Combine
        if crash_detected:
            regime, confidence = "crash", min(abs(last_return / self.crash_threshold), 1.0)
        elif recovery_detected:
            regime, confidence = "recovery", 0.7
        elif vol_class == "high" and not is_strong:
            regime, confidence = "high_volatility", 0.8
        elif vol_class == "low" and not is_strong:
            regime, confidence = "low_volatility", 0.8
        elif is_range or is_weak:
            regime, confidence = "range", 0.6 + (0.2 if is_weak else 0.0)
        elif is_bullish and is_strong:
            regime, confidence = "strong_bull", min(adx / 50, 1.0) if adx else 0.7
        elif is_bullish:
            regime, confidence = "weak_bull", 0.6
        elif is_strong:
            regime, confidence = "strong_bear", min(adx / 50, 1.0) if adx else 0.7
        else:
            regime, confidence = "weak_bear", 0.6

        # Duration: count consecutive candles with same regime direction
        duration = self._compute_duration(candles, regime)

        return RegimeResult(
            regime=regime,
            confidence=min(max(confidence, 0.0), 1.0),
            duration=duration,
            indicators={
                "sma_fast": sma_f,
                "sma_slow": sma_s,
                "adx": adx,
                "volatility_class": vol_class,
                "last_return": last_return,
            },
        )

    def detect_series(self, candles: list[dict], window: int = 50) -> list[RegimeResult]:
        """Erkennt Regime für jeden Punkt in der Reihe (rolling)."""
        results: list[RegimeResult] = []
        min_len = max(self.sma_slow, self.adx_period, self.vol_period) + 1
        if len(candles) < min_len:
            return [RegimeResult(regime="range", confidence=0.5, duration=0)]
        for i in range(min_len - 1, len(candles)):
            results.append(self.detect(candles[: i + 1]))
        return results

    def _compute_adx(
        self, highs: list[float], lows: list[float], closes: list[float]
    ) -> float | None:
        """Berechnet ADX (Average Directional Index)."""
        n = len(highs)
        if n < self.adx_period + 2:
            return None

        plus_dm: list[float] = []
        minus_dm: list[float] = []
        tr_list: list[float] = []

        for i in range(1, n):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm.append(max(up, 0.0) if up > down else 0.0)
            minus_dm.append(max(down, 0.0) if down > up else 0.0)

            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            tr_list.append(tr)

        if len(tr_list) < self.adx_period:
            return None

        # Wilder smoothing
        atr = sum(tr_list[: self.adx_period]) / self.adx_period
        plus_di_sum = sum(plus_dm[: self.adx_period])
        minus_di_sum = sum(minus_dm[: self.adx_period])

        dx_values: list[float] = []
        for i in range(self.adx_period, len(tr_list)):
            atr = (atr * (self.adx_period - 1) + tr_list[i]) / self.adx_period
            plus_di_sum = (plus_di_sum * (self.adx_period - 1) + plus_dm[i]) / self.adx_period
            minus_di_sum = (minus_di_sum * (self.adx_period - 1) + minus_dm[i]) / self.adx_period

            if atr > 0:
                plus_di = plus_di_sum / atr * 100
                minus_di = minus_di_sum / atr * 100
            else:
                plus_di = minus_di = 0.0

            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx_values.append(abs(plus_di - minus_di) / di_sum * 100)
            else:
                dx_values.append(0.0)

        if not dx_values:
            return None

        adx = statistics.mean(dx_values[-self.adx_period:])
        return adx

    def _classify_volatility(self, closes: list[float]) -> str:
        """Klassifiziert Volatilität: high/normal/low."""
        returns = _log_returns(closes)
        if len(returns) < self.vol_period:
            return "normal"

        current_vol = _pstdev(returns[-self.vol_period :])
        hist_vols = [
            _pstdev(returns[i - self.vol_period : i])
            for i in range(self.vol_period, len(returns) + 1)
            if i - self.vol_period >= 0
        ]

        if not hist_vols:
            return "normal"

        median_vol = statistics.median(hist_vols)
        if median_vol == 0:
            return "normal"

        ratio = current_vol / median_vol
        if ratio > 1.5:
            return "high"
        elif ratio < 0.5:
            return "low"
        return "normal"

    def _compute_duration(self, candles: list[dict], current_regime: str) -> int:
        """Zählt wie viele der letzten Kerzen im selben Regime waren."""
        if len(candles) < 2:
            return 1

        # Simplified: check trend direction persistence
        count = 1
        for i in range(len(candles) - 1, 0, -1):
            if i < self.sma_slow:
                break
            sub_candles = candles[: i + 1]
            sub_closes = [c["close"] for c in sub_candles]
            sma_f = _sma(sub_closes, self.sma_fast)
            sma_s = _sma(sub_closes, self.sma_slow)
            if sma_f is None or sma_s is None:
                break
            sub_ratio = abs(sma_f - sma_s) / sma_s if sma_s != 0 else 0
            if sub_ratio < 0.005:
                sub_regime = "range"
            elif sma_f > sma_s:
                sub_regime = "bull"
            else:
                sub_regime = "bear"

            # Map to current regime family
            if current_regime in ("strong_bull", "weak_bull"):
                expected = "bull"
            elif current_regime in ("strong_bear", "weak_bear"):
                expected = "bear"
            else:
                expected = "range"

            if sub_regime == expected:
                count += 1
            else:
                break
        return count
