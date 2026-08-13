"""Rule-Based Regime Detector.

Nutzt ADX, RSI, SMA-Crossovers zur Marktregime-Erkennung.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from packages.indicators import ADX, RSI, SMA

from .base import BaseRegimeDetector, MarketRegime, RegimeResult


class RuleBasedRegimeDetector(BaseRegimeDetector):
    """Regime-Detektor auf Basis heuristischer Regeln."""

    name = "rule_based"

    def __init__(
        self,
        adx_bull_threshold: float = 25.0,
        adx_choppy_threshold: float = 20.0,
        rsi_bull_upper: float = 50.0,
        rsi_bear_lower: float = 50.0,
        sma_fast_period: int = 20,
        sma_slow_period: int = 50,
    ) -> None:
        self.adx_bull_threshold = adx_bull_threshold
        self.adx_choppy_threshold = adx_choppy_threshold
        self.rsi_bull_upper = rsi_bull_upper
        self.rsi_bear_lower = rsi_bear_lower
        self.sma_fast_period = sma_fast_period
        self.sma_slow_period = sma_slow_period

    def detect(self, data: dict[str, NDArray[np.float64]]) -> RegimeResult:
        """Erkennt Regime via ADX + RSI + SMA-Kreuzungen.

        Score-Logik:
          - Trendstärke: ADX > bull_threshold → Trend aktiv, sonst Range
          - Richtung: SMA_fast > SMA_slow → Bull, sonst Bear
          - Momentum: RSI > bull_upper → Bull-Verstärkung, RSI < bear_lower → Bear-Verstärkung
        """
        close = data["close"]
        high = data.get("high", close)
        low = data.get("low", close)

        # Kurz-Daten: nicht genug Punkte für Indikatoren → CHOPPY-Fallback
        if len(close) < 60:
            return RegimeResult(
                regime=MarketRegime.CHOPPY,
                confidence=0.0,
                scores={
                    MarketRegime.BULL: 0.0,
                    MarketRegime.BEAR: 0.0,
                    MarketRegime.CHOPPY: 1.0,
                },
            )

        # Indikatoren berechnen
        adx_result = ADX().compute({"high": high, "low": low, "close": close, "volume": np.ones_like(close)})
        rsi_result = RSI().compute({"close": close, "volume": np.ones_like(close)})
        sma_fast_result = SMA(self.sma_fast_period).compute({"close": close, "volume": np.ones_like(close)})
        sma_slow_result = SMA(self.sma_slow_period).compute({"close": close, "volume": np.ones_like(close)})

        adx = adx_result.values
        rsi = rsi_result.values
        sma_fast = sma_fast_result.values
        sma_slow = sma_slow_result.values

        # Nur gültige Indizes betrachten (alle Indikatoren müssen Wert haben)
        valid = ~np.isnan(adx) & ~np.isnan(rsi) & ~np.isnan(sma_fast) & ~np.isnan(sma_slow)
        valid_indices = np.where(valid)[0]

        if len(valid_indices) == 0:
            return RegimeResult(
                regime=MarketRegime.CHOPPY,
                confidence=0.0,
                scores={
                    MarketRegime.BULL: 0.0,
                    MarketRegime.BEAR: 0.0,
                    MarketRegime.CHOPPY: 1.0,
                },
            )

        # Score für letzten gültigen Index berechnen
        idx = valid_indices[-1]
        adx_val = adx[idx]
        rsi_val = rsi[idx]
        is_uptrend = bool(sma_fast[idx] > sma_slow[idx])

        # Bull score
        trend_score = min(adx_val / (self.adx_bull_threshold * 2), 1.0) if adx_val > self.adx_choppy_threshold else 0.0
        direction_score = 1.0 if is_uptrend else 0.0
        momentum_score = max(0.0, (rsi_val - self.rsi_bull_upper) / 50.0) if is_uptrend else max(0.0, (self.rsi_bear_lower - rsi_val) / 50.0)
        bull_score = trend_score * direction_score * (0.6 + 0.4 * max(momentum_score, 0.0))
        bear_score = trend_score * (1.0 - direction_score) * (0.6 + 0.4 * max(1.0 - momentum_score, 0.0))
        choppy_score = 1.0 - trend_score

        # Normalisieren
        total = bull_score + bear_score + choppy_score
        if total > 0:
            bull_score /= total
            bear_score /= total
            choppy_score /= total

        scores = {
            MarketRegime.BULL: float(bull_score),
            MarketRegime.BEAR: float(bear_score),
            MarketRegime.CHOPPY: float(choppy_score),
        }

        best = max(scores, key=scores.get)
        return RegimeResult(
            regime=best,
            confidence=scores[best],
            scores=scores,
            metadata={
                "adx": float(adx_val),
                "rsi": float(rsi_val),
                "sma_fast": float(sma_fast[idx]),
                "sma_slow": float(sma_slow[idx]),
                "is_uptrend": is_uptrend,
            },
        )
