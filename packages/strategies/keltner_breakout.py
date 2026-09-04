"""Keltner-Breakout-Strategie."""

from __future__ import annotations

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from . import indicators as ta
from ._common import RuleStrategy


class KeltnerBreakoutStrategy(RuleStrategy):
    """Breakout über Keltner-Kanäle (EMA-Mittel, ATR-Bänder).

    BUY: der Schließkurs bricht das obere Band nach oben (Kreuzung).
    SELL: Bruch des unteren Bandes nach unten. Conviction skaliert mit
    dem ATR14-normierten Bandabstand.
    """

    strategy_name = "keltner_breakout"
    description = "Breakout über Keltner-Kanäle (EMA ± ATR)."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {
        "period": (20.0, 5.0, 50.0),
        "multiplier": (2.0, 1.0, 4.0),
    }
    min_bars = 60

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        period = int(self.params["period"])
        multiplier = float(self.params["multiplier"])
        _, highs, lows, closes, _ = self._arrays()
        _, upper, lower = ta.keltner(closes, highs, lows, period, multiplier)
        prev = float(closes[-2])
        curr = float(closes[-1])
        prev_upper = float(upper[-2])
        curr_upper = float(upper[-1])
        prev_lower = float(lower[-2])
        curr_lower = float(lower[-1])
        atr14 = ta.atr(highs, lows, closes, 14)
        if atr14 <= 0:
            return None
        if prev <= prev_upper and curr > curr_upper:
            strength = min(max(curr - curr_upper, 0.0) / atr14, 1.0)
            return self._signal(candle, SignalAction.BUY, self._conviction(strength), "Breakout über Keltner-Oberband")
        if prev >= prev_lower and curr < curr_lower:
            strength = min(max(prev_lower - prev, 0.0) / atr14, 1.0)
            return self._signal(candle, SignalAction.SELL, self._conviction(strength), "Bruch des Keltner-Unterbands")
        return None
