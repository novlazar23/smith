"""Supertrend-Strategie (Trendfolge)."""

from __future__ import annotations

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from . import indicators as ta
from ._common import RuleStrategy


class SupertrendStrategy(RuleStrategy):
    """Trendfolge über Supertrend-Wende.

    BUY: Supertrend wechselt von Abwärts (-1) auf Aufwärts (+1).
    SELL: Wechsel von Aufwärts auf Abwärts. Conviction skaliert mit dem
    ATR14-normierten Abstand Schließkurs zur Trendlinie.
    """

    strategy_name = "supertrend"
    description = "Trendfolge über Supertrend-Wende."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {
        "period": (10.0, 5.0, 50.0),
        "multiplier": (3.0, 1.0, 5.0),
    }
    min_bars = 60

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        period = int(self.params["period"])
        multiplier = float(self.params["multiplier"])
        _, highs, lows, closes, _ = self._arrays()
        trend_line, direction = ta.supertrend(highs, lows, closes, period, multiplier)
        prev_dir = int(direction[-2])
        curr_dir = int(direction[-1])
        atr14 = ta.atr(highs, lows, closes, 14)
        if atr14 <= 0:
            return None
        strength = min(abs(float(closes[-1]) - float(trend_line[-1])) / atr14, 1.0)
        if prev_dir < 0 and curr_dir > 0:
            return self._signal(candle, SignalAction.BUY, self._conviction(strength), "Supertrend dreht aufwärts")
        if prev_dir > 0 and curr_dir < 0:
            return self._signal(candle, SignalAction.SELL, self._conviction(strength), "Supertrend dreht abwärts")
        return None
