"""Bollinger-Mean-Reversion-Strategie."""

from __future__ import annotations

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from . import indicators as ta
from ._common import RuleStrategy


class BollingerReversionStrategy(RuleStrategy):
    """Mean Reversion über Bollinger-Bänder.

    BUY: der Schließkurs fällt unter das untere Band und kehrt in der
    aktuellen Bar zurück in den Band-Kanal. SELL: Sprung über das obere
    Band mit Rückkehr in den Kanal. Conviction skaliert mit dem
    ATR14-normierten Ausbruchsabstand.
    """

    strategy_name = "bollinger_reversion"
    description = "Mean Reversion über Bollinger-Bänder (Rückkehr in den Kanal)."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {
        "period": (20.0, 5.0, 50.0),
        "num_std": (2.0, 1.0, 3.5),
    }
    min_bars = 60

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        period = int(self.params["period"])
        num_std = float(self.params["num_std"])
        _, highs, lows, closes, _ = self._arrays()
        _, upper, lower = ta.bollinger(closes, period, num_std)
        prev = float(closes[-2])
        curr = float(closes[-1])
        prev_lower = float(lower[-2])
        curr_lower = float(lower[-1])
        prev_upper = float(upper[-2])
        curr_upper = float(upper[-1])
        atr14 = ta.atr(highs, lows, closes, 14)
        if atr14 <= 0:
            return None
        if prev < prev_lower and curr >= curr_lower:
            strength = min(max(prev_lower - prev, 0.0) / atr14, 1.0)
            return self._signal(candle, SignalAction.BUY, self._conviction(strength), "Rückkehr in den Bollinger-Kanal (unten)")
        if prev > prev_upper and curr <= curr_upper:
            strength = min(max(prev - prev_upper, 0.0) / atr14, 1.0)
            return self._signal(candle, SignalAction.SELL, self._conviction(strength), "Rückkehr in den Bollinger-Kanal (oben)")
        return None
