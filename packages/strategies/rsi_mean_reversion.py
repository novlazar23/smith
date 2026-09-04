"""RSI-Mean-Reversion-Strategie."""

from __future__ import annotations

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from . import indicators as ta
from ._common import RuleStrategy


class RsiMeanReversionStrategy(RuleStrategy):
    """Mean Reversion über Wilder-RSI.

    BUY: RSI steigt aus der Oversold-Zone (unter ``buy_below``) wieder
    über die Schwelle (Wende-Erkennung, nicht Dauer-Signal).
    SELL: RSI fällt aus der Overbought-Zone (über ``sell_above``) wieder
    unter die Schwelle. Conviction skaliert mit der Oversold-/
    Overbought-Tiefe vor der Wende.
    """

    strategy_name = "rsi_mean_reversion"
    description = "Mean Reversion über Wilder-RSI (Wende aus Oversold/Overbought)."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {
        "period": (14.0, 5.0, 50.0),
        "buy_below": (30.0, 5.0, 45.0),
        "sell_above": (70.0, 55.0, 95.0),
    }
    min_bars = 60

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        period = int(self.params["period"])
        buy_below = float(self.params["buy_below"])
        sell_above = float(self.params["sell_above"])
        _, _, _, closes, _ = self._arrays()
        series = ta._rsi_series(closes, period)
        prev = float(series[-2])
        curr = float(series[-1])
        if prev < buy_below <= curr:
            strength = min(max(buy_below - prev, 0.0) / buy_below, 1.0)
            return self._signal(candle, SignalAction.BUY, self._conviction(strength), f"RSI steigt aus Oversold (<{buy_below:g})")
        if prev > sell_above >= curr:
            strength = min(max(curr - sell_above, 0.0) / (100.0 - sell_above), 1.0)
            return self._signal(candle, SignalAction.SELL, self._conviction(strength), f"RSI fällt aus Overbought (>{sell_above:g})")
        return None
