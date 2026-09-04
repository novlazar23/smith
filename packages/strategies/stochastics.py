"""Stochastik-Strategie (Mean Reversion / Wende)."""

from __future__ import annotations

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from . import indicators as ta
from ._common import RuleStrategy


class StochasticsStrategy(RuleStrategy):
    """Mean Reversion über Stochastik %K/%D.

    BUY: %K steigt aus der Oversold-Zone (unter ``buy_below``) und
    kreuzt %D von unten nach oben. SELL: %K fällt aus der Overbought-Zone
    (über ``sell_above``) und kreuzt %D von oben nach unten.
    """

    strategy_name = "stochastics"
    description = "Mean Reversion über Stochastik %K/%D-Crossover."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {
        "k": (14.0, 5.0, 50.0),
        "d": (3.0, 2.0, 10.0),
        "buy_below": (20.0, 5.0, 40.0),
        "sell_above": (80.0, 60.0, 95.0),
    }
    min_bars = 60

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        k = int(self.params["k"])
        d = int(self.params["d"])
        buy_below = float(self.params["buy_below"])
        sell_above = float(self.params["sell_above"])
        _, highs, lows, closes, _ = self._arrays()
        k_vals, d_vals = ta.stochastic(highs, lows, closes, k, d)
        k_prev, k_curr = float(k_vals[-2]), float(k_vals[-1])
        d_prev, d_curr = float(d_vals[-2]), float(d_vals[-1])
        if k_prev < buy_below and k_prev <= d_prev and k_curr > d_curr:
            strength = min(max(buy_below - k_prev, 0.0) / buy_below, 1.0)
            return self._signal(candle, SignalAction.BUY, self._conviction(strength), f"Stochastik %K/%D-Crossover aus Oversold (<{buy_below:g})")
        if k_prev > sell_above and k_prev >= d_prev and k_curr < d_curr:
            strength = min(max(k_prev - sell_above, 0.0) / (100.0 - sell_above), 1.0)
            return self._signal(candle, SignalAction.SELL, self._conviction(strength), f"Stochastik %K/%D-Crossover aus Overbought (>{sell_above:g})")
        return None
