"""EMA-Cross-Strategie (Trendfolge)."""

from __future__ import annotations

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from . import indicators as ta
from ._common import RuleStrategy, StrategyParamError


class EmaCrossStrategy(RuleStrategy):
    """Trendfolge über EMA-Crossover (schnell/langsam).

    BUY: die schnelle EMA kreuzt die langsame von unten nach oben.
    SELL: Kreuzung von oben nach unten. Conviction skaliert mit dem
    ATR14-normierten EMA-Abstand.
    """

    strategy_name = "ema_cross"
    description = "Trendfolge über EMA-Crossover (schnell/langsam)."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {
        "fast": (12.0, 5.0, 30.0),
        "slow": (26.0, 15.0, 100.0),
    }
    min_bars = 110

    def _check_params(self, resolved: dict[str, float]) -> None:
        if resolved["fast"] >= resolved["slow"]:
            raise StrategyParamError("ema_cross: fast muss kleiner als slow sein")

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        fast = int(self.params["fast"])
        slow = int(self.params["slow"])
        _, highs, lows, closes, _ = self._arrays()
        fast_series = ta.ema(closes, fast)
        slow_series = ta.ema(closes, slow)
        prev_diff = fast_series[-2] - slow_series[-2]
        curr_diff = fast_series[-1] - slow_series[-1]
        atr14 = ta.atr(highs, lows, closes, 14)
        if atr14 <= 0:
            return None
        strength = min(abs(curr_diff) / atr14, 1.0)
        if prev_diff <= 0 < curr_diff:
            return self._signal(candle, SignalAction.BUY, self._conviction(strength), f"EMA{fast} kreuzt EMA{slow} aufwärts")
        if prev_diff >= 0 > curr_diff:
            return self._signal(candle, SignalAction.SELL, self._conviction(strength), f"EMA{fast} kreuzt EMA{slow} abwärts")
        return None
