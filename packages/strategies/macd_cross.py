"""MACD-Cross-Strategie (Trendfolge)."""

from __future__ import annotations

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from . import indicators as ta
from ._common import RuleStrategy, StrategyParamError


class MacdCrossStrategy(RuleStrategy):
    """Trendfolge über MACD-Histogramm-Crossover.

    BUY: das Histogramm kreuzt von <= 0 auf > 0 (bullischer Crossover).
    SELL: Kreuzung von >= 0 auf < 0. Conviction skaliert mit dem
    ATR14-normierten Histogramm-Abstand.
    """

    strategy_name = "macd_cross"
    description = "Trendfolge über MACD-Histogramm-Crossover."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {
        "fast": (12.0, 5.0, 30.0),
        "slow": (26.0, 15.0, 60.0),
        "signal": (9.0, 3.0, 30.0),
    }
    min_bars = 130

    def _check_params(self, resolved: dict[str, float]) -> None:
        if resolved["fast"] >= resolved["slow"]:
            raise StrategyParamError("macd_cross: fast muss kleiner als slow sein")

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        fast = int(self.params["fast"])
        slow = int(self.params["slow"])
        signal = int(self.params["signal"])
        _, highs, lows, closes, _ = self._arrays()
        _, _, hist = ta.macd(closes, fast, slow, signal)
        prev = float(hist[-2])
        curr = float(hist[-1])
        atr14 = ta.atr(highs, lows, closes, 14)
        if atr14 <= 0:
            return None
        strength = min(abs(curr) / atr14, 1.0)
        if prev <= 0 < curr:
            return self._signal(candle, SignalAction.BUY, self._conviction(strength), "MACD bullischer Crossover")
        if prev >= 0 > curr:
            return self._signal(candle, SignalAction.SELL, self._conviction(strength), "MACD bärischer Crossover")
        return None
