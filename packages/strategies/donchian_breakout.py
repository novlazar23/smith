"""Donchian-Breakout-Strategie (Turtle-Style)."""

from __future__ import annotations

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from . import indicators as ta
from ._common import RuleStrategy, StrategyParamError


class DonchianBreakoutStrategy(RuleStrategy):
    """Breakout über Donchian-Kanäle (Turtle-Style).

    BUY: Schließkurs bricht den Kanal-Hochstand der ``entry_period``
    Vor-Kerzen (ohne die aktuelle Bar) nach oben. SELL: Bruch des
    Kanal-Tiefs der ``exit_period`` Vor-Kerzen nach unten. Solange die
    Bedingung gilt, wiederholt die Strategie das Signal (der Motor
    öffnet nur bei flacher Position).
    """

    strategy_name = "donchian_breakout"
    description = "Breakout über Donchian-Kanäle (Turtle-Style)."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {
        "entry_period": (20.0, 10.0, 50.0),
        "exit_period": (40.0, 10.0, 100.0),
    }
    min_bars = 110

    def _check_params(self, resolved: dict[str, float]) -> None:
        if resolved["exit_period"] < resolved["entry_period"]:
            raise StrategyParamError("donchian_breakout: exit_period muss >= entry_period sein")

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        entry = int(self.params["entry_period"])
        exit_period = int(self.params["exit_period"])
        _, highs, lows, closes, _ = self._arrays()
        if len(closes) <= max(entry, exit_period):
            return None
        entry_high_prev = float(highs[-entry - 1 : -1].max())
        exit_low_prev = float(lows[-exit_period - 1 : -1].min())
        atr14 = ta.atr(highs, lows, closes, 14)
        if atr14 <= 0:
            return None
        close = float(closes[-1])
        if close > entry_high_prev:
            strength = min((close - entry_high_prev) / atr14, 1.0)
            return self._signal(candle, SignalAction.BUY, self._conviction(strength), f"Breakout über {entry}Bar-Donchian-Hoch")
        if close < exit_low_prev:
            strength = min((exit_low_prev - close) / atr14, 1.0)
            return self._signal(candle, SignalAction.SELL, self._conviction(strength), f"Bruch des {exit_period}Bar-Donchian-Tiefs")
        return None
